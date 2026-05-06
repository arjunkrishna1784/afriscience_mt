"""Prompt building utilities for translation."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..utils.config import load_prompt_templates
from ..utils.language_utils import get_language_name
from ..data.glossary import Glossary


class PromptBuilder:
    """Builder for translation prompts."""

    def __init__(
        self,
        template_type: str = "sentence",
        config_dir: Optional[Path] = None,
    ):
        """
        Initialize prompt builder.

        Args:
            template_type: Type of templates ("sentence" or "document")
            config_dir: Directory containing config files
        """
        self.template_type = template_type
        self.templates = load_prompt_templates(template_type, config_dir)

    def build_prompt(
        self,
        source_text: str,
        source_lang: str,
        target_lang: str,
        template_name: str = "basic",
        examples: Optional[List[Dict[str, str]]] = None,
        glossary: Optional[Union[Glossary, str]] = None,
        domain: Optional[str] = None,
        **kwargs,
    ) -> str:
        source_name = get_language_name(source_lang)
        target_name = get_language_name(target_lang)

        template_config = self.templates.templates.get(template_name)
        if template_config is None:
            raise ValueError(f"Unknown template: {template_name}")

        template = template_config.template

        # FIX: Include both standard and "full" keys to match your YAML
        variables = {
            "source_language": source_name,
            "target_language": target_name,
            "source_language_full": source_name,
            "target_language_full": target_name,
            "source_text": source_text,
            "domain": domain or "scientific",
            **kwargs,
        }

        # Format examples if provided
        if examples:
            example_format = template_config.get("example_format", "{source_language_full}: {src}\n{target_language_full}: {tgt}")
            formatted_examples = []
            for ex in examples:
                # We use .format(**variables, src=..., tgt=...) so the example_format 
                # can also use {source_language_full} if it wants to.
                formatted_examples.append(
                    example_format.format(**variables, src=ex["source"], tgt=ex["target"])
                )
            variables["examples"] = "\n\n".join(formatted_examples)

        # Format glossary if provided
        if glossary:
            if isinstance(glossary, Glossary):
                glossary_str = glossary.format_for_prompt(text=source_text)
            else:
                glossary_str = glossary
            variables["glossary"] = glossary_str

        # Fill template with safety check
        try:
            return template.format(**variables).strip()
        except KeyError as e:
            missing_key = str(e).strip("'")
            print(f"Variable '{missing_key}' missing in build_prompt for template '{template_name}'")
            # Fallback: Fill missing key with empty string so it doesn't crash
            variables[missing_key] = ""
            return template.format(**variables).strip()

    def build_chat_prompt(
        self,
        source_text: str,
        source_lang: str,
        target_lang: str,
        template_name: str = "basic",
        examples: Optional[List[Dict[str, str]]] = None,
        glossary: Optional[Union[Glossary, str]] = None,
        domain: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, str]:
        """
        Build prompt in chat format (system + user) with full variable mapping.
        """
        source_name = get_language_name(source_lang)
        target_name = get_language_name(target_lang)

        # Retrieve the chat templates from the YAML config
        chat_templates = self.templates.get("chat_templates", {})
        chat_config = chat_templates.get(template_name, {})

        # Default system message if none is found in the template
        system_template = chat_config.get(
            "system",
            f"You are a professional translator specializing in scientific texts from {source_name} to {target_name}."
        )
        user_template = chat_config.get("user", "{source_text}")

        # Build variables mapping (matching the YAML keys)
        variables = {
            "source_language": source_name,
            "target_language": target_name,
            "source_language_full": source_name,
            "target_language_full": target_name,
            "source_text": source_text,
            "domain": domain or "scientific",
            **kwargs,
        }

        # Format examples if they exist
        if examples:
            formatted_examples = []
            for ex in examples:
                formatted_examples.append(f"{source_name}: {ex['source']}\n{target_name}: {ex['target']}")
            variables["examples"] = "\n\n".join(formatted_examples)

        # Format glossary if it exists
        if glossary:
            if isinstance(glossary, Glossary):
                variables["glossary"] = glossary.format_for_prompt(text=source_text)
            else:
                variables["glossary"] = glossary

        # Format the System and User strings safely
        def safe_format(template_str, vars_dict):
            try:
                return template_str.format(**vars_dict)
            except KeyError as e:
                # If a key like {domain} is in the YAML but not in variables, 
                # we fill it with an empty string to avoid the crash.
                missing_key = str(e).strip("'")
                vars_dict[missing_key] = ""
                return template_str.format(**vars_dict)

        system_message = safe_format(system_template, variables)
        user_message = safe_format(user_template, variables)

        return {
            "system": system_message.strip(),
            "user": user_message.strip()
        }


def build_translation_prompt(
    source_text: str,
    source_lang: str,
    target_lang: str,
    template: str = "basic",
    examples: Optional[List[Dict[str, str]]] = None,
    glossary: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Convenience function to build a translation prompt.

    Args:
        source_text: Text to translate
        source_lang: Source language code
        target_lang: Target language code
        template: Template name or custom template string
        examples: Optional list of examples
        glossary: Optional glossary string
        **kwargs: Additional template variables

    Returns:
        Formatted prompt
    """
    source_name = get_language_name(source_lang)
    target_name = get_language_name(target_lang)

    # Check if template is a custom string
    if "{source_text}" in template or "{source_language}" in template:
        # Custom template string
        variables = {
            "source_language": source_name,
            "target_language": target_name,
            "source_text": source_text,
            "glossary": glossary or "",
            "examples": "",
            **kwargs,
        }

        if examples:
            formatted = []
            for ex in examples:
                formatted.append(f"Source: {ex['source']}\nTranslation: {ex['target']}")
            variables["examples"] = "\n\n".join(formatted)

        return template.format(**variables).strip()

    # Use template builder
    builder = PromptBuilder(template_type="sentence")
    return builder.build_prompt(
        source_text=source_text,
        source_lang=source_lang,
        target_lang=target_lang,
        template_name=template,
        examples=examples,
        glossary=glossary,
        **kwargs,
    )


def build_document_prompt(
    source_document: str,
    source_lang: str,
    target_lang: str,
    template: str = "basic",
    glossary: Optional[str] = None,
    domain: Optional[str] = None,
) -> str:
    """
    Build a document-level translation prompt.

    Args:
        source_document: Document to translate
        source_lang: Source language code
        target_lang: Target language code
        template: Template name
        glossary: Optional glossary string
        domain: Scientific domain

    Returns:
        Formatted prompt
    """
    builder = PromptBuilder(template_type="document")
    return builder.build_prompt(
        source_text=source_document,
        source_lang=source_lang,
        target_lang=target_lang,
        template_name=template,
        glossary=glossary,
        domain=domain,
        source_document=source_document,
    )
