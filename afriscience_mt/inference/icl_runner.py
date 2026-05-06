"""In-context learning experiment runner."""

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from tqdm import tqdm

from ..data.dataset import AfriScienceDataset
from ..data.glossary import Glossary, load_glossary
from ..api.base import APIClient
from ..api.factory import create_api_client
from .prompt_builder import PromptBuilder, build_translation_prompt
from .example_selector import ExampleSelector, create_example_selector
from ..utils.logging import get_logger

logger = get_logger(__name__)


def filter_glossary_for_context(
    glossary: Glossary,
    source_text: str,
    examples: Optional[List[Dict[str, str]]] = None,
    max_terms: int = 15,
) -> str:
    """
    Filter glossary to only include terms found in source text and examples.

    This is a best practice to avoid overwhelming the model with irrelevant terms.

    Args:
        glossary: Full glossary object
        source_text: Source text to translate
        examples: Optional few-shot examples
        max_terms: Maximum number of terms to include

    Returns:
        Formatted glossary string with only relevant terms
    """
    # Build combined text to search
    text_to_search = source_text.lower()
    if examples:
        for ex in examples:
            text_to_search += " " + ex.get("source", "").lower()

    # Find matching terms
    matched_terms = []
    for src_term, tgt_term in glossary.terms.items():
        # Use word boundary matching for short terms
        if len(src_term) <= 3:
            if re.search(r'\b' + re.escape(src_term) + r'\b', text_to_search):
                matched_terms.append((src_term, tgt_term, len(src_term)))
        else:
            if src_term in text_to_search:
                matched_terms.append((src_term, tgt_term, len(src_term)))

    # Sort by term length (longer first for specificity)
    matched_terms.sort(key=lambda x: x[2], reverse=True)

    # Format terms
    formatted = []
    for src, tgt, _ in matched_terms[:max_terms]:
        formatted.append(f"{src} → {tgt}")

    return "\n".join(formatted)


class ICLRunner:
    """Runner for in-context learning translation experiments."""

    def __init__(
        self,
        api_client: APIClient,
        train_dataset: Optional[AfriScienceDataset] = None,
        example_selector: Optional[ExampleSelector] = None,
        glossary: Optional[Glossary] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        prepared_examples_path: Optional[Path] = None,
        selection_method: str = "random",
        filter_glossary: bool = True,
    ):
        """
        Initialize ICL runner.

        Args:
            api_client: API client for model inference
            train_dataset: Training data for example selection
            example_selector: Example selector (created from train_dataset if None)
            glossary: Glossary for terminology
            prompt_builder: Prompt builder (uses default if None)
            prepared_examples_path: Path to pre-computed semantic examples (from prepare_icl_data.py)
            selection_method: Example selection method ("random", "semantic", "domain_balanced")
            filter_glossary: Whether to filter glossary to only relevant terms (recommended)
        """
        self.api_client = api_client
        self.train_dataset = train_dataset
        self.glossary = glossary
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.filter_glossary = filter_glossary
        self.selection_method = selection_method

        # Load pre-computed semantic examples if available
        self.prepared_examples = None
        if prepared_examples_path and Path(prepared_examples_path).exists():
            self._load_prepared_examples(prepared_examples_path)
            logger.info(f"Loaded pre-computed semantic examples from {prepared_examples_path}")

        # Create example selector if needed
        if example_selector is not None:
            self.example_selector = example_selector
        elif train_dataset is not None:
            self.example_selector = create_example_selector(
                method=selection_method,
                dataset=train_dataset,
            )
        else:
            self.example_selector = None

    def _load_prepared_examples(self, path: Path) -> None:
        """Load pre-computed semantic examples from prepare_icl_data.py output."""
        path = Path(path)
        if path.is_dir():
            path = path / "semantic_examples.json"

        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                self.prepared_examples = json.load(f)
            logger.info(f"Loaded {len(self.prepared_examples.get('examples', {}))} pre-computed example sets")

    def _get_prepared_examples(self, test_idx: int, n_examples: int) -> Optional[List[Dict[str, str]]]:
        """Get pre-computed examples for a test sample."""
        if self.prepared_examples is None:
            return None

        examples_data = self.prepared_examples.get("examples", {})
        sample_data = examples_data.get(str(test_idx))

        if sample_data is None:
            return None

        similar_examples = sample_data.get("similar_examples", [])
        return similar_examples[:n_examples]

    def translate_single(
        self,
        source_text: str,
        source_lang: str,
        target_lang: str,
        n_shots: int = 0,
        template: str = "basic",
        use_glossary: bool = False,
        selection_method: str = "random",
        test_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Translate a single text with ICL.

        Args:
            source_text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            n_shots: Number of examples (0 for zero-shot)
            template: Prompt template name
            use_glossary: Whether to include glossary
            selection_method: Example selection method
            test_idx: Index in test set (for using pre-computed examples)

        Returns:
            Dict with translation and metadata
        """
        # Select examples if needed
        examples = None
        examples_source = None

        if n_shots > 0:
            # Try to use pre-computed semantic examples first
            if (selection_method == "semantic" and
                self.prepared_examples is not None and
                test_idx is not None):
                examples = self._get_prepared_examples(test_idx, n_shots)
                if examples:
                    examples_source = "prepared_semantic"
                    logger.debug(f"Using {len(examples)} pre-computed semantic examples for test_idx={test_idx}")

            # Fall back to selector if no prepared examples
            if examples is None and self.example_selector is not None:
                examples = self.example_selector.select(
                    query=source_text,
                    n_examples=n_shots,
                )
                examples_source = selection_method

        # Get glossary terms if needed - now with filtering!
        glossary_str = None
        glossary_terms_count = 0

        if use_glossary and self.glossary is not None:
            if self.filter_glossary:
                # Filter glossary to only terms in source text AND examples
                glossary_str = filter_glossary_for_context(
                    glossary=self.glossary,
                    source_text=source_text,
                    examples=examples,
                    max_terms=15,
                )
                glossary_terms_count = len(glossary_str.split("\n")) if glossary_str else 0
            else:
                # Use all glossary terms (not recommended)
                glossary_str = self.glossary.format_for_prompt(max_terms=20)
                glossary_terms_count = len(self.glossary)

        # Build prompt
        if n_shots > 0:
            template_to_use = "few_shot_glossary" if use_glossary else "few_shot"
        else:
            template_to_use = "glossary_augmented" if use_glossary else template

        prompt = self.prompt_builder.build_prompt(
            source_text=source_text,
            source_lang=source_lang,
            target_lang=target_lang,
            template_name=template_to_use,
            examples=examples,
            glossary=glossary_str,
        )

        # Call API
        system_prompt = f"You are a professional translator for scientific texts."
        translation = self.api_client.generate(prompt, system_prompt)

        return {
            "source": source_text,
            "translation": translation,
            "n_shots": n_shots,
            "template": template_to_use,
            "used_glossary": use_glossary,
            "glossary_terms_used": glossary_terms_count,
            "examples_source": examples_source,
            "prompt": prompt,
        }

    def translate_batch(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        n_shots: int = 0,
        template: str = "basic",
        use_glossary: bool = False,
        show_progress: bool = True,
        delay: float = 0.5,
        selection_method: str = "random",
    ) -> List[Dict[str, Any]]:
        """
        Translate a batch of texts.

        Args:
            texts: Texts to translate
            source_lang: Source language code
            target_lang: Target language code
            n_shots: Number of examples
            template: Prompt template
            use_glossary: Whether to use glossary
            show_progress: Show progress bar
            delay: Delay between requests
            selection_method: Example selection method

        Returns:
            List of translation results
        """
        results = []

        iterator = enumerate(texts)
        if show_progress:
            iterator = tqdm(list(iterator), desc=f"ICL ({n_shots}-shot, {selection_method})")

        for idx, text in iterator:
            result = self.translate_single(
                source_text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                n_shots=n_shots,
                template=template,
                use_glossary=use_glossary,
                selection_method=selection_method,
                test_idx=idx,  # Pass test index for pre-computed examples
            )
            results.append(result)

            if delay > 0:
                time.sleep(delay)

        return results

    def run_experiment(
        self,
        test_dataset: AfriScienceDataset,
        shot_counts: List[int] = [0, 5, 10, 20],
        templates: List[str] = ["basic"],
        use_glossary_variants: List[bool] = [False],
        selection_methods: List[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Run a full ICL experiment with multiple configurations.

        Args:
            test_dataset: Test dataset
            shot_counts: List of shot counts to try
            templates: List of template names
            use_glossary_variants: Whether to test with/without glossary
            selection_methods: List of example selection methods to test
            output_dir: Directory to save results

        Returns:
            Experiment results
        """
        source_lang = test_dataset.source_lang
        target_lang = test_dataset.target_lang
        sources = test_dataset.get_sources()
        references = test_dataset.get_targets()

        # Default selection method
        if selection_methods is None:
            selection_methods = [self.selection_method]

        results = {
            "source_lang": source_lang,
            "target_lang": target_lang,
            "num_samples": len(sources),
            "selection_methods": selection_methods,
            "filter_glossary": self.filter_glossary,
            "configurations": [],
        }

        for n_shots in shot_counts:
            for template in templates:
                for use_glossary in use_glossary_variants:
                    # For 0-shot, selection method doesn't matter
                    methods_to_test = selection_methods if n_shots > 0 else [selection_methods[0]]

                    for sel_method in methods_to_test:
                        config_name = f"{n_shots}shot_{template}"
                        if n_shots > 0:
                            config_name += f"_{sel_method}"
                        if use_glossary:
                            config_name += "_glossary"

                        logger.info(f"Running configuration: {config_name}")

                        start_time = time.time()

                        translations = self.translate_batch(
                            texts=sources,
                            source_lang=source_lang,
                            target_lang=target_lang,
                            n_shots=n_shots,
                            template=template,
                            use_glossary=use_glossary,
                            selection_method=sel_method,
                        )

                        elapsed = time.time() - start_time

                        # Compute stats
                        glossary_terms_stats = [t.get("glossary_terms_used", 0) for t in translations]
                        avg_glossary_terms = sum(glossary_terms_stats) / len(glossary_terms_stats) if glossary_terms_stats else 0

                        config_result = {
                            "name": config_name,
                            "n_shots": n_shots,
                            "template": template,
                            "use_glossary": use_glossary,
                            "selection_method": sel_method if n_shots > 0 else None,
                            "avg_glossary_terms": round(avg_glossary_terms, 2),
                            "hypotheses": [t["translation"] for t in translations],
                            "sources": sources,
                            "references": references,
                            "runtime_seconds": elapsed,
                        }

                        results["configurations"].append(config_result)

                        # Save intermediate results
                        if output_dir:
                            self._save_config_results(config_result, output_dir, config_name)

        if output_dir:
            self._save_experiment_results(results, output_dir)

        return results

    def _save_config_results(
        self,
        config_result: Dict,
        output_dir: Union[str, Path],
        config_name: str,
    ) -> None:
        """Save results for a single configuration."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        path = output_dir / f"{config_name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config_result, f, ensure_ascii=False, indent=2)

    def _save_experiment_results(
        self,
        results: Dict,
        output_dir: Union[str, Path],
    ) -> None:
        """Save full experiment results."""
        output_dir = Path(output_dir)
        path = output_dir / "experiment_results.json"

        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"Experiment results saved to {path}")


def run_icl_experiment(
    provider: str,
    model_name: str,
    test_data_path: Union[str, Path],
    source_lang: str,
    target_lang: str,
    train_data_path: Optional[Union[str, Path]] = None,
    glossary_path: Optional[Union[str, Path]] = None,
    shot_counts: List[int] = [0, 5, 10, 20],
    output_dir: Optional[Union[str, Path]] = None,
    api_key: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function to run an ICL experiment.

    Args:
        provider: API provider ("openai", "google", "together")
        model_name: Model name
        test_data_path: Path to test data
        source_lang: Source language code
        target_lang: Target language code
        train_data_path: Path to training data (for examples)
        glossary_path: Path to glossary
        shot_counts: Shot counts to test
        output_dir: Output directory
        api_key: API key
        **kwargs: Additional arguments

    Returns:
        Experiment results
    """
    # Create API client
    api_client = create_api_client(
        provider=provider,
        model_name=model_name,
        api_key=api_key,
    )

    # Load test data
    test_dataset = AfriScienceDataset(
        data_path=test_data_path,
        source_lang=source_lang,
        target_lang=target_lang,
        split="test",
    )

    # Load training data for examples
    train_dataset = None
    if train_data_path:
        train_dataset = AfriScienceDataset(
            data_path=train_data_path,
            source_lang=source_lang,
            target_lang=target_lang,
            split="train",
        )

    # Load glossary
    glossary = None
    if glossary_path:
        glossary = load_glossary(glossary_path, target_lang)

    # Create runner
    runner = ICLRunner(
        api_client=api_client,
        train_dataset=train_dataset,
        glossary=glossary,
    )

    # Run experiment
    return runner.run_experiment(
        test_dataset=test_dataset,
        shot_counts=shot_counts,
        output_dir=output_dir,
        **kwargs,
    )
