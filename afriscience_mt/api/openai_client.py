"""
OpenAI API client for GPT models (GPT-5, o3, GPT-4.1).
Updated for 2026 standards with full translation evaluation suites.
"""

import asyncio
import os
import re
from typing import Any, Dict, List, Optional


def _reasoning_effort_default() -> str:
    return os.environ.get("OPENAI_REASONING_EFFORT", "medium")

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from .base import APIClient, APIConfig

# Max concurrent OpenAI requests. GPT-5/o3 tier-2 allows 5000 RPM, but
# conservative concurrency avoids 429s while still giving ~5× throughput.
_OPENAI_CONCURRENCY = 5


class OpenAIClient(APIClient):
    """Client for OpenAI API (GPT-5, o3, GPT-4o-mini, etc.)."""

    def _initialize_client(self) -> None:
        """Initialize sync and async OpenAI clients."""
        try:
            from openai import OpenAI, AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai package not installed. "
                "Install with: pip install openai"
            )

        self.client = OpenAI(api_key=self.api_key)
        self._async_client = AsyncOpenAI(api_key=self.api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type(Exception),
    )
    def _call_api(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> str:
        """
        Handles parameter routing for 2026 Reasoning (GPT-5/o3) and Legacy models.
        """
        if self.client is None:
            self._initialize_client()

        model_name = self.config.model_name.lower()
        is_reasoning_model = any(m in model_name for m in ["gpt-5", "o1", "o3"])

        # 1. Role Migration: 'system' -> 'developer' (GPT-5 standard)
        formatted_messages = []
        for msg in messages:
            role = msg["role"]
            if role == "system" and is_reasoning_model:
                role = "developer"
            formatted_messages.append({"role": role, "content": msg["content"]})

        generation_kwargs = {
            "model": self.config.model_name,
            "messages": formatted_messages,
        }

        # 2. Modern Parameter Handling
        if is_reasoning_model:
            # max_completion_tokens accounts for internal reasoning tokens
            generation_kwargs["max_completion_tokens"] = kwargs.get("max_tokens", self.config.max_tokens)
            # reasoning_effort is the 2026 standard for intelligence control
            generation_kwargs["reasoning_effort"] = kwargs.get("reasoning_effort", _reasoning_effort_default())
        else:
            generation_kwargs["max_tokens"] = kwargs.get("max_tokens", self.config.max_tokens)
            generation_kwargs["temperature"] = kwargs.get("temperature", self.config.temperature)
            generation_kwargs["top_p"] = kwargs.get("top_p", self.config.top_p)

        response = self.client.chat.completions.create(**generation_kwargs)
        return response.choices[0].message.content

    # ------------------------------------------------------------------
    # Async helper — used by generate_batch
    # ------------------------------------------------------------------

    async def _generate_async(
        self,
        prompt: str,
        semaphore: asyncio.Semaphore,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Single async generation call, rate-limited by semaphore."""
        async with semaphore:
            model_name = self.config.model_name.lower()
            is_reasoning = any(m in model_name for m in ["gpt-5", "o1", "o3"])

            messages: List[Dict[str, str]] = []
            if system_prompt:
                role = "developer" if is_reasoning else "system"
                messages.append({"role": role, "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            gen_kwargs: Dict[str, Any] = {
                "model": self.config.model_name,
                "messages": messages,
            }
            if is_reasoning:
                gen_kwargs["max_completion_tokens"] = self.config.max_tokens
                gen_kwargs["reasoning_effort"] = _reasoning_effort_default()
            else:
                gen_kwargs["max_tokens"] = self.config.max_tokens
                gen_kwargs["temperature"] = self.config.temperature
                gen_kwargs["top_p"] = self.config.top_p

            response = await self._async_client.chat.completions.create(**gen_kwargs)
            return response.choices[0].message.content

    # ------------------------------------------------------------------
    # Batch — overrides base to run concurrently instead of sequentially
    # ------------------------------------------------------------------

    def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> List[str]:
        """
        Generate responses for all prompts concurrently.

        Uses asyncio.gather with a semaphore so at most
        ``_OPENAI_CONCURRENCY`` requests are in-flight at once, giving
        ~5× throughput over sequential calls while respecting rate limits.
        """
        if self.client is None:
            self._initialize_client()

        async def _run_all() -> List[str]:
            sem = asyncio.Semaphore(_OPENAI_CONCURRENCY)
            tasks = [self._generate_async(p, sem, system_prompt) for p in prompts]
            return await asyncio.gather(*tasks)

        return asyncio.run(_run_all())

    # ------------------------------------------------------------------
    # True Batch API — submit-and-retrieve (50% cheaper, no rate limits)
    # ------------------------------------------------------------------

    def submit_batch_job(
        self,
        prompts: List[str],
        job_name: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Submit all prompts as a single OpenAI Batch API job.

        Returns the batch job ID — store it and call retrieve_batch_results()
        later (minutes to a few hours) to get the outputs.

        Cost: 50% of real-time pricing.
        Limit: up to 50,000 requests or 200 MB per batch.
        Completion window: up to 24 h (usually much faster).
        """
        import io
        import json as _json

        if self.client is None:
            self._initialize_client()

        model_name = self.config.model_name.lower()
        is_reasoning = any(m in model_name for m in ["gpt-5", "o1", "o3"])

        lines: List[str] = []
        for i, prompt in enumerate(prompts):
            messages: List[Dict[str, str]] = []
            if system_prompt:
                role = "developer" if is_reasoning else "system"
                messages.append({"role": role, "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            body: Dict[str, Any] = {
                "model": self.config.model_name,
                "messages": messages,
            }
            if is_reasoning:
                body["max_completion_tokens"] = self.config.max_tokens
                body["reasoning_effort"] = _reasoning_effort_default()
            else:
                body["max_tokens"] = self.config.max_tokens
                body["temperature"] = self.config.temperature
                body["top_p"] = self.config.top_p

            lines.append(_json.dumps({
                "custom_id": f"{job_name}-{i:05d}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }))

        jsonl_bytes = "\n".join(lines).encode("utf-8")

        # Upload JSONL file
        uploaded = self.client.files.create(
            file=(f"{job_name}.jsonl", io.BytesIO(jsonl_bytes)),
            purpose="batch",
        )

        # Create the batch job
        batch = self.client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        return batch.id

    def cancel_batch_job(self, batch_id: str) -> None:
        """Cancel a running OpenAI batch job."""
        if self.client is None:
            self._initialize_client()
        self.client.batches.cancel(batch_id)

    def retrieve_batch_results(
        self,
        batch_id: str,
    ) -> Optional[List[str]]:
        """
        Poll an OpenAI Batch job and return results when complete.

        Returns:
            Ordered list of response strings (same order as submission) if done.
            None if the job is still queued / in_progress.
        Raises:
            RuntimeError if the batch failed, was cancelled, or expired.
        """
        import json as _json

        if self.client is None:
            self._initialize_client()

        batch = self.client.batches.retrieve(batch_id)

        if batch.status in ("failed", "cancelled", "expired"):
            raise RuntimeError(
                f"OpenAI batch {batch_id} ended with status '{batch.status}'. "
                f"Errors: {getattr(batch, 'errors', None)}"
            )

        if batch.status != "completed":
            counts = batch.request_counts
            return None  # Still queued / in_progress

        # Download output JSONL and re-sort by the index in custom_id
        content = self.client.files.content(batch.output_file_id).text
        raw: Dict[str, str] = {}
        error_sample: List[str] = []   # collect a few error messages for diagnosis
        for line in content.strip().splitlines():
            if not line:
                continue
            item = _json.loads(line)
            custom_id = item["custom_id"]

            # Top-level error (e.g. invalid request, model unavailable)
            if item.get("error"):
                msg = str(item["error"])
                if len(error_sample) < 3:
                    error_sample.append(f"{custom_id}: {msg}")
                raw[custom_id] = ""
                continue

            # Response-level error (e.g. content filter, context length exceeded)
            resp_body = (item.get("response") or {}).get("body", {})
            if resp_body.get("error"):
                msg = str(resp_body["error"])
                if len(error_sample) < 3:
                    error_sample.append(f"{custom_id}: {msg}")
                raw[custom_id] = ""
                continue

            try:
                raw[custom_id] = resp_body["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                if len(error_sample) < 3:
                    error_sample.append(f"{custom_id}: parse error — {exc}")
                raw[custom_id] = ""

        if error_sample:
            import warnings
            n_errors = sum(1 for v in raw.values() if v == "")
            warnings.warn(
                f"OpenAI batch {batch_id}: {n_errors}/{len(raw)} requests failed. "
                f"Sample errors: {error_sample}"
            )

        if not raw:
            return []

        # Re-sort by the numeric suffix in custom_id (e.g. "job-name-00042" → 42).
        # Some individual requests may be absent from the output file if they
        # failed silently — detect the prefix and fill those slots with "" so the
        # returned list always has the same length as the submission and the order
        # is aligned with sources/references.
        prefix = next(iter(raw)).rsplit("-", 1)[0]
        max_idx = max(int(k.rsplit("-", 1)[-1]) for k in raw)
        return [raw.get(f"{prefix}-{i:05d}", "") for i in range(max_idx + 1)]

    # --- Feature Methods ---

    def get_embeddings(
        self,
        texts: List[str],
        model: str = "text-embedding-3-large",
    ) -> List[List[float]]:
        """Get high-precision embeddings."""
        if self.client is None:
            self._initialize_client()
        response = self.client.embeddings.create(model=model, input=texts)
        return [item.embedding for item in response.data]

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken (2026 cl100k_base/o200k_base)."""
        import tiktoken
        try:
            encoding = tiktoken.encoding_for_model(self.config.model_name)
        except KeyError:
            encoding = tiktoken.get_encoding("o200k_base") # Standard for GPT-5
        return len(encoding.encode(text))

    # --- Evaluation Logic ---

    def evaluate_translation(
        self,
        source: str,
        hypothesis: str,
        reference: Optional[str] = None,
        evaluation_prompt: str = "",
        aspect: str = "fluency",
    ) -> Dict[str, Any]:
        """Evaluate translation quality across various metrics."""
        if not evaluation_prompt:
            evaluation_prompt = self._get_default_eval_prompt(aspect)

        prompt = evaluation_prompt.format(
            source_text=source,
            hypothesis=hypothesis,
            reference=reference or "Not provided",
        )

        response = self.generate(prompt)
        result = self._parse_evaluation_response(response)
        result.update({"aspect": aspect, "raw_response": response})
        return result

    def _get_default_eval_prompt(self, aspect: str) -> str:
        """Returns standard 2026 evaluation prompts."""
        prompts = {
            "fluency": """Evaluate the FLUENCY of this translation (1-5).
Focus on natural flow and grammatical correctness in the target language.
Source: {source_text}
Translation: {hypothesis}
Score (1-5):
Justification:""",

            "accuracy": """Evaluate the ACCURACY of this translation (1-5).
Source: {source_text}
Reference: {reference}
Translation: {hypothesis}
Score (1-5):
Justification:""",

            "cohesion": """Evaluate the COHESION of this translation (1-5).
Analyze logical transitions, thematic consistency, and the relationship between sentences.
Source: {source_text}
Translation: {hypothesis}
Score (1-5):
Justification:""",
        }
        return prompts.get(aspect, prompts["fluency"])

    def _parse_evaluation_response(self, response: str) -> Dict[str, Any]:
        """Robustly extracts score and justification from model output."""
        result = {"score": None, "justification": ""}
        
        # Extract Score (Handles formats like "Score: 4/5" or "4")
        score_match = re.search(r"score\s*[:\-]*\s*(\d)", response, re.IGNORE_CASE)
        if score_match:
            result["score"] = int(score_match.group(1))

        # Extract Justification
        if "Justification:" in response:
            result["justification"] = response.split("Justification:", 1)[-1].strip()
        else:
            # Fallback: take everything after the first newline if no label exists
            lines = response.strip().split("\n")
            if len(lines) > 1:
                result["justification"] = " ".join(lines[1:]).strip()

        return result