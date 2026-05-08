"""Google AI (Gemini) API client — google-genai SDK (2026)."""

import asyncio
from typing import Any, Dict, List, Optional

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from .base import APIClient, APIConfig

# Max concurrent Gemini requests (stay well inside rate-limit headroom).
# Gemini 1.5/2.x pay-as-you-go: up to 1000 RPM, but bursts get 429s.
# 10 parallel requests is safe and gives ~10× throughput vs sequential.
_GEMINI_CONCURRENCY = 10


class GoogleAIClient(APIClient):
    """Client for Google AI API using the google-genai SDK."""

    def _initialize_client(self) -> None:
        """Initialize the GenAI client with built-in retry."""
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError(
                "google-genai not installed. Run: pip install google-genai"
            )

        self._genai = genai
        self._types = types

        # SDK-level retry: handles 429 / 503 automatically before we ever see them.
        self.client = genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(
                    attempts=6,
                    initial_delay=2000,   # ms — start with 2 s back-off
                    max_delay=60000,      # ms — cap at 60 s
                )
            ),
        )

        self._gen_config = types.GenerateContentConfig(
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
        )

    # ------------------------------------------------------------------
    # Single-call (sync) — used by base.generate()
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        retry=retry_if_exception_type(Exception),
    )
    def _call_api(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> str:
        if self.client is None:
            self._initialize_client()

        # Separate system prompt (native system_instruction support).
        system_parts: List[str] = []
        contents = []
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
                continue
            role = "user" if msg["role"] == "user" else "model"
            contents.append(
                self._types.Content(
                    role=role,
                    parts=[self._types.Part.from_text(text=msg["content"])],
                )
            )

        cfg = self._gen_config
        if system_parts:
            cfg = self._types.GenerateContentConfig(
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                system_instruction="\n".join(system_parts),
            )

        response = self.client.models.generate_content(
            model=self.config.model_name,
            contents=contents,
            config=cfg,
        )
        return response.text

    # ------------------------------------------------------------------
    # Async helper — used by generate_batch
    # ------------------------------------------------------------------

    async def _generate_async(
        self,
        prompt: str,
        semaphore: asyncio.Semaphore,
    ) -> str:
        """Single async generation call, rate-limited by semaphore."""
        async with semaphore:
            contents = self._types.Content(
                role="user",
                parts=[self._types.Part.from_text(text=prompt)],
            )
            response = await self.client.aio.models.generate_content(
                model=self.config.model_name,
                contents=contents,
                config=self._gen_config,
            )
            return response.text

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
        ``_GEMINI_CONCURRENCY`` requests are in-flight at once, giving
        ~10× throughput over sequential calls while respecting rate limits.
        """
        if self.client is None:
            self._initialize_client()

        # If a system prompt is requested, fold it in via config override.
        if system_prompt:
            self._gen_config = self._types.GenerateContentConfig(
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                system_instruction=system_prompt,
            )

        async def _run_all() -> List[str]:
            sem = asyncio.Semaphore(_GEMINI_CONCURRENCY)
            tasks = [self._generate_async(p, sem) for p in prompts]
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
        Submit all prompts as a single Gemini Batch API job (file-based JSONL).

        Uses file-based JSONL with indexed 'key' fields so that results can be
        re-sorted by key on retrieval, guaranteeing alignment with the source
        regardless of the order Gemini returns them.

        Returns the batch job name — store it and call retrieve_batch_results()
        later (minutes to a few hours) to get the outputs.

        Cost: 50% of real-time pricing.
        Completion window: up to 24 h (usually much faster).
        """
        import io
        import json as _json
        import tempfile, os

        if self.client is None:
            self._initialize_client()

        # Build JSONL with indexed keys for reliable re-ordering on retrieval.
        # The Batch API expects generationConfig and systemInstruction as
        # top-level request fields (not nested under "config"). Earlier code
        # used "config" which the server rejects with "no such field: 'config'".
        gen_config: Dict[str, Any] = {}
        if self.config.temperature is not None:
            gen_config["temperature"] = self.config.temperature
        if self.config.max_tokens:
            gen_config["maxOutputTokens"] = self.config.max_tokens
        if self.config.top_p is not None:
            gen_config["topP"] = self.config.top_p

        lines: List[str] = []
        for i, prompt in enumerate(prompts):
            req: Dict[str, Any] = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            }
            if system_prompt:
                req["systemInstruction"] = {
                    "parts": [{"text": system_prompt}],
                }
            if gen_config:
                req["generationConfig"] = gen_config
            lines.append(_json.dumps({"key": str(i), "request": req}, ensure_ascii=False))

        jsonl_content = "\n".join(lines)

        # Write to a temp file and upload via the Files API.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", encoding="utf-8", delete=False
        ) as tmp:
            tmp.write(jsonl_content)
            tmp_path = tmp.name

        try:
            uploaded = self.client.files.upload(
                file=tmp_path,
                config=self._types.UploadFileConfig(
                    display_name=f"{job_name}_input",
                    mime_type="application/jsonl",
                ),
            )
        finally:
            os.unlink(tmp_path)

        batch = self.client.batches.create(
            model=self.config.model_name,
            src=uploaded.name,
            config={"display_name": job_name},
        )
        return batch.name

    def cancel_batch_job(self, batch_name: str) -> None:
        """Cancel a running Gemini batch job."""
        if self.client is None:
            self._initialize_client()
        self.client.batches.cancel(name=batch_name)

    def retrieve_batch_results(
        self,
        batch_name: str,
    ) -> Optional[List[str]]:
        """
        Poll a Gemini Batch job and return results when complete.

        Returns:
            Ordered list of response strings (same order as submission) if done.
            None if the job is still pending / running.
        Raises:
            RuntimeError if the batch failed or was cancelled.
        """
        if self.client is None:
            self._initialize_client()

        job = self.client.batches.get(name=batch_name)
        # job.state is a protobuf enum; .name gives the string like "JOB_STATE_SUCCEEDED"
        state = job.state.name if hasattr(job.state, "name") else str(job.state)

        if any(s in state for s in ("FAILED", "CANCELLED", "EXPIRED")):
            raise RuntimeError(
                f"Gemini batch '{batch_name}' ended with state: {state}"
            )

        if "SUCCEEDED" not in state:
            return None  # Still pending / running

        import json as _json
        import warnings

        dest = job.dest

        # --- File-based batch (new path): sort by key to guarantee order -------
        if getattr(dest, "file_name", None):
            raw_bytes = self.client.files.download(file=dest.file_name)
            content = raw_bytes.decode("utf-8") if isinstance(raw_bytes, bytes) else raw_bytes

            raw: Dict[str, str] = {}
            n_errors = 0
            error_sample: List[str] = []

            for line in content.strip().splitlines():
                if not line:
                    continue
                item = _json.loads(line)
                key = item.get("key", "0")

                if item.get("error"):
                    n_errors += 1
                    if len(error_sample) < 3:
                        error_sample.append(f"key {key}: {item['error']}")
                    raw[key] = ""
                    continue

                try:
                    # File-based response: nested under "response"
                    resp = item.get("response", item)
                    text = resp["candidates"][0]["content"]["parts"][0]["text"]
                    raw[key] = text
                except (KeyError, IndexError, TypeError) as exc:
                    n_errors += 1
                    if len(error_sample) < 3:
                        error_sample.append(f"key {key}: parse error — {exc}")
                    raw[key] = ""

            if n_errors:
                warnings.warn(
                    f"Gemini batch '{batch_name}': {n_errors}/{len(raw)} items failed. "
                    f"Sample errors: {error_sample}"
                )

            if not raw:
                return []

            max_idx = max(int(k) for k in raw)
            return [raw.get(str(i), "") for i in range(max_idx + 1)]

        # --- Inline batch (legacy path): enumerate in API order ----------------
        # NOTE: inlined_responses order is NOT guaranteed by the Gemini API.
        # This path is kept only for batches submitted before the file-based fix.
        results: List[str] = []
        n_errors = 0
        error_sample_inline: List[str] = []
        for i, item in enumerate(dest.inlined_responses):
            if item.error:
                n_errors += 1
                if len(error_sample_inline) < 3:
                    error_sample_inline.append(f"item {i}: {item.error}")
                results.append("")
                continue
            try:
                results.append(item.response.text)
            except (AttributeError, ValueError, IndexError):
                try:
                    results.append(
                        item.response.candidates[0].content.parts[0].text
                    )
                except (AttributeError, IndexError):
                    n_errors += 1
                    results.append("")

        if n_errors:
            warnings.warn(
                f"Gemini batch '{batch_name}': {n_errors}/{len(results)} items "
                f"failed (inline batch). Sample errors: {error_sample_inline}"
            )
        return results

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        if self.client is None:
            self._initialize_client()
        response = self.client.models.count_tokens(
            model=self.config.model_name,
            contents=text,
        )
        return response.total_tokens
