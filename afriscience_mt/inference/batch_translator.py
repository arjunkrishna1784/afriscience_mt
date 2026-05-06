"""Batch translation utilities."""

import time
import asyncio
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from tqdm.asyncio import tqdm  # Use the async version of tqdm

from ..data.dataset import AfriScienceDataset
from ..models.base import BaseTranslationModel
from ..api.base import APIClient
from ..utils.logging import get_logger

logger = get_logger(__name__)

class BatchTranslator:
    def __init__(
        self,
        model: Optional[BaseTranslationModel] = None,
        api_client: Optional[APIClient] = None,
        batch_size: int = 8,        # For local models
        max_concurrent: int = 10,   # For API clients
        show_progress: bool = True,
    ):
        """
        Initialize batch translator.

        Args:
            model: Local model for translation
            api_client: API client for translation
            batch_size: Batch size for local models
            show_progress: Show progress bar
        """
        self.model = model
        self.api_client = api_client
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.show_progress = show_progress
        # Semaphore prevents too many simultaneous API calls
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def translate(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        **kwargs,
    ) -> List[str]:
        """
        Translate a list of texts.

        Args:
            texts: Texts to translate
            source_lang: Source language code
            target_lang: Target language code
            **kwargs: Additional arguments

        Returns:
            List of translations
        """
        if self.model is not None:
            # Local models still usually benefit from sequential batching
            return self._translate_with_model(texts, source_lang, target_lang, **kwargs)
        else:
            return await self._translate_with_api_async(texts, source_lang, target_lang, **kwargs)

    async def _translate_with_model(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        **kwargs,
    ) -> List[str]:
        """Translate using local model."""
        translations = []

        iterator = range(0, len(texts), self.batch_size)
        if self.show_progress:
            # Use tqdm.asyncio to ensure the progress bar renders 
            # while the loop is 'awaiting' batches
            from tqdm.asyncio import tqdm
            iterator = tqdm(iterator, desc="Translating", total=(len(texts) // self.batch_size) + 1)

        for i in iterator:
            batch = texts[i:i + self.batch_size]
            
            # 2. Use 'run_in_executor' if the model is heavy. 
            # This offloads the heavy GPU/CPU work so it doesn't freeze the app.
            loop = asyncio.get_event_loop()
            batch_translations = await loop.run_in_executor(
                None, 
                lambda: self.model.translate_batch(
                    batch, source_lang, target_lang, **kwargs
                )
            )
            
            translations.extend(batch_translations)

        return translations
    
    async def _translate_with_api_async(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        **kwargs,
    ) -> List[str]:
        """Parallelized API translation."""
        
        async def semaphore_wrapper(text):
            # Each task must wait its turn to 'acquire' the semaphore
            async with self.semaphore:
                # This calls your async GoogleAIClient._call_api
                result = await self.api_client.translate_async(
                    text=text,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    **kwargs
                )
                # Small jitter to prevent burst-mode rate limiting
                await asyncio.sleep(0.1) 
                return result

        # Create a list of tasks
        tasks = [semaphore_wrapper(t) for t in texts]
        
        # Run them all concurrently with a progress bar
        translations = await tqdm.gather(
            *tasks, 
            desc=f"Translating {source_lang}→{target_lang}",
            disable=not self.show_progress
        )
        
        return translations

    async def translate_dataset(
        self,
        dataset: AfriScienceDataset,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Translate an entire dataset.

        Args:
            dataset: Dataset to translate
            source_lang: Source language (uses dataset default if None)
            target_lang: Target language (uses dataset default if None)
            **kwargs: Additional translation arguments

        Returns:
            Dictionary with translations and metadata
        """
        source_lang = source_lang or dataset.source_lang
        target_lang = target_lang or dataset.target_lang

        sources = dataset.get_sources()
        references = dataset.get_targets()

        logger.info(f"Translating {len(sources)} samples ({source_lang} -> {target_lang})")

        start_time = time.time()
        
        # This is the key change: await the internal translate method
        translations = await self.translate(sources, source_lang, target_lang, **kwargs)
        
        elapsed = time.time() - start_time

        return {
            "sources": sources,
            "references": references,
            "hypotheses": translations,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "num_samples": len(sources),
            "translation_time": elapsed,
            "samples_per_second": len(sources) / elapsed if elapsed > 0 else 0,
        }

async def translate_dataset(
    dataset: AfriScienceDataset,
    model: Optional[BaseTranslationModel] = None,
    api_client: Optional[APIClient] = None,
    batch_size: int = 8,
    output_path: Optional[Union[str, Path]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function to translate a dataset.

    Args:
        dataset: Dataset to translate
        model: Local model (optional)
        api_client: API client (optional)
        batch_size: Batch size
        output_path: Optional path to save results
        **kwargs: Additional arguments

    Returns:
        Translation results
    """
    translator = BatchTranslator(
        model=model,
        api_client=api_client,
        batch_size=batch_size,
    )

    # Use 'await' here because translator.translate is now async
    results = await translator.translate_dataset(dataset, **kwargs)

    if output_path:
        import json
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"Results saved to {output_path}")

    return results