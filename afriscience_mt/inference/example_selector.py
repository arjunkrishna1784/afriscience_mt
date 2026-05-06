"""Example selection strategies for few-shot learning."""

import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..data.dataset import AfriScienceDataset
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ExampleSelector(ABC):
    """Abstract base class for example selection."""

    def __init__(self, examples_pool: List[Dict[str, str]]):
        """
        Initialize selector.

        Args:
            examples_pool: Pool of examples to select from
        """
        self.examples_pool = examples_pool

    @abstractmethod
    def select(
        self,
        query: str,
        n_examples: int,
        **kwargs,
    ) -> List[Dict[str, str]]:
        """
        Select examples for a query.

        Args:
            query: Query text (source text to translate)
            n_examples: Number of examples to select
            **kwargs: Additional arguments

        Returns:
            List of selected examples
        """
        pass

    @classmethod
    def from_dataset(
        cls,
        dataset: AfriScienceDataset,
        **kwargs,
    ) -> "ExampleSelector":
        """
        Create selector from a dataset.

        Args:
            dataset: Dataset to use as example pool
            **kwargs: Additional arguments for selector

        Returns:
            ExampleSelector instance
        """
        examples = [
            {"source": dataset[i]["source"], "target": dataset[i]["target"]}
            for i in range(len(dataset))
        ]
        return cls(examples_pool=examples, **kwargs)


class RandomSelector(ExampleSelector):
    """Random example selection."""

    def __init__(
        self,
        examples_pool: List[Dict[str, str]],
        seed: int = 42,
    ):
        """
        Initialize random selector.

        Args:
            examples_pool: Pool of examples
            seed: Random seed for reproducibility
        """
        super().__init__(examples_pool)
        self.rng = random.Random(seed)

    def select(
        self,
        query: str,
        n_examples: int,
        **kwargs,
    ) -> List[Dict[str, str]]:
        """Select random examples."""
        n_examples = min(n_examples, len(self.examples_pool))
        return self.rng.sample(self.examples_pool, n_examples)


class SemanticSelector(ExampleSelector):
    """
    Semantic similarity-based example selection using embeddings.

    Supports multiple embedding providers:
    - OpenAI: text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002
    - Local: sentence-transformers models (no API needed)

    Best practice: Use semantic selection to find examples similar to the
    source text being translated. This often outperforms random selection,
    especially for domain-specific or technical content.
    """

    def __init__(
        self,
        examples_pool: List[Dict[str, str]],
        embedding_model: str = "text-embedding-3-small",
        provider: str = "openai",  # "openai" or "local"
        api_key: Optional[str] = None,
        cache_embeddings: bool = True,
        cache_path: Optional[Path] = None,
        diversity_penalty: float = 0.0,  # 0 = pure similarity, >0 = encourage diversity
    ):
        """
        Initialize semantic selector.

        Args:
            examples_pool: Pool of examples
            embedding_model: Model name for embeddings
                - OpenAI: "text-embedding-3-small", "text-embedding-3-large"
                - Local: "all-MiniLM-L6-v2", "all-mpnet-base-v2", etc.
            provider: "openai" or "local" (sentence-transformers)
            api_key: API key for OpenAI (optional, uses env var)
            cache_embeddings: Whether to cache embeddings
            cache_path: Path to save/load cached embeddings
            diversity_penalty: Penalty for selecting similar examples (MMR-like)
        """
        super().__init__(examples_pool)
        self.embedding_model = embedding_model
        self.provider = provider
        self.api_key = api_key
        self.cache_embeddings = cache_embeddings
        self.cache_path = cache_path
        self.diversity_penalty = diversity_penalty

        self.embeddings = None
        self.client = None
        self.local_model = None

        # Try to load cached embeddings
        if cache_path and Path(cache_path).exists():
            self._load_cached_embeddings()
        elif cache_embeddings:
            self._compute_pool_embeddings()

    def _get_client(self):
        """Get or create OpenAI client."""
        if self.client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("openai package required for OpenAI embeddings. Install with: pip install openai")

            import os
            api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found. Set it in .env or pass api_key parameter.")
            self.client = OpenAI(api_key=api_key)

        return self.client

    def _get_local_model(self):
        """Get or create local sentence-transformers model."""
        if self.local_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers package required for local embeddings. "
                    "Install with: pip install sentence-transformers"
                )

            logger.info(f"Loading local embedding model: {self.embedding_model}")
            self.local_model = SentenceTransformer(self.embedding_model)

        return self.local_model

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for texts."""
        if self.provider == "openai":
            return self._get_openai_embeddings(texts)
        else:
            return self._get_local_embeddings(texts)

    def _get_openai_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings using OpenAI API."""
        client = self._get_client()

        # Handle empty texts
        texts = [t if t.strip() else " " for t in texts]

        response = client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )

        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)

    def _get_local_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings using local sentence-transformers model."""
        model = self._get_local_model()
        embeddings = model.encode(texts, show_progress_bar=False)
        return np.array(embeddings)

    def _compute_pool_embeddings(self) -> None:
        """Compute embeddings for all examples in pool."""
        logger.info(f"Computing embeddings for {len(self.examples_pool)} examples using {self.provider}...")

        sources = [ex["source"] for ex in self.examples_pool]

        if self.provider == "openai":
            # Batch for API efficiency
            batch_size = 100
            all_embeddings = []

            for i in range(0, len(sources), batch_size):
                batch = sources[i:i + batch_size]
                embeddings = self._get_embeddings(batch)
                all_embeddings.append(embeddings)
                if i > 0 and i % 500 == 0:
                    logger.info(f"  Processed {i}/{len(sources)} examples...")

            self.embeddings = np.vstack(all_embeddings)
        else:
            # Local model can handle all at once more efficiently
            self.embeddings = self._get_embeddings(sources)

        logger.info(f"Embeddings computed: shape {self.embeddings.shape}")

        # Save cache if path specified
        if self.cache_path:
            self._save_cached_embeddings()

    def _save_cached_embeddings(self) -> None:
        """Save embeddings to cache file."""
        if self.cache_path and self.embeddings is not None:
            cache_path = Path(self.cache_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, self.embeddings)
            logger.info(f"Embeddings cached to: {cache_path}")

    def _load_cached_embeddings(self) -> None:
        """Load embeddings from cache file."""
        if self.cache_path and Path(self.cache_path).exists():
            self.embeddings = np.load(self.cache_path)
            logger.info(f"Loaded cached embeddings: shape {self.embeddings.shape}")

    def select(
        self,
        query: str,
        n_examples: int,
        **kwargs,
    ) -> List[Dict[str, str]]:
        """
        Select most similar examples using cosine similarity.

        If diversity_penalty > 0, uses Maximal Marginal Relevance (MMR)
        to balance similarity with diversity.
        """
        if self.embeddings is None:
            self._compute_pool_embeddings()

        # Get query embedding
        query_embedding = self._get_embeddings([query])[0]

        # Compute cosine similarities
        norms = np.linalg.norm(self.embeddings, axis=1)
        query_norm = np.linalg.norm(query_embedding)
        similarities = np.dot(self.embeddings, query_embedding) / (norms * query_norm + 1e-8)

        if self.diversity_penalty == 0:
            # Pure similarity-based selection
            top_indices = np.argsort(similarities)[-n_examples:][::-1]
        else:
            # MMR-based selection for diversity
            top_indices = self._mmr_select(similarities, n_examples)

        return [self.examples_pool[i] for i in top_indices]

    def _mmr_select(self, similarities: np.ndarray, n_examples: int) -> List[int]:
        """
        Maximal Marginal Relevance selection.

        Balances relevance (similarity to query) with diversity
        (dissimilarity to already selected examples).
        """
        selected = []
        candidates = list(range(len(self.examples_pool)))

        for _ in range(n_examples):
            if not candidates:
                break

            if not selected:
                # First selection: most similar to query
                best_idx = max(candidates, key=lambda i: similarities[i])
            else:
                # Subsequent selections: balance relevance and diversity
                best_score = float('-inf')
                best_idx = candidates[0]

                for idx in candidates:
                    relevance = similarities[idx]

                    # Compute max similarity to already selected
                    max_sim_to_selected = max(
                        np.dot(self.embeddings[idx], self.embeddings[s]) /
                        (np.linalg.norm(self.embeddings[idx]) * np.linalg.norm(self.embeddings[s]) + 1e-8)
                        for s in selected
                    )

                    # MMR score
                    score = relevance - self.diversity_penalty * max_sim_to_selected

                    if score > best_score:
                        best_score = score
                        best_idx = idx

            selected.append(best_idx)
            candidates.remove(best_idx)

        return selected

    def get_similarity_scores(self, query: str) -> List[Tuple[int, float, str]]:
        """
        Get similarity scores for all examples (useful for debugging).

        Returns:
            List of (index, similarity, source_text) tuples, sorted by similarity
        """
        if self.embeddings is None:
            self._compute_pool_embeddings()

        query_embedding = self._get_embeddings([query])[0]
        norms = np.linalg.norm(self.embeddings, axis=1)
        query_norm = np.linalg.norm(query_embedding)
        similarities = np.dot(self.embeddings, query_embedding) / (norms * query_norm + 1e-8)

        results = [
            (i, float(similarities[i]), self.examples_pool[i]["source"][:100])
            for i in range(len(self.examples_pool))
        ]
        return sorted(results, key=lambda x: x[1], reverse=True)


class DomainBalancedSelector(ExampleSelector):
    """Domain-balanced example selection."""

    def __init__(
        self,
        examples_pool: List[Dict[str, str]],
        domain_key: str = "domain",
        seed: int = 42,
    ):
        """
        Initialize domain-balanced selector.

        Args:
            examples_pool: Pool of examples (with domain metadata)
            domain_key: Key for domain in example dict
            seed: Random seed
        """
        super().__init__(examples_pool)
        self.domain_key = domain_key
        self.rng = random.Random(seed)

        # Group examples by domain
        self.domain_examples = {}
        for ex in examples_pool:
            domain = ex.get(domain_key, "unknown")
            if domain not in self.domain_examples:
                self.domain_examples[domain] = []
            self.domain_examples[domain].append(ex)

        self.domains = list(self.domain_examples.keys())
        logger.info(f"Domain-balanced selector with {len(self.domains)} domains")

    def select(
        self,
        query: str,
        n_examples: int,
        target_domain: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, str]]:
        """
        Select domain-balanced examples.

        Args:
            query: Query text
            n_examples: Number of examples
            target_domain: If specified, prioritize this domain
        """
        selected = []
        n_per_domain = n_examples // len(self.domains)
        remainder = n_examples % len(self.domains)

        # Shuffle domain order
        domains = self.domains.copy()
        self.rng.shuffle(domains)

        # If target domain specified, put it first
        if target_domain and target_domain in domains:
            domains.remove(target_domain)
            domains.insert(0, target_domain)

        for i, domain in enumerate(domains):
            n_select = n_per_domain + (1 if i < remainder else 0)
            domain_pool = self.domain_examples[domain]

            if n_select > 0 and domain_pool:
                n_select = min(n_select, len(domain_pool))
                selected.extend(self.rng.sample(domain_pool, n_select))

        return selected[:n_examples]


def create_example_selector(
    method: str,
    examples_pool: Optional[List[Dict[str, str]]] = None,
    dataset: Optional[AfriScienceDataset] = None,
    **kwargs,
) -> ExampleSelector:
    """
    Factory function to create an example selector.

    Args:
        method: Selection method ("random", "semantic", "domain_balanced")
        examples_pool: Pool of examples (optional if dataset provided)
        dataset: Dataset to create pool from (optional if pool provided)
        **kwargs: Additional arguments for selector

    Returns:
        ExampleSelector instance
    """
    if examples_pool is None and dataset is None:
        raise ValueError("Must provide either examples_pool or dataset")

    if examples_pool is None:
        examples_pool = [
            {"source": dataset[i]["source"], "target": dataset[i]["target"]}
            for i in range(len(dataset))
        ]

    selectors = {
        "random": RandomSelector,
        "semantic": SemanticSelector,
        "semantic_similarity": SemanticSelector,
        "domain_balanced": DomainBalancedSelector,
    }

    selector_class = selectors.get(method.lower())
    if selector_class is None:
        raise ValueError(f"Unknown selection method: {method}")

    return selector_class(examples_pool=examples_pool, **kwargs)
