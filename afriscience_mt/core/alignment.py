"""Bind predictions to a reference manifest before inference; align by sample ID."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

from .schema import PredictionFile


def validate_manifest(manifest):
    """Validate explicit dataset IDs and text without changing their spelling/order."""
    if not isinstance(manifest, dict):
        raise ValueError("Reference manifest must be an object")
    for key in ("dataset", "split", "source_lang", "target_lang"):
        if not isinstance(manifest.get(key), str) or not manifest[key].strip():
            raise ValueError(f"Manifest {key} must be a nonempty string")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Manifest samples must be a nonempty list")
    seen = set()
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("Manifest sample must be an object")
        sample_id = sample.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError("sample_id must be a nonempty string")
        if sample_id in seen:
            raise ValueError(f"Duplicate manifest sample_id: {sample_id}")
        seen.add(sample_id)
        for key in ("src", "ref"):
            if not isinstance(sample.get(key), str):
                raise ValueError(f"Manifest {key} must be a string")
        if not sample["ref"].strip():
            raise ValueError("Manifest reference must not be empty")
    return manifest


def manifest_fingerprint(manifest):
    """SHA-256 over identity, IDs and exact UTF-8 texts, independent of row order."""
    validate_manifest(manifest)
    canonical = {key: manifest[key] for key in
                 ("dataset", "split", "source_lang", "target_lang")}
    canonical["samples"] = sorted(
        ({key: row[key] for key in ("sample_id", "src", "ref")}
         for row in manifest["samples"]), key=lambda row: row["sample_id"])
    encoded = json.dumps(canonical, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def load_manifest(path):
    with Path(path).open(encoding="utf-8") as handle:
        return validate_manifest(json.load(handle))


async def translate_manifest(manifest, experiment_id, translate):
    """Snapshot IDs/text before inference and return canonical bound predictions.

    ``translate`` is an async callable accepting (texts, source_lang, target_lang)
    and returning one hypothesis per input, in order (e.g. BatchTranslator.translate).
    Do not use this to relabel old positional predictions: their order is unknown.
    """
    manifest = deepcopy(validate_manifest(manifest))
    fingerprint = manifest_fingerprint(manifest)
    hypotheses = await translate([row["src"] for row in manifest["samples"]],
                                 manifest["source_lang"], manifest["target_lang"])
    if not isinstance(hypotheses, (list, tuple)) or len(hypotheses) != len(manifest["samples"]):
        raise ValueError("Inference must return one hypothesis per manifest sample")
    if any(not isinstance(hyp, str) for hyp in hypotheses):
        raise ValueError("Inference hypotheses must be strings")
    return PredictionFile(
        experiment_id, manifest["source_lang"], manifest["target_lang"],
        [{"sample_id": row["sample_id"], "src": row["src"], "ref": row["ref"], "hyp": hyp}
         for row, hyp in zip(manifest["samples"], hypotheses)],
        schema_version="1.1", reference_fingerprint=fingerprint)


def aligned_inputs(predictions, result, manifest):
    """Reject incomplete/stale artifacts, then put predictions in manifest order."""
    fingerprint = manifest_fingerprint(manifest)
    for key in ("dataset", "split", "source_lang", "target_lang"):
        if getattr(result, key) != manifest[key]:
            raise ValueError(f"Result/manifest {key} mismatch")
    if (predictions.source_lang, predictions.target_lang) != (
        manifest["source_lang"], manifest["target_lang"]
    ):
        raise ValueError("Prediction/manifest language mismatch")
    if predictions.reference_fingerprint != fingerprint:
        raise ValueError("Missing or mismatched reference fingerprint; cannot verify alignment")
    if not isinstance(predictions.samples, list):
        raise ValueError("Prediction samples must be a list")
    indexed = {}
    for sample in predictions.samples:
        if not isinstance(sample, dict):
            raise ValueError("Prediction sample must be an object")
        sample_id = sample.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError("Prediction sample_id must be a nonempty string")
        if sample_id in indexed:
            raise ValueError(f"Duplicate prediction sample_id: {sample_id}")
        if not isinstance(sample.get("hyp"), str):
            raise ValueError("Prediction hypothesis must be a string")
        indexed[sample_id] = sample
    expected = {row["sample_id"] for row in manifest["samples"]}
    if set(indexed) != expected:
        raise ValueError(f"Prediction ID set mismatch: {len(expected - set(indexed))} missing, "
                         f"{len(set(indexed) - expected)} extra")
    sources, hypotheses, references = [], [], []
    for row in manifest["samples"]:
        sample = indexed[row["sample_id"]]
        for key in ("src", "ref"):
            if key in sample and sample[key] != row[key]:
                raise ValueError(f"Prediction/manifest {key} mismatch for {row['sample_id']}")
        sources.append(row["src"])
        references.append(row["ref"])
        hypotheses.append(sample["hyp"])
    return sources, hypotheses, references
