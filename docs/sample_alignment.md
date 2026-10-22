# Rescoring with explicit sample identity

Matching row counts cannot establish whether saved hypotheses belong to the
current test examples. For new inference, record a reference manifest **before**
translation and bind each hypothesis to its sample ID. IDs must come from your
chosen dataset version; no IDs are inferred from old prediction row positions.

A UTF-8 JSON manifest has this shape (use your actual dataset and split):

```json
{
  "dataset": "my_dataset_v1", "split": "test",
  "source_lang": "eng", "target_lang": "yor",
  "samples": [
    {"sample_id": "document-42:sentence-3", "src": "Source text", "ref": "Reference text"}
  ]
}
```

The new opt-in inference adapter takes an async callable that returns one output
per input in the same order. It snapshots the manifest before calling inference,
checks output counts and types, and returns canonical predictions containing IDs,
source/reference text and a SHA-256 reference fingerprint. It accepts empty
hypotheses, which must remain in evaluation.

```python
from afriscience_mt.core.alignment import load_manifest, translate_manifest
from afriscience_mt.core.schema import save_predictions

manifest = load_manifest("reference_manifest.json")
# Inside your async inference driver; translator is an existing BatchTranslator
# configured with an API client, or provide your own async translation callable.
predictions = await translate_manifest(manifest, "my_experiment", translator.translate)
save_predictions(predictions, "experiments/my_run/predictions.json")
```

Write the sibling canonical `results.json` with the same experiment/languages and
manifest dataset/split, then rescore without inference:

```bash
afriscience-mt evaluate --mode rescore --src experiments/my_run \
  --metric chrf --reference-manifest reference_manifest.json
```

The rescorer requires an exact ID set, matching experiment/language identity,
dataset/split, fingerprint and any inline source/reference texts. Reordered rows
are aligned to manifest order; duplicate, missing, extra or blank IDs fail during
batch preflight before scoring or writing any results. A manifest is required for
ID-bearing predictions. One invocation uses one manifest: scope `--src` to runs
on that dataset version and language pair.

The fingerprint covers exact text, IDs, dataset/split and language codes. It is
independent of row order and incidental extra metadata. This detects accidental
artifact/reference mismatch; it does not authenticate who created a file or
prove the translation callable respected its order contract.

Existing artifacts still load. Without a manifest the existing positional path
remains available and is recorded as `legacy_positional_unverified` in rescore
metadata. It does **not** establish provenance. Do not add a current fingerprint
and positional IDs to old files and claim their order was verified. Existing
training scripts are not automatically migrated to this opt-in inference API.

```bash
python -m pytest -q tests/test_sample_alignment.py tests/test_evaluation_integrity.py
```
