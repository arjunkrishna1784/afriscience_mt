"""Data integrity utilities using checksums."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def compute_hash(data: Union[str, bytes, Dict, List], algorithm: str = "sha256") -> str:
    """
    Compute hash of data.

    Args:
        data: Data to hash (string, bytes, or JSON-serializable object)
        algorithm: Hash algorithm to use

    Returns:
        Hexadecimal hash string
    """
    hasher = hashlib.new(algorithm)

    if isinstance(data, bytes):
        hasher.update(data)
    elif isinstance(data, str):
        hasher.update(data.encode("utf-8"))
    else:
        # JSON-serialize for dicts/lists
        json_str = json.dumps(data, sort_keys=True, ensure_ascii=False)
        hasher.update(json_str.encode("utf-8"))

    return hasher.hexdigest()


def compute_file_hash(
    file_path: Union[str, Path],
    algorithm: str = "sha256",
    chunk_size: int = 8192
) -> str:
    """
    Compute hash of a file.

    Args:
        file_path: Path to file
        algorithm: Hash algorithm to use
        chunk_size: Size of chunks to read

    Returns:
        Hexadecimal hash string
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    hasher = hashlib.new(algorithm)

    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)

    return hasher.hexdigest()


def verify_integrity(
    data: Union[str, bytes, Dict, List],
    expected_hash: str,
    algorithm: str = "sha256"
) -> bool:
    """
    Verify data integrity against expected hash.

    Args:
        data: Data to verify
        expected_hash: Expected hash value
        algorithm: Hash algorithm used

    Returns:
        True if hash matches, False otherwise
    """
    computed = compute_hash(data, algorithm)
    return computed == expected_hash


def verify_file_integrity(
    file_path: Union[str, Path],
    expected_hash: str,
    algorithm: str = "sha256"
) -> bool:
    """
    Verify file integrity against expected hash.

    Args:
        file_path: Path to file
        expected_hash: Expected hash value
        algorithm: Hash algorithm used

    Returns:
        True if hash matches, False otherwise
    """
    computed = compute_file_hash(file_path, algorithm)
    return computed == expected_hash


def create_data_manifest(
    data_dir: Union[str, Path],
    patterns: Optional[List[str]] = None,
    algorithm: str = "sha256"
) -> Dict[str, Any]:
    """
    Create a manifest of all data files with their hashes.

    Args:
        data_dir: Directory containing data files
        patterns: Optional glob patterns to match files
        algorithm: Hash algorithm to use

    Returns:
        Manifest dictionary with file paths and hashes
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    if patterns is None:
        patterns = ["**/*.json", "**/*.tsv", "**/*.txt", "**/*.csv"]

    manifest = {
        "root": str(data_dir.resolve()),
        "algorithm": algorithm,
        "files": {},
    }

    for pattern in patterns:
        for file_path in data_dir.glob(pattern):
            if file_path.is_file():
                rel_path = str(file_path.relative_to(data_dir))
                manifest["files"][rel_path] = {
                    "hash": compute_file_hash(file_path, algorithm),
                    "size": file_path.stat().st_size,
                }

    manifest["total_files"] = len(manifest["files"])
    return manifest


def verify_data_manifest(
    data_dir: Union[str, Path],
    manifest: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Verify data files against a manifest.

    Args:
        data_dir: Directory containing data files
        manifest: Manifest to verify against

    Returns:
        Verification result with status and any mismatches
    """
    data_dir = Path(data_dir)
    algorithm = manifest.get("algorithm", "sha256")

    result = {
        "valid": True,
        "missing": [],
        "modified": [],
        "extra": [],
    }

    # Check each file in manifest
    for rel_path, file_info in manifest["files"].items():
        file_path = data_dir / rel_path

        if not file_path.exists():
            result["valid"] = False
            result["missing"].append(rel_path)
        else:
            computed = compute_file_hash(file_path, algorithm)
            if computed != file_info["hash"]:
                result["valid"] = False
                result["modified"].append({
                    "path": rel_path,
                    "expected": file_info["hash"],
                    "computed": computed,
                })

    # Find extra files (optional)
    existing_files = set()
    for pattern in ["**/*.json", "**/*.tsv", "**/*.txt", "**/*.csv"]:
        for file_path in data_dir.glob(pattern):
            if file_path.is_file():
                existing_files.add(str(file_path.relative_to(data_dir)))

    manifest_files = set(manifest["files"].keys())
    result["extra"] = list(existing_files - manifest_files)

    return result


def save_manifest(manifest: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    Save manifest to a JSON file.

    Args:
        manifest: Manifest dictionary
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def load_manifest(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load manifest from a JSON file.

    Args:
        path: Path to manifest file

    Returns:
        Manifest dictionary
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class IntegrityChecker:
    """Utility class for checking data integrity across experiments."""

    def __init__(self, data_dir: Union[str, Path]):
        """
        Initialize integrity checker.

        Args:
            data_dir: Root data directory
        """
        self.data_dir = Path(data_dir)
        self.manifest_path = self.data_dir / "data_manifest.json"
        self.manifest = None

    def create_manifest(self, force: bool = False) -> Dict[str, Any]:
        """
        Create or load data manifest.

        Args:
            force: Force recreation even if manifest exists

        Returns:
            Data manifest
        """
        if self.manifest_path.exists() and not force:
            self.manifest = load_manifest(self.manifest_path)
        else:
            self.manifest = create_data_manifest(self.data_dir)
            save_manifest(self.manifest, self.manifest_path)

        return self.manifest

    def verify(self) -> Dict[str, Any]:
        """
        Verify data integrity.

        Returns:
            Verification result
        """
        if self.manifest is None:
            self.create_manifest()

        return verify_data_manifest(self.data_dir, self.manifest)

    def get_file_hash(self, rel_path: str) -> Optional[str]:
        """
        Get hash of a specific file from manifest.

        Args:
            rel_path: Relative path within data directory

        Returns:
            Hash string or None if not in manifest
        """
        if self.manifest is None:
            self.create_manifest()

        return self.manifest["files"].get(rel_path, {}).get("hash")
