"""DRUM-ML Dataset Website & Web Portal Packaging Utility.

Packages all website-relevant files (portal index.html, interactive dataset explorer,
benchmark leaderboard dashboard, benchmark question reviewer, manifest, metadata JSONs,
scientific reports, and benchmark evaluation scorecards) into a compact, standalone,
deployable zip archive.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import zipfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from drum_ml.portal.generator import save_portal_html

# Default patterns for files considered essential to the dataset website
DEFAULT_WEBSITE_EXPLICIT_FILES: set[str] = {
    "index.html",
    "dataset_viewer.html",
    "manifest.json",
    "dataset_stats.json",
    "dataset_card.md",
    "archetype_report.json",
    "archetype_report.md",
    "persona_report.json",
    "persona_report.md",
    "train_persona_report.md",
    "dpo_preferences.jsonl",
}

DEFAULT_WEBSITE_BENCHMARK_FILES: set[str] = {
    "benchmark/leaderboard.html",
    "benchmark/benchmark_viewer.html",
    "benchmark/benchmark_report.json",
    "benchmark/drum_benchmark_all.jsonl",
    "benchmark/drum_benchmark_mcq.jsonl",
    "benchmark/drum_benchmark_open.jsonl",
}

DEFAULT_ASSET_EXTENSIONS: set[str] = {
    ".html",
    ".htm",
    ".css",
    ".js",
    ".mjs",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".ico",
    ".webp",
    ".json",
    ".md",
}

DEFAULT_EXCLUDE_DIR_NAMES: set[str] = {
    "by_persona",
    "by_archetype",
    "by_category",
    "formats",
    "__pycache__",
    ".git",
    ".github",
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
}

DEFAULT_EXCLUDE_FILE_PATTERNS: list[str] = [
    ".*",
    ".*.checkpoint.json",
    "*.checkpoint.json",
    ".DS_Store",
    "Thumbs.db",
    "*.tmp",
    "*.bak",
    "*~",
    "*.pyc",
    "*.pyo",
    "train.jsonl",
    "val.jsonl",
    "test.jsonl",
]


@dataclass
class PackagedFileInfo:
    """Metadata for an individual packaged file in the zip archive."""

    relative_path: str
    category: str
    size_bytes: int
    compressed_bytes: int
    crc32: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WebsitePackageResult:
    """Comprehensive summary result of the website packaging process."""

    zip_path: Path
    file_count: int
    uncompressed_bytes: int
    compressed_bytes: int
    compression_ratio: float
    savings_percent: float
    sha256_hash: str
    files: list[PackagedFileInfo] = field(default_factory=list)
    verification_passed: bool = True
    missing_recommended_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["zip_path"] = str(self.zip_path.resolve())
        return data


def compute_sha256(file_path: str | Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def categorize_file(relative_path: str) -> str:
    """Assign a human-readable categorization label to a website file."""
    norm = relative_path.replace("\\", "/")
    if norm == "index.html":
        return "Web Portal Landing"
    if norm == "dataset_viewer.html":
        return "Dataset Explorer App"
    if norm == "benchmark/leaderboard.html":
        return "Model Leaderboard App"
    if norm == "benchmark/benchmark_viewer.html":
        return "Benchmark Reviewer App"
    if norm.startswith("benchmark/") and norm.endswith("_scorecard.json"):
        return "Model Evaluation Scorecard"
    if norm.startswith("benchmark/") and norm.endswith(".jsonl"):
        return "Benchmark Gold Testbed"
    if norm.startswith("benchmark/"):
        return "Benchmark Report & Metadata"
    if norm in {"manifest.json", "dataset_stats.json"}:
        return "Dataset Manifest & Telemetry"
    if "report" in norm.lower() or norm == "dataset_card.md":
        return "Scientific Report & Card"
    if norm.endswith(".jsonl"):
        return "Dataset Split / Samples"
    if norm.startswith("by_persona/"):
        return "Persona Partition"
    if norm.startswith("by_archetype/"):
        return "Archetype Partition"
    if norm.startswith("by_category/"):
        return "Category Partition"
    if norm.startswith("formats/"):
        return "Export Format Partition"
    return "Website Asset"


def collect_website_files(
    dataset_dir: str | Path,
    include_all_data: bool = False,
    output_zip_path: str | Path | None = None,
    extra_include_patterns: Sequence[str] | None = None,
    extra_exclude_patterns: Sequence[str] | None = None,
) -> list[tuple[Path, str]]:
    """Collect all files relevant to the dataset website from the dataset directory.

    Args:
        dataset_dir: Root dataset directory path.
        include_all_data: If True, include large training/val/test splits and partition folders.
        output_zip_path: Optional path to the output zip, ensuring it is never packed into itself.
        extra_include_patterns: Additional glob patterns to include.
        extra_exclude_patterns: Additional glob patterns to exclude.

    Returns:
        List of tuples: (absolute_file_path, relative_archive_path).
    """
    root_p = Path(dataset_dir).resolve()
    if not root_p.exists() or not root_p.is_dir():
        raise FileNotFoundError(
            f"Dataset directory '{dataset_dir}' does not exist or is not a directory."
        )

    out_zip_resolved: Path | None = None
    if output_zip_path:
        out_zip_resolved = Path(output_zip_path).resolve()

    collected: list[tuple[Path, str]] = []
    seen_rel_paths: set[str] = set()

    for dirpath_str, dirnames, filenames in os.walk(root_p):
        cur_dir = Path(dirpath_str)
        rel_dir = cur_dir.relative_to(root_p)

        # Skip hidden directories and partition directories unless include_all_data is True
        dirnames[:] = [
            d
            for d in dirnames
            if not d.startswith(".") and (include_all_data or d not in DEFAULT_EXCLUDE_DIR_NAMES)
        ]

        for fname in sorted(filenames):
            # Never include hidden files
            if fname.startswith("."):
                continue

            file_abs = cur_dir / fname
            # Never include the output zip itself
            if out_zip_resolved and file_abs.resolve() == out_zip_resolved:
                continue

            rel_file_path = (rel_dir / fname).as_posix() if rel_dir != Path(".") else fname

            # Check extra exclude patterns
            if extra_exclude_patterns:
                if any(
                    fnmatch.fnmatch(fname, pat) or fnmatch.fnmatch(rel_file_path, pat)
                    for pat in extra_exclude_patterns
                ):
                    continue

            # In full data mode, include all non-hidden, non-temp files
            if include_all_data:
                if not any(
                    fnmatch.fnmatch(fname, pat) for pat in ["*.tmp", "*.bak", "*~", "Thumbs.db"]
                ):
                    if rel_file_path not in seen_rel_paths:
                        collected.append((file_abs, rel_file_path))
                        seen_rel_paths.add(rel_file_path)
                continue

            # Standard Website Package Mode (Curated relevant files)
            # 1. Check default exclude file patterns
            if any(
                fnmatch.fnmatch(fname, pat) or fnmatch.fnmatch(rel_file_path, pat)
                for pat in DEFAULT_EXCLUDE_FILE_PATTERNS
            ):
                continue

            # 2. Check explicit root & benchmark files
            is_website_file = False
            if rel_file_path in DEFAULT_WEBSITE_EXPLICIT_FILES:
                is_website_file = True
            elif rel_file_path in DEFAULT_WEBSITE_BENCHMARK_FILES:
                is_website_file = True
            elif rel_file_path.startswith("benchmark/") and rel_file_path.endswith(
                "_scorecard.json"
            ):
                is_website_file = True
            elif file_abs.suffix.lower() in DEFAULT_ASSET_EXTENSIONS and (
                rel_dir.as_posix() in {"assets", "static", "img", "images", "css", "js", "fonts"}
                or rel_dir == Path(".")
                or rel_dir == Path("benchmark")
            ):
                is_website_file = True

            # 3. Check extra include patterns
            if not is_website_file and extra_include_patterns:
                if any(
                    fnmatch.fnmatch(fname, pat) or fnmatch.fnmatch(rel_file_path, pat)
                    for pat in extra_include_patterns
                ):
                    is_website_file = True

            if is_website_file and rel_file_path not in seen_rel_paths:
                collected.append((file_abs, rel_file_path))
                seen_rel_paths.add(rel_file_path)

    # Sort files deterministically (root files first, then benchmark, then rest)
    def sort_key(item: tuple[Path, str]) -> tuple[int, str]:
        rel = item[1]
        if "/" not in rel:
            return (0, rel)
        if rel.startswith("benchmark/"):
            return (1, rel)
        return (2, rel)

    collected.sort(key=sort_key)
    return collected


def package_website_zip(
    dataset_dir: str | Path = "./dataset",
    output_zip: str | Path = "./dataset/drum_ml_website.zip",
    include_all_data: bool = False,
    rebuild_portal: bool = True,
    compresslevel: int = 9,
    extra_include_patterns: Sequence[str] | None = None,
    extra_exclude_patterns: Sequence[str] | None = None,
    verify: bool = True,
) -> WebsitePackageResult:
    """Packages all dataset website files into a clean, standalone, highly-compressed ZIP archive.

    Args:
        dataset_dir: Path to dataset directory.
        output_zip: Destination path for output ZIP file.
        include_all_data: Whether to include all data splits/partitions.
        rebuild_portal: If True, rebuilds `index.html` from manifest and stats prior to packaging.
        compresslevel: Compression level (1-9, default 9 for maximal compression).
        extra_include_patterns: Additional globs to include.
        extra_exclude_patterns: Additional globs to exclude.
        verify: If True, performs integrity and entrypoint verification on the created zip.

    Returns:
        WebsitePackageResult with comprehensive telemetry and file list.
    """
    root_p = Path(dataset_dir).resolve()
    out_zip_p = Path(output_zip).resolve()
    out_zip_p.parent.mkdir(parents=True, exist_ok=True)

    # Optional: Rebuild portal index.html to guarantee fresh embedded telemetry
    if rebuild_portal:
        manifest_file = root_p / "manifest.json"
        stats_file = root_p / "dataset_stats.json"
        index_file = root_p / "index.html"
        save_portal_html(
            output_html_path=index_file,
            manifest_path=manifest_file,
            stats_path=stats_file,
            open_browser=False,
        )

    # Collect files
    files_to_pack = collect_website_files(
        dataset_dir=root_p,
        include_all_data=include_all_data,
        output_zip_path=out_zip_p,
        extra_include_patterns=extra_include_patterns,
        extra_exclude_patterns=extra_exclude_patterns,
    )

    if not files_to_pack:
        raise ValueError(f"No website files found to package in '{dataset_dir}'.")

    # Create Zip Archive with Deflate compression
    with zipfile.ZipFile(
        out_zip_p,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=compresslevel,
    ) as zf:
        for file_abs, rel_path in files_to_pack:
            zf.write(file_abs, arcname=rel_path)

    # Inspect the newly created zip archive to extract exact entry telemetry
    packaged_files: list[PackagedFileInfo] = []
    total_uncompressed = 0
    total_compressed_entries = 0

    with zipfile.ZipFile(out_zip_p, mode="r") as zf:
        for info in zf.infolist():
            cat = categorize_file(info.filename)
            packaged_files.append(
                PackagedFileInfo(
                    relative_path=info.filename,
                    category=cat,
                    size_bytes=info.file_size,
                    compressed_bytes=info.compress_size,
                    crc32=info.CRC,
                )
            )
            total_uncompressed += info.file_size
            total_compressed_entries += info.compress_size

    final_zip_size = out_zip_p.stat().st_size
    sha256 = compute_sha256(out_zip_p)

    ratio = (total_uncompressed / final_zip_size) if final_zip_size > 0 else 1.0
    savings = (
        ((1.0 - (final_zip_size / total_uncompressed)) * 100.0) if total_uncompressed > 0 else 0.0
    )

    verification_ok = True
    missing_recommended: list[str] = []

    if verify:
        v_res = verify_website_zip(out_zip_p)
        verification_ok = v_res.get("integrity_ok", False)
        missing_recommended = v_res.get("missing_key_files", [])

    return WebsitePackageResult(
        zip_path=out_zip_p,
        file_count=len(packaged_files),
        uncompressed_bytes=total_uncompressed,
        compressed_bytes=final_zip_size,
        compression_ratio=ratio,
        savings_percent=savings,
        sha256_hash=sha256,
        files=packaged_files,
        verification_passed=verification_ok,
        missing_recommended_files=missing_recommended,
    )


def verify_website_zip(zip_path: str | Path) -> dict[str, Any]:
    """Verify zip file integrity and check for essential website entrypoints.

    Args:
        zip_path: Path to the ZIP file.

    Returns:
        Dict containing integrity check status, file list, and missing key files.
    """
    p = Path(zip_path).resolve()
    if not p.exists():
        return {"exists": False, "integrity_ok": False, "error": f"File {zip_path} not found"}

    try:
        with zipfile.ZipFile(p, mode="r") as zf:
            corrupted = zf.testzip()
            if corrupted is not None:
                return {
                    "exists": True,
                    "integrity_ok": False,
                    "corrupted_file": corrupted,
                    "error": f"CRC check failed on member: {corrupted}",
                }

            namelist = set(zf.namelist())

            recommended_entrypoints = [
                "index.html",
                "dataset_viewer.html",
                "benchmark/leaderboard.html",
                "benchmark/benchmark_viewer.html",
                "manifest.json",
                "dataset_stats.json",
            ]

            missing = [r for r in recommended_entrypoints if r not in namelist]

            return {
                "exists": True,
                "integrity_ok": True,
                "file_count": len(namelist),
                "namelist": sorted(namelist),
                "missing_key_files": missing,
            }
    except Exception as e:
        return {"exists": True, "integrity_ok": False, "error": str(e)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DRUM-ML Dataset Website ZIP Packager")
    parser.add_argument(
        "--dataset-dir",
        "-d",
        default="./dataset",
        help="Path to dataset root directory (default: ./dataset)",
    )
    parser.add_argument(
        "--output-zip",
        "-o",
        default="./dataset/drum_ml_website.zip",
        help="Path for output ZIP file (default: ./dataset/drum_ml_website.zip)",
    )
    parser.add_argument(
        "--include-all-data",
        "-a",
        action="store_true",
        help="Include full master training splits and partition folders",
    )
    parser.add_argument(
        "--no-rebuild",
        action="store_true",
        help="Skip rebuilding portal index.html before packaging",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result metadata as JSON",
    )

    args = parser.parse_args()

    res = package_website_zip(
        dataset_dir=args.dataset_dir,
        output_zip=args.output_zip,
        include_all_data=args.include_all_data,
        rebuild_portal=not args.no_rebuild,
    )

    if args.json:
        print(json.dumps(res.to_dict(), indent=2))
    else:
        print(f"✓ Packaged {res.file_count} website files into {res.zip_path}")
        print(f"  Uncompressed Size: {res.uncompressed_bytes:,} bytes")
        print(
            f"  ZIP File Size:     {res.compressed_bytes:,} bytes ({res.savings_percent:.1f}% savings)"
        )
        print(f"  SHA-256 Checksum:  {res.sha256_hash}")
