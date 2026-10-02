"""DRUM-ML Interactive HTML Dataset Browser & Viewer Generator.

Generates a standalone, feature-rich, high-performance interactive HTML dashboard
for exploring, filtering, reviewing, and auditing DRUM-ML training, validation,
test, and DPO datasets, including breakdowns by persona, archetype, and category.
"""

from __future__ import annotations

import json
import random
import webbrowser
from pathlib import Path
from typing import Any

from drum_ml.models.scaffolds import PersonaType


def parse_persona_from_record(item: dict[str, Any]) -> str:
    """Extract or infer persona identifier from a dataset record."""
    if item.get("persona"):
        p = item["persona"]
        return p.value if hasattr(p, "value") else str(p)
    if "metadata" in item and isinstance(item["metadata"], dict):
        if "persona" in item["metadata"] and item["metadata"]["persona"]:
            return str(item["metadata"]["persona"])

    rec_id = item.get("id", "")
    for p in PersonaType:
        val = p.value
        if f"_{val}_" in rec_id or rec_id.endswith(f"_{val}") or rec_id.startswith(f"{val}_"):
            return val
    return "general_user"


def parse_category_from_record(item: dict[str, Any]) -> str:
    """Extract or infer entity category (units, constants, quantity_kinds)."""
    if item.get("category"):
        return str(item["category"])
    if "metadata" in item and isinstance(item["metadata"], dict):
        if "category" in item["metadata"] and item["metadata"]["category"]:
            return str(item["metadata"]["category"])

    uri = item.get("entity_uri", "")
    if "/constant/" in uri or "constants" in uri:
        return "constants"
    if "/quantitykind/" in uri or "quantity_kind" in uri:
        return "quantity_kinds"
    if "/unit/" in uri or "units" in uri:
        return "units"
    return "units"


def normalize_record_for_viewer(
    item: dict[str, Any], default_split: str = "train"
) -> dict[str, Any]:
    """Normalizes any record format (OpenAI, ShareGPT, DPO, Augmented) into a standard viewer item."""
    rec_id = item.get("id", f"rec_{random.randint(100000, 999999)}")
    archetype = item.get("archetype", "direct_identification")
    entity_uri = item.get("entity_uri", "")
    split = item.get("split", default_split)
    persona = parse_persona_from_record(item)
    category = parse_category_from_record(item)
    metadata = item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {}

    user_query = ""
    ground_truth = ""
    system_prompt = "You are an authoritative SI and metrology assistant grounded in the BIPM SI Digital Framework, CODATA fundamental constants, and QUDT ontologies."
    rejected = item.get("rejected", "")
    rejection_reason = item.get("rejection_reason", "")
    is_dpo = (
        bool(item.get("prompt") and item.get("chosen") and item.get("rejected")) or split == "dpo"
    )

    if is_dpo:
        split = "dpo"
        user_query = item.get("prompt", "")
        ground_truth = item.get("chosen", "")
        rejected = item.get("rejected", "")
        rejection_reason = item.get("rejection_reason", "")
    elif "messages" in item and isinstance(item["messages"], list):
        for msg in item["messages"]:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                system_prompt = content
            elif role == "user":
                user_query = content
            elif role == "assistant":
                ground_truth = content
    elif "conversations" in item and isinstance(item["conversations"], list):
        for turn in item["conversations"]:
            from_role = turn.get("from") or turn.get("from_")
            val = turn.get("value", "")
            if from_role == "system":
                system_prompt = val
            elif from_role in ("human", "user"):
                user_query = val
            elif from_role in ("gpt", "assistant"):
                ground_truth = val
    else:
        user_query = (
            item.get("user_query", "") or item.get("canonical_query", "") or item.get("prompt", "")
        )
        ground_truth = (
            item.get("ground_truth_answer", "") or item.get("chosen", "") or item.get("answer", "")
        )

    # Rough token estimation (~3.8 characters per token for scientific text)
    total_chars = len(system_prompt) + len(user_query) + len(ground_truth) + len(rejected)
    approx_tokens = max(1, int(total_chars / 3.8))

    return {
        "id": rec_id,
        "split": split,
        "persona": persona,
        "archetype": archetype,
        "category": category,
        "entity_uri": entity_uri,
        "system_prompt": system_prompt,
        "user_query": user_query,
        "ground_truth_answer": ground_truth,
        "rejected": rejected,
        "rejection_reason": rejection_reason,
        "is_dpo": is_dpo,
        "approx_tokens": approx_tokens,
        "metadata": metadata,
    }


def load_dataset_records(
    dataset_dir_or_file: str | Path,
    max_samples: int | None = 5000,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Loads dataset files from directory or a single file and extracts a balanced stratified sample + global statistics."""
    p = Path(dataset_dir_or_file).resolve()
    all_items: list[dict[str, Any]] = []
    stats = {
        "total_records": 0,
        "splits": {"train": 0, "val": 0, "test": 0, "dpo": 0},
        "personas": {},
        "archetypes": {},
        "categories": {},
        "total_tokens_est": 0,
        "file_inventory": [],
    }

    if p.is_file():
        # Single file loader
        split_guess = "train"
        if "test" in p.name.lower():
            split_guess = "test"
        elif "val" in p.name.lower():
            split_guess = "val"
        elif "dpo" in p.name.lower():
            split_guess = "dpo"

        file_count = 0
        file_bytes = p.stat().st_size
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    norm = normalize_record_for_viewer(raw, default_split=split_guess)
                    all_items.append(norm)
                    file_count += 1
                except Exception:
                    continue

        stats["total_records"] = file_count
        stats["splits"][split_guess] = file_count
        stats["file_inventory"].append(
            {
                "path": p.name,
                "split": split_guess,
                "count": file_count,
                "bytes": file_bytes,
            }
        )
    else:
        # Directory loader: load train.jsonl, val.jsonl, test.jsonl, dpo_preferences.jsonl
        split_files = [
            ("train", p / "train.jsonl"),
            ("val", p / "val.jsonl"),
            ("test", p / "test.jsonl"),
            ("dpo", p / "dpo_preferences.jsonl"),
        ]

        # Also inspect by_persona and by_archetype directories if present
        for split_name, split_path in split_files:
            if not split_path.exists():
                continue
            f_bytes = split_path.stat().st_size
            f_count = 0
            with open(split_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                        norm = normalize_record_for_viewer(raw, default_split=split_name)
                        all_items.append(norm)
                        f_count += 1
                    except Exception:
                        continue
            stats["splits"][split_name] = f_count
            stats["file_inventory"].append(
                {
                    "path": split_path.name,
                    "split": split_name,
                    "count": f_count,
                    "bytes": f_bytes,
                }
            )

    # Compute global stats over all parsed records
    stats["total_records"] = len(all_items)
    for rec in all_items:
        p_name = rec["persona"]
        arch = rec["archetype"]
        cat = rec["category"]
        stats["personas"][p_name] = stats["personas"].get(p_name, 0) + 1
        stats["archetypes"][arch] = stats["archetypes"].get(arch, 0) + 1
        stats["categories"][cat] = stats["categories"].get(cat, 0) + 1
        stats["total_tokens_est"] += rec.get("approx_tokens", 0)

    # Perform stratified sampling if max_samples is requested and dataset exceeds limit
    if max_samples and len(all_items) > max_samples:
        random.seed(seed)
        # Stratify by split & persona
        buckets: dict[str, list[dict[str, Any]]] = {}
        for rec in all_items:
            key = f"{rec['split']}_{rec['persona']}_{rec['archetype']}"
            buckets.setdefault(key, []).append(rec)

        sample_records: list[dict[str, Any]] = []
        # Target proportional sample per bucket
        sample_ratio = max_samples / len(all_items)
        for _, bucket in buckets.items():
            k = max(1, round(len(bucket) * sample_ratio))
            sample_records.extend(random.sample(bucket, min(k, len(bucket))))

        random.shuffle(sample_records)
        selected_records = sample_records[:max_samples]
    else:
        selected_records = all_items

    return selected_records, stats


DATASET_VIEWER_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>__DATASET_TITLE__ - Interactive Dataset Browser & Review Tool</title>

  <!-- Google Fonts -->
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&family=Outfit:wght@500;600;700;800&display=swap" rel="stylesheet" />

  <!-- KaTeX for math rendering -->
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css" />
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>

  <style>
    :root {
      --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      --font-display: 'Outfit', var(--font-sans);
      --font-mono: 'JetBrains Mono', monospace;

      /* Dark Theme (Default) */
      --bg-main: #0a0e17;
      --bg-card: #111827;
      --bg-card-hover: #172236;
      --bg-sidebar: #0d1322;
      --bg-subtle: #1e293b;
      --bg-input: #0f172a;

      --border-color: #24324d;
      --border-subtle: #1c273e;
      --border-focus: #38bdf8;

      --text-primary: #f8fafc;
      --text-secondary: #94a3b8;
      --text-muted: #64748b;
      --text-link: #38bdf8;

      --accent-blue: #3b82f6;
      --accent-cyan: #06b6d4;
      --accent-emerald: #10b981;
      --accent-amber: #f59e0b;
      --accent-rose: #f43f5e;
      --accent-purple: #8b5cf6;
      --accent-indigo: #6366f1;

      /* Splits */
      --split-train: #3b82f6;
      --split-val: #a855f7;
      --split-test: #10b981;
      --split-dpo: #f59e0b;

      /* Archetype Colors */
      --arch-direct: #0ea5e9;
      --arch-dim: #8b5cf6;
      --arch-conv: #10b981;
      --arch-error: #f43f5e;
      --arch-tool: #06b6d4;
      --arch-unc: #f59e0b;

      --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.4);
      --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.5), 0 2px 4px -1px rgba(0, 0, 0, 0.4);
      --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.6), 0 4px 6px -2px rgba(0, 0, 0, 0.4);

      --radius-sm: 6px;
      --radius-md: 10px;
      --radius-lg: 16px;
      --radius-xl: 24px;
    }

    [data-theme="light"] {
      --bg-main: #f8fafc;
      --bg-card: #ffffff;
      --bg-card-hover: #f1f5f9;
      --bg-sidebar: #f8fafc;
      --bg-subtle: #e2e8f0;
      --bg-input: #ffffff;

      --border-color: #cbd5e1;
      --border-subtle: #e2e8f0;
      --border-focus: #0284c7;

      --text-primary: #0f172a;
      --text-secondary: #475569;
      --text-muted: #64748b;
      --text-link: #0284c7;

      --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
      --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
      --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    html, body {
      height: 100%;
      width: 100%;
      margin: 0;
      padding: 0;
      overflow: hidden; /* Strict: window/body NEVER scrolls */
    }

    body {
      font-family: var(--font-sans);
      background-color: var(--bg-main);
      color: var(--text-primary);
      height: 100vh;
      max-height: 100vh;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
      width: 6px;
      height: 6px;
    }
    ::-webkit-scrollbar-track {
      background: transparent;
    }
    ::-webkit-scrollbar-thumb {
      background: var(--border-color);
      border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
      background: var(--text-muted);
    }

    /* Header Bar */
    header {
      flex-shrink: 0;
      background: rgba(13, 19, 34, 0.95);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--border-color);
      padding: 0.65rem 1.25rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      z-index: 100;
      gap: 1rem;
    }
    [data-theme="light"] header {
      background: rgba(255, 255, 255, 0.95);
    }

    .brand-section {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }

    .logo-badge {
      background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
      color: #fff;
      font-family: var(--font-display);
      font-weight: 800;
      font-size: 1rem;
      padding: 0.35rem 0.65rem;
      border-radius: var(--radius-sm);
      letter-spacing: 0.05em;
      box-shadow: 0 0 15px rgba(14, 165, 233, 0.4);
    }

    .brand-title {
      font-family: var(--font-display);
      font-weight: 700;
      font-size: 1.15rem;
      color: var(--text-primary);
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    .brand-title span.version {
      font-size: 0.75rem;
      font-family: var(--font-mono);
      background: var(--bg-subtle);
      color: var(--accent-cyan);
      padding: 0.15rem 0.45rem;
      border-radius: 9999px;
      border: 1px solid var(--border-color);
    }

    /* Navigation Mode Tabs in Header */
    .nav-tabs {
      display: flex;
      align-items: center;
      gap: 0.35rem;
      background: var(--bg-input);
      padding: 0.25rem;
      border-radius: var(--radius-md);
      border: 1px solid var(--border-color);
    }

    .nav-tab-btn {
      background: transparent;
      border: none;
      color: var(--text-secondary);
      font-family: var(--font-sans);
      font-size: 0.82rem;
      font-weight: 600;
      padding: 0.4rem 0.8rem;
      border-radius: var(--radius-sm);
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 0.4rem;
      transition: all 0.15s ease;
    }
    .nav-tab-btn:hover {
      color: var(--text-primary);
      background: var(--bg-subtle);
    }
    .nav-tab-btn.active {
      background: var(--accent-blue);
      color: #fff;
      box-shadow: var(--shadow-sm);
    }

    /* Header Actions */
    .header-actions {
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .btn {
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      padding: 0.4rem 0.75rem;
      font-size: 0.82rem;
      font-weight: 600;
      font-family: var(--font-sans);
      border-radius: var(--radius-sm);
      border: 1px solid var(--border-color);
      background: var(--bg-card);
      color: var(--text-primary);
      cursor: pointer;
      transition: all 0.15s ease;
      text-decoration: none;
      white-space: nowrap;
    }
    .btn:hover {
      background: var(--bg-subtle);
      border-color: var(--border-focus);
    }
    .btn-primary {
      background: var(--accent-blue);
      border-color: var(--accent-blue);
      color: #fff;
    }
    .btn-primary:hover {
      background: #2563eb;
    }
    .btn-emerald {
      background: var(--accent-emerald);
      border-color: var(--accent-emerald);
      color: #fff;
    }
    .btn-emerald:hover {
      background: #059669;
    }
    .btn-icon {
      padding: 0.4rem;
      border-radius: var(--radius-sm);
      color: var(--text-secondary);
    }

    /* Main Container Layout */
    .app-layout {
      display: flex;
      flex: 1;
      min-height: 0;
      height: 0;
      overflow: hidden;
    }

    /* Sidebar / Master List */
    .sidebar {
      width: 380px;
      min-width: 320px;
      max-width: 460px;
      background: var(--bg-sidebar);
      border-right: 1px solid var(--border-color);
      display: flex;
      flex-direction: column;
      height: 100%;
      min-height: 0;
      flex-shrink: 0;
      overflow: hidden;
    }

    .sidebar-header {
      padding: 0.85rem 1rem;
      border-bottom: 1px solid var(--border-color);
      display: flex;
      flex-direction: column;
      gap: 0.65rem;
    }

    .search-box {
      position: relative;
      display: flex;
      align-items: center;
    }
    .search-box input {
      width: 100%;
      background: var(--bg-input);
      border: 1px solid var(--border-color);
      color: var(--text-primary);
      font-size: 0.85rem;
      padding: 0.45rem 0.65rem 0.45rem 2rem;
      border-radius: var(--radius-sm);
      outline: none;
      font-family: var(--font-sans);
      transition: border-color 0.15s ease;
    }
    .search-box input:focus {
      border-color: var(--border-focus);
      box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2);
    }
    .search-box svg {
      position: absolute;
      left: 0.65rem;
      color: var(--text-muted);
      pointer-events: none;
    }
    .search-kbd {
      position: absolute;
      right: 0.5rem;
      font-size: 0.65rem;
      font-family: var(--font-mono);
      background: var(--bg-subtle);
      color: var(--text-muted);
      padding: 0.1rem 0.35rem;
      border-radius: 4px;
      border: 1px solid var(--border-color);
      pointer-events: none;
    }

    /* Filters Row */
    .filter-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.45rem;
    }

    .filter-select {
      background: var(--bg-input);
      border: 1px solid var(--border-color);
      color: var(--text-primary);
      font-size: 0.78rem;
      padding: 0.35rem 0.5rem;
      border-radius: var(--radius-sm);
      outline: none;
      font-family: var(--font-sans);
      width: 100%;
      cursor: pointer;
    }
    .filter-select:focus {
      border-color: var(--border-focus);
    }

    /* Split Quick Selector Bar */
    .split-pills {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 0.35rem;
    }
    .split-pill {
      font-size: 0.72rem;
      font-weight: 700;
      padding: 0.2rem 0.55rem;
      border-radius: 9999px;
      border: 1px solid var(--border-color);
      background: var(--bg-input);
      color: var(--text-secondary);
      cursor: pointer;
      white-space: nowrap;
      transition: all 0.15s ease;
      display: inline-flex;
      align-items: center;
    }
    .split-pill:hover {
      color: var(--text-primary);
      border-color: var(--text-muted);
    }
    .split-pill.active {
      background: var(--accent-blue);
      border-color: var(--accent-blue);
      color: #fff;
    }
    .split-pill[data-split="train"].active {
      background: var(--split-train);
      border-color: var(--split-train);
    }
    .split-pill[data-split="val"].active {
      background: var(--split-val);
      border-color: var(--split-val);
    }
    .split-pill[data-split="test"].active {
      background: var(--split-test);
      border-color: var(--split-test);
    }
    .split-pill[data-split="dpo"].active {
      background: var(--split-dpo);
      border-color: var(--split-dpo);
    }

    /* Sample Counter & Sort Bar */
    .list-meta-bar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0.4rem 1rem;
      background: var(--bg-card);
      border-bottom: 1px solid var(--border-subtle);
      font-size: 0.75rem;
      color: var(--text-secondary);
    }

    /* Sample List Scroll Area */
    .sample-list {
      flex: 1;
      min-height: 0;
      overflow-y: auto;
      overscroll-behavior: contain;
      padding: 0.5rem;
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
    }

    .sample-card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-md);
      padding: 0.75rem;
      cursor: pointer;
      transition: all 0.15s ease;
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
      position: relative;
    }
    .sample-card:hover {
      background: var(--bg-card-hover);
      border-color: var(--border-focus);
      transform: translateY(-1px);
    }
    .sample-card.active {
      background: rgba(14, 165, 233, 0.08);
      border-color: var(--accent-cyan);
      box-shadow: 0 0 0 1px var(--accent-cyan);
    }

    .sample-card-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.5rem;
    }

    .badge-row {
      display: flex;
      align-items: center;
      gap: 0.35rem;
      flex-wrap: wrap;
    }

    .badge {
      font-size: 0.68rem;
      font-weight: 700;
      padding: 0.15rem 0.45rem;
      border-radius: 4px;
      letter-spacing: 0.02em;
      text-transform: uppercase;
      font-family: var(--font-mono);
      display: inline-flex;
      align-items: center;
      gap: 0.25rem;
    }
    .badge-train { background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }
    .badge-val { background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.3); }
    .badge-test { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-dpo { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-persona { background: var(--bg-subtle); color: var(--text-secondary); border: 1px solid var(--border-color); font-weight: 600; text-transform: none; }
    .badge-arch { background: rgba(6, 182, 212, 0.12); color: #22d3ee; border: 1px solid rgba(6, 182, 212, 0.25); text-transform: none; }
    .badge-starred { color: #f59e0b; }

    .sample-card-preview {
      font-size: 0.82rem;
      line-height: 1.35;
      color: var(--text-primary);
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
      font-weight: 500;
    }

    .sample-card-footer {
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 0.72rem;
      color: var(--text-muted);
      font-family: var(--font-mono);
    }

    /* Pagination Bar */
    .pagination-bar {
      padding: 0.5rem 1rem;
      border-top: 1px solid var(--border-color);
      background: var(--bg-sidebar);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.5rem;
    }
    .pagination-controls {
      display: flex;
      align-items: center;
      gap: 0.25rem;
    }
    .page-btn {
      background: var(--bg-input);
      border: 1px solid var(--border-color);
      color: var(--text-primary);
      font-size: 0.75rem;
      padding: 0.25rem 0.5rem;
      border-radius: var(--radius-sm);
      cursor: pointer;
    }
    .page-btn:disabled {
      opacity: 0.4;
      cursor: not-allowed;
    }

    /* Inspector / Detail Area */
    .inspector-pane {
      flex: 1;
      min-height: 0;
      height: 100%;
      background: var(--bg-main);
      display: flex;
      flex-direction: column;
      overflow: hidden; /* Header fixed at top, only content container scrolls */
    }

    .inspector-header {
      flex-shrink: 0;
      padding: 0.85rem 1.5rem;
      border-bottom: 1px solid var(--border-color);
      background: var(--bg-card);
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
      z-index: 10;
    }

    .inspector-top-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
    }

    .inspector-id-block {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }
    .inspector-id {
      font-family: var(--font-mono);
      font-weight: 700;
      font-size: 1.05rem;
      color: var(--text-primary);
    }

    .inspector-toolbar {
      display: flex;
      align-items: center;
      gap: 0.45rem;
    }

    /* Format View Switcher (Rendered / OpenAI / ShareGPT / Raw) */
    .format-switcher {
      display: flex;
      align-items: center;
      gap: 0.25rem;
      background: var(--bg-input);
      padding: 0.2rem;
      border-radius: var(--radius-sm);
      border: 1px solid var(--border-color);
    }
    .format-btn {
      background: transparent;
      border: none;
      color: var(--text-secondary);
      font-size: 0.75rem;
      font-weight: 600;
      padding: 0.25rem 0.6rem;
      border-radius: 4px;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .format-btn.active {
      background: var(--accent-blue);
      color: #fff;
    }

    /* Inspector Content Container */
    .inspector-content {
      flex: 1;
      min-height: 0;
      overflow-y: auto;
      overscroll-behavior: contain;
      padding: 1.5rem;
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
      max-width: 1080px;
      margin: 0 auto;
      width: 100%;
    }

    /* Metadata Bar */
    .provenance-card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-md);
      padding: 1rem 1.25rem;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 1rem;
      box-shadow: var(--shadow-sm);
    }
    .provenance-item {
      display: flex;
      flex-direction: column;
      gap: 0.2rem;
    }
    .provenance-label {
      font-size: 0.72rem;
      font-weight: 700;
      text-transform: uppercase;
      color: var(--text-muted);
      letter-spacing: 0.04em;
    }
    .provenance-val {
      font-size: 0.85rem;
      color: var(--text-primary);
      font-family: var(--font-mono);
      word-break: break-all;
    }
    .provenance-val a {
      color: var(--text-link);
      text-decoration: none;
    }
    .provenance-val a:hover {
      text-decoration: underline;
    }

    /* Section Cards (System, User Query, Assistant Ground Truth) */
    .turn-card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-lg);
      padding: 1.25rem 1.5rem;
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
      box-shadow: var(--shadow-sm);
      position: relative;
    }
    .turn-card.turn-system {
      border-left: 4px solid var(--accent-purple);
      background: rgba(139, 92, 246, 0.03);
    }
    .turn-card.turn-user {
      border-left: 4px solid var(--accent-cyan);
      background: rgba(6, 182, 212, 0.03);
    }
    .turn-card.turn-assistant {
      border-left: 4px solid var(--accent-emerald);
      background: rgba(16, 185, 129, 0.03);
    }
    .turn-card.turn-rejected {
      border-left: 4px solid var(--accent-rose);
      background: rgba(244, 63, 94, 0.03);
    }

    .turn-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid var(--border-subtle);
      padding-bottom: 0.5rem;
    }
    .turn-title {
      font-family: var(--font-display);
      font-weight: 700;
      font-size: 0.92rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      color: var(--text-primary);
    }

    .turn-body {
      font-size: 0.95rem;
      line-height: 1.65;
      color: var(--text-primary);
      word-break: break-word;
    }
    .turn-body p {
      margin-bottom: 0.75rem;
    }
    .turn-body p:last-child {
      margin-bottom: 0;
    }
    .turn-body ul, .turn-body ol {
      margin-left: 1.5rem;
      margin-bottom: 0.75rem;
    }
    .turn-body li {
      margin-bottom: 0.35rem;
    }
    .turn-body strong {
      color: #fff;
    }
    [data-theme="light"] .turn-body strong {
      color: #0f172a;
    }

    /* DPO Arena Side-by-Side View */
    .dpo-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.25rem;
    }
    @media (max-width: 900px) {
      .dpo-grid {
        grid-template-columns: 1fr;
      }
    }

    .dpo-card-chosen {
      border-top: 4px solid var(--accent-emerald);
    }
    .dpo-card-rejected {
      border-top: 4px solid var(--accent-rose);
    }

    .dpo-reason-banner {
      background: rgba(244, 63, 94, 0.1);
      border: 1px solid rgba(244, 63, 94, 0.3);
      color: #fda4af;
      padding: 0.75rem 1rem;
      border-radius: var(--radius-sm);
      font-size: 0.85rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-family: var(--font-sans);
    }

    /* Raw JSON & Code blocks */
    .code-view {
      background: #090d16;
      border: 1px solid var(--border-color);
      border-radius: var(--radius-md);
      padding: 1.25rem;
      font-family: var(--font-mono);
      font-size: 0.85rem;
      line-height: 1.5;
      overflow-x: auto;
      color: #38bdf8;
      white-space: pre-wrap;
    }
    [data-theme="light"] .code-view {
      background: #f1f5f9;
      color: #0369a1;
    }

    /* Modal Styles */
    .modal-backdrop {
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.75);
      backdrop-filter: blur(4px);
      display: none;
      align-items: center;
      justify-content: center;
      z-index: 200;
      padding: 1.5rem;
    }
    .modal-backdrop.active {
      display: flex;
    }

    .modal-window {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-xl);
      max-width: 860px;
      width: 100%;
      max-height: 85vh;
      display: flex;
      flex-direction: column;
      box-shadow: var(--shadow-lg);
      overflow: hidden;
      animation: modalFade 0.2s ease-out;
    }
    @keyframes modalFade {
      from { opacity: 0; transform: scale(0.96); }
      to { opacity: 1; transform: scale(1); }
    }

    .modal-header {
      padding: 1.25rem 1.5rem;
      border-bottom: 1px solid var(--border-color);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .modal-title {
      font-family: var(--font-display);
      font-size: 1.25rem;
      font-weight: 700;
    }

    .modal-body {
      padding: 1.5rem;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
    }

    /* Grid for Persona Matrix View */
    .matrix-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
      gap: 1rem;
    }
    .matrix-card {
      background: var(--bg-input);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-md);
      padding: 1rem;
      cursor: pointer;
      transition: all 0.15s ease;
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }
    .matrix-card:hover {
      border-color: var(--accent-cyan);
      background: var(--bg-subtle);
      transform: translateY(-2px);
    }
    .matrix-card-title {
      font-weight: 700;
      font-size: 0.9rem;
      color: var(--text-primary);
    }
    .matrix-card-union {
      font-size: 0.72rem;
      color: var(--accent-cyan);
      font-family: var(--font-mono);
    }
    .matrix-card-count {
      font-size: 1.15rem;
      font-weight: 800;
      color: var(--text-primary);
      font-family: var(--font-mono);
    }

    /* Stat Dashboard Grid */
    .stats-metric-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 0.75rem;
    }
    .stat-metric-card {
      background: var(--bg-input);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-md);
      padding: 0.85rem 1rem;
      display: flex;
      flex-direction: column;
      gap: 0.2rem;
    }
    .stat-metric-label {
      font-size: 0.72rem;
      color: var(--text-muted);
      font-weight: 700;
      text-transform: uppercase;
    }
    .stat-metric-val {
      font-size: 1.35rem;
      font-weight: 800;
      color: var(--text-primary);
      font-family: var(--font-mono);
    }

    /* Distribution Progress Bars */
    .dist-bar-group {
      display: flex;
      flex-direction: column;
      gap: 0.65rem;
    }
    .dist-row {
      display: flex;
      flex-direction: column;
      gap: 0.25rem;
    }
    .dist-header {
      display: flex;
      justify-content: space-between;
      font-size: 0.8rem;
      color: var(--text-secondary);
    }
    .dist-track {
      height: 8px;
      background: var(--bg-subtle);
      border-radius: 4px;
      overflow: hidden;
    }
    .dist-fill {
      height: 100%;
      border-radius: 4px;
      transition: width 0.3s ease;
    }

    /* Toast Notification */
    .toast-container {
      position: fixed;
      bottom: 1.5rem;
      right: 1.5rem;
      z-index: 300;
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }
    .toast {
      background: rgba(17, 24, 39, 0.95);
      border: 1px solid var(--border-focus);
      color: #fff;
      padding: 0.65rem 1rem;
      border-radius: var(--radius-sm);
      font-size: 0.82rem;
      font-weight: 600;
      box-shadow: var(--shadow-lg);
      animation: toastIn 0.2s ease-out;
    }
    @keyframes toastIn {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }

    /* Drag & Drop File Zone */
    .dropzone {
      border: 2px dashed var(--border-color);
      border-radius: var(--radius-lg);
      padding: 2.5rem;
      text-align: center;
      cursor: pointer;
      transition: all 0.2s ease;
      background: var(--bg-input);
    }
    .dropzone:hover, .dropzone.dragover {
      border-color: var(--accent-cyan);
      background: rgba(6, 182, 212, 0.05);
    }

    /* Highlight Search Terms */
    mark.highlight {
      background: rgba(245, 158, 11, 0.3);
      color: #fbbf24;
      padding: 0.1em 0.25em;
      border-radius: 2px;
    }
  </style>
</head>
<body>

  <!-- Header -->
  <header>
    <div class="brand-section">
      <div class="logo-badge">DRUM-ML</div>
      <div class="brand-title">
        Dataset Explorer & Review Tool
        <span class="version">v0.1.0</span>
      </div>
    </div>

    <!-- Navigation Tabs -->
    <div class="nav-tabs">
      <button class="nav-tab-btn active" id="tabExplorer" onclick="app.switchView('explorer')">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
        Explorer
      </button>
      <button class="nav-tab-btn" id="tabPersonaMatrix" onclick="app.openModal('modalPersonaMatrix')">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path></svg>
        Personas Matrix
      </button>
      <button class="nav-tab-btn" id="tabArchetypes" onclick="app.openModal('modalArchetypes')">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>
        Archetypes
      </button>
      <button class="nav-tab-btn" id="tabStats" onclick="app.openModal('modalStats')">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 20V10"></path><path d="M12 20V4"></path><path d="M6 20v-6"></path></svg>
        Analytics
      </button>
      <button class="nav-tab-btn" id="tabFiles" onclick="app.openModal('modalFiles')">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path></svg>
        Partitions & Subsets
      </button>
    </div>

    <!-- Header Actions -->
    <div class="header-actions">
      <button class="btn" onclick="app.pickRandomSample()" title="Jump to Random Sample (Shortcut: r)">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="16 3 21 3 21 8"></polyline><line x1="4" y1="20" x2="21" y2="3"></line><polyline points="21 16 21 21 16 21"></polyline><line x1="15" y1="15" x2="21" y2="21"></line><line x1="4" y1="4" x2="9" y2="9"></line></svg>
        Random
      </button>
      <button class="btn btn-emerald" onclick="app.exportFilteredJSONL()" title="Export current filtered view as JSONL">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
        Export View
      </button>
      <button class="btn btn-icon" onclick="app.toggleTheme()" title="Toggle Dark / Light Theme (Shortcut: t)">
        <svg id="themeIcon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>
      </button>
      <button class="btn btn-icon" onclick="app.openModal('modalShortcuts')" title="Keyboard Shortcuts (Shortcut: ?)">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
      </button>
    </div>
  </header>

  <!-- App Layout -->
  <div class="app-layout">

    <!-- Left Sidebar / Master List -->
    <aside class="sidebar">
      <div class="sidebar-header">
        <!-- Search Input -->
        <div class="search-box">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
          <input type="text" id="searchInput" placeholder="Search prompts, entities, math..." oninput="app.handleSearch(this.value)" />
          <span class="search-kbd">/</span>
        </div>

        <!-- Split Filter Pills -->
        <div class="split-pills">
          <button class="split-pill active" data-split="all" onclick="app.setSplitFilter('all')">All (<span id="countSplitAll">0</span>)</button>
          <button class="split-pill" data-split="train" onclick="app.setSplitFilter('train')">Train (<span id="countSplitTrain">0</span>)</button>
          <button class="split-pill" data-split="val" onclick="app.setSplitFilter('val')">Val (<span id="countSplitVal">0</span>)</button>
          <button class="split-pill" data-split="test" onclick="app.setSplitFilter('test')">Test (<span id="countSplitTest">0</span>)</button>
          <button class="split-pill" data-split="dpo" onclick="app.setSplitFilter('dpo')">DPO (<span id="countSplitDpo">0</span>)</button>
        </div>

        <!-- Filter Dropdowns Grid -->
        <div class="filter-grid">
          <select id="personaFilter" class="filter-select" onchange="app.handleFilterChange()">
            <option value="all">All Personas (35+)</option>
          </select>
          <select id="archetypeFilter" class="filter-select" onchange="app.handleFilterChange()">
            <option value="all">All Archetypes (6)</option>
            <option value="direct_identification">1. Direct Identification</option>
            <option value="dimensional_decomposition">2. Dimensional Decomposition</option>
            <option value="conversion_scaling">3. Conversion & Scaling</option>
            <option value="error_detection">4. Error Detection</option>
            <option value="semantic_tool_use">5. Semantic Tool Use</option>
            <option value="metrological_uncertainty">6. Metrological Uncertainty</option>
          </select>
        </div>
      </div>

      <!-- Sample List Meta Bar -->
      <div class="list-meta-bar">
        <span>Showing <strong id="visibleSampleCount" style="color:var(--text-primary)">0</strong> samples</span>
        <select id="sortSelect" class="filter-select" style="width:auto; padding:0.2rem 0.4rem;" onchange="app.handleSortChange(this.value)">
          <option value="default">Default Order</option>
          <option value="prompt_len_desc">Longest Query</option>
          <option value="prompt_len_asc">Shortest Query</option>
          <option value="persona">By Persona</option>
          <option value="archetype">By Archetype</option>
        </select>
      </div>

      <!-- Scrollable List of Samples -->
      <div class="sample-list" id="sampleListContainer">
        <!-- Injected via JavaScript -->
      </div>

      <!-- Pagination -->
      <div class="pagination-bar">
        <span style="font-size:0.75rem; color:var(--text-muted);">
          Page <strong id="currentPageNum" style="color:var(--text-primary)">1</strong> of <span id="totalPagesNum">1</span>
        </span>
        <div class="pagination-controls">
          <button class="page-btn" id="btnPrevPage" onclick="app.prevPage()" title="Previous Page">Prev</button>
          <button class="page-btn" id="btnNextPage" onclick="app.nextPage()" title="Next Page">Next</button>
        </div>
      </div>
    </aside>

    <!-- Main Detail / Inspector Pane -->
    <main class="inspector-pane" id="inspectorPane">

      <!-- Top Inspector Sticky Header -->
      <div class="inspector-header">
        <div class="inspector-top-row">
          <div class="inspector-id-block">
            <span class="inspector-id" id="insId">rec_000000</span>
            <span class="badge badge-train" id="insSplitBadge">TRAIN</span>
            <span class="badge badge-persona" id="insPersonaBadge">general_user</span>
            <span class="badge badge-arch" id="insArchetypeBadge">direct_identification</span>
          </div>

          <div class="inspector-toolbar">
            <!-- Format Switcher -->
            <div class="format-switcher">
              <button class="format-btn active" id="btnFmtRendered" onclick="app.setInspectorFormat('rendered')">Rendered</button>
              <button class="format-btn" id="btnFmtOpenAI" onclick="app.setInspectorFormat('openai')">OpenAI</button>
              <button class="format-btn" id="btnFmtShareGPT" onclick="app.setInspectorFormat('sharegpt')">ShareGPT</button>
              <button class="format-btn" id="btnFmtRaw" onclick="app.setInspectorFormat('raw')">JSON</button>
            </div>

            <!-- Action Buttons -->
            <button class="btn" onclick="app.copyCurrentRecordJSON()" title="Copy Full Record JSON">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
              Copy
            </button>
            <button class="btn" id="btnBookmark" onclick="app.toggleBookmark()" title="Bookmark this sample">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>
            </button>
          </div>
        </div>
      </div>

      <!-- Detail Body Content Container -->
      <div class="inspector-content" id="inspectorContent">
        <!-- Rendered dynamically -->
      </div>
    </main>

  </div>

  <!-- MODAL 1: Personas Matrix -->
  <div class="modal-backdrop" id="modalPersonaMatrix">
    <div class="modal-window">
      <div class="modal-header">
        <h2 class="modal-title">CODATA DRUM Persona Taxonomy Matrix (35+ Profiles)</h2>
        <button class="btn btn-icon" onclick="app.closeModal('modalPersonaMatrix')">✕</button>
      </div>
      <div class="modal-body">
        <p style="font-size:0.85rem; color:var(--text-secondary);">
          Each persona reflects real-world metrological nuance from International Scientific Unions, National Metrology Institutes (BIPM, NIST, NRC), and engineering domains. Click any card to drill down into that persona's training samples.
        </p>
        <div class="matrix-grid" id="personaMatrixGrid">
          <!-- Injected via JavaScript -->
        </div>
      </div>
    </div>
  </div>

  <!-- MODAL 2: Pedagogical Archetypes -->
  <div class="modal-backdrop" id="modalArchetypes">
    <div class="modal-window">
      <div class="modal-header">
        <h2 class="modal-title">The 6 Pedagogical Metrology Archetypes</h2>
        <button class="btn btn-icon" onclick="app.closeModal('modalArchetypes')">✕</button>
      </div>
      <div class="modal-body">
        <div style="display:flex; flex-direction:column; gap:1rem;" id="archetypesGuideContainer">
          <!-- Injected via JavaScript -->
        </div>
      </div>
    </div>
  </div>

  <!-- MODAL 3: Analytics & Token Geometry -->
  <div class="modal-backdrop" id="modalStats">
    <div class="modal-window">
      <div class="modal-header">
        <h2 class="modal-title">Dataset Metrics & Token Geometry</h2>
        <button class="btn btn-icon" onclick="app.closeModal('modalStats')">✕</button>
      </div>
      <div class="modal-body">
        <div class="stats-metric-grid" id="analyticsMetricGrid">
          <!-- Injected via JS -->
        </div>

        <h3 style="font-size:0.95rem; font-weight:700; margin-top:0.5rem;">Dataset Split Distribution</h3>
        <div class="dist-bar-group" id="splitDistBars"></div>

        <h3 style="font-size:0.95rem; font-weight:700; margin-top:0.5rem;">Archetype Breakdown</h3>
        <div class="dist-bar-group" id="archetypeDistBars"></div>

        <h3 style="font-size:0.95rem; font-weight:700; margin-top:0.5rem;">Top Personas</h3>
        <div class="dist-bar-group" id="personaDistBars"></div>
      </div>
    </div>
  </div>

  <!-- MODAL 4: Partitions & File Subsets -->
  <div class="modal-backdrop" id="modalFiles">
    <div class="modal-window">
      <div class="modal-header">
        <h2 class="modal-title">Partitioned Dataset Files & Subsets</h2>
        <button class="btn btn-icon" onclick="app.closeModal('modalFiles')">✕</button>
      </div>
      <div class="modal-body">
        <p style="font-size:0.85rem; color:var(--text-secondary);">
          DRUM-ML breaks down the corpus into modular files per persona, archetype, category, and split.
          Use the Hugging Face code snippet below to load any individual partition into Python:
        </p>
        <pre class="code-view">from datasets import load_dataset

# Load a specific persona subset (e.g. Academic Metrologist training split)
ds = load_dataset("json", data_files={"train": "by_persona/academic_metrologist_train.jsonl"})

# Load a specific pedagogical archetype (e.g. Dimensional Decomposition)
ds_arch = load_dataset("json", data_files={"train": "by_archetype/dimensional_decomposition.jsonl"})</pre>

        <h3 style="font-size:0.95rem; font-weight:700; margin-top:0.5rem;">Generated File Inventory</h3>
        <div id="fileInventoryTable" style="max-height: 250px; overflow-y: auto;"></div>

        <h3 style="font-size:0.95rem; font-weight:700; margin-top:0.5rem;">Drop / Load Local Custom JSONL File</h3>
        <div class="dropzone" id="fileDropzone" onclick="document.getElementById('fileInput').click()">
          <input type="file" id="fileInput" accept=".jsonl,.json" style="display:none;" onchange="app.handleCustomFileUpload(event)" />
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-bottom:0.5rem; color:var(--accent-cyan);"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>
          <div style="font-weight:700; font-size:0.9rem;">Drop a dataset file here (.jsonl or .json)</div>
          <div style="font-size:0.75rem; color:var(--text-muted); margin-top:0.25rem;">Immediately explore any DRUM-ML partitioned file in the browser</div>
        </div>
      </div>
    </div>
  </div>

  <!-- MODAL 5: Keyboard Shortcuts -->
  <div class="modal-backdrop" id="modalShortcuts">
    <div class="modal-window" style="max-width: 480px;">
      <div class="modal-header">
        <h2 class="modal-title">Keyboard Shortcuts</h2>
        <button class="btn btn-icon" onclick="app.closeModal('modalShortcuts')">✕</button>
      </div>
      <div class="modal-body">
        <div style="display:flex; flex-direction:column; gap:0.5rem; font-size:0.85rem;">
          <div style="display:flex; justify-content:space-between;"><span>Next sample</span><kbd class="search-kbd">j</kbd> / <kbd class="search-kbd">↓</kbd></div>
          <div style="display:flex; justify-content:space-between;"><span>Previous sample</span><kbd class="search-kbd">k</kbd> / <kbd class="search-kbd">↑</kbd></div>
          <div style="display:flex; justify-content:space-between;"><span>Focus search</span><kbd class="search-kbd">/</kbd></div>
          <div style="display:flex; justify-content:space-between;"><span>Random sample</span><kbd class="search-kbd">r</kbd></div>
          <div style="display:flex; justify-content:space-between;"><span>Toggle Dark / Light Theme</span><kbd class="search-kbd">t</kbd></div>
          <div style="display:flex; justify-content:space-between;"><span>Switch view format</span><kbd class="search-kbd">1</kbd> - <kbd class="search-kbd">4</kbd></div>
          <div style="display:flex; justify-content:space-between;"><span>Close modals</span><kbd class="search-kbd">Esc</kbd></div>
        </div>
      </div>
    </div>
  </div>

  <!-- Toast Container -->
  <div class="toast-container" id="toastContainer"></div>

  <!-- Embedded Dataset & Statistics Data -->
  <script>
    window.EMBEDDED_DATA = __DATA_JSON__;
    window.GLOBAL_STATS = __STATS_JSON__;
  </script>

  <!-- Application Logic -->
  <script>
    (function() {
      "use strict";

      const PERSONA_LABELS = {
        "general_user": { name: "General User", union: "Everyday Metrology", icon: "🌐" },
        "physics_student": { name: "Physics Student", union: "Educational Metrology", icon: "🎓" },
        "academic_metrologist": { name: "Academic Metrologist", union: "BIPM / VIM3 Metrology", icon: "📐" },
        "nist_metrologist": { name: "NIST Metrologist", union: "NIST (USA)", icon: "🏛️" },
        "nrc_metrologist": { name: "NRC Metrologist", union: "NRC (Canada)", icon: "🍁" },
        "isc_science_policy": { name: "ISC Science Policy", union: "International Science Council", icon: "🌍" },
        "iupap_physicist": { name: "IUPAP Physicist", union: "Pure & Applied Physics", icon: "⚛️" },
        "iupac_chemist": { name: "IUPAC Chemist", union: "Pure & Applied Chemistry", icon: "🧪" },
        "iau_astronomer": { name: "IAU Astronomer", union: "Astronomical Union", icon: "🔭" },
        "iucr_crystallographer": { name: "IUCr Crystallographer", union: "Crystallography", icon: "💎" },
        "imu_mathematician": { name: "IMU Mathematician", union: "Mathematical Union", icon: "🔢" },
        "ursi_radio_scientist": { name: "URSI Radio Scientist", union: "Radio Science", icon: "📡" },
        "iutam_mechanics_engineer": { name: "IUTAM Mechanics", union: "Theoretical & Applied Mechanics", icon: "⚙️" },
        "aerospace_propulsion_engineer": { name: "Aerospace Propulsion", union: "AIAA / Propulsion", icon: "🚀" },
        "particle_physicist": { name: "Particle Physicist", union: "High Energy Physics (CERN)", icon: "💥" },
        "iugg_geodesist_geophysicist": { name: "IUGG Geodesist", union: "Geodesy & Geophysics", icon: "🗺️" },
        "igu_geographer": { name: "IGU Geographer", union: "Geographical Union", icon: "🧭" },
        "isprs_photogrammetrist": { name: "ISPRS Photogrammetry", union: "Remote Sensing", icon: "🛰️" },
        "isde_digital_earth": { name: "ISDE Digital Earth", union: "Digital Earth Society", icon: "🌐" },
        "iuss_soil_scientist": { name: "IUSS Soil Scientist", union: "Soil Science", icon: "🌱" },
        "energy_environmental_scientist": { name: "Energy & Env Scientist", union: "Environmental Metrology", icon: "⚡" },
        "iubs_biologist": { name: "IUBS Biologist", union: "Biological Sciences", icon: "🧬" },
        "iuis_immunologist": { name: "IUIS Immunologist", union: "Immunological Societies", icon: "🛡️" },
        "iuphar_pharmacologist": { name: "IUPHAR Pharmacologist", union: "Basic & Clinical Pharmacology", icon: "💊" },
        "iups_physiologist": { name: "IUPS Physiologist", union: "Physiological Sciences", icon: "🫀" },
        "iutox_toxicologist": { name: "IUTOX Toxicologist", union: "Toxicology", icon: "☣️" },
        "iuns_nutritionist": { name: "IUNS Nutritionist", union: "Nutritional Sciences", icon: "🥗" },
        "iufost_food_scientist": { name: "IUFoST Food Scientist", union: "Food Science & Tech", icon: "🌾" },
        "iomp_medical_physicist": { name: "IOMP Medical Physicist", union: "Medical Physics", icon: "🩺" },
        "ifmbe_biomedical_engineer": { name: "IFMBE Biomedical", union: "Biomedical Engineering", icon: "🩻" },
        "iupsys_psychologist": { name: "IUPsyS Psychologist", union: "Psychological Science", icon: "🧠" },
        "isa_sociologist": { name: "ISA Sociologist", union: "Sociological Association", icon: "👥" },
        "iussp_demographer": { name: "IUSSP Demographer", union: "Population Studies", icon: "📊" },
        "wau_anthropologist": { name: "WAU Anthropologist", union: "Anthropological Union", icon: "🗿" },
        "four_s_science_studies": { name: "4S Social Studies", union: "Social Studies of Science", icon: "📜" },
        "iuhpst_historian_philosopher": { name: "IUHPST Historian", union: "History & Philosophy of Science", icon: "📚" }
      };

      const ARCHETYPE_INFO = {
        "direct_identification": {
          title: "1. Direct Identification & Symbols",
          desc: "Mapping physical quantities, base and derived units, defining constants, and symbols in SI/QUDT.",
          color: "var(--arch-direct)"
        },
        "dimensional_decomposition": {
          title: "2. Dimensional Decomposition",
          desc: "Decomposing units into the 7 base SI dimensions [L, M, T, I, Theta, N, J] with exact power vectors.",
          color: "var(--arch-dim)"
        },
        "conversion_scaling": {
          title: "3. Conversion & Scaling Chains",
          desc: "Converting non-SI, imperial, and legacy scientific units to coherent SI with exact multipliers and affine offsets.",
          color: "var(--arch-conv)"
        },
        "error_detection": {
          title: "4. Dimensional Error Detection",
          desc: "Auditing equations for dimensional homogeneity violations, invalid sums, and prefix incompatibilities.",
          color: "var(--arch-error)"
        },
        "semantic_tool_use": {
          title: "5. Semantic Tool Use & Serialization",
          desc: "Executing SPARQL queries, Pint symbolic operations, and generating JSON-LD / Turtle serialization.",
          color: "var(--arch-tool)"
        },
        "metrological_uncertainty": {
          title: "6. Metrological Uncertainty & GUM",
          desc: "Evaluating standard and relative uncertainties, coverage factors (k), and defining constant exactness.",
          color: "var(--arch-unc)"
        }
      };

      class DatasetViewerApp {
        constructor() {
          this.allRecords = window.EMBEDDED_DATA || [];
          this.stats = window.GLOBAL_STATS || { total_records: this.allRecords.length, splits: {} };
          this.filteredRecords = [];
          this.currentSplit = "all";
          this.currentPersona = "all";
          this.currentArchetype = "all";
          this.searchQuery = "";
          this.currentSort = "default";
          this.currentIndex = 0;
          this.pageSize = 50;
          this.currentPage = 1;
          this.activeFormat = "rendered"; // "rendered" | "openai" | "sharegpt" | "raw"
          this.bookmarkedIds = new Set(JSON.parse(localStorage.getItem("drum_ml_bookmarks") || "[]"));
        }

        init() {
          this.populateFilterDropdowns();
          this.updateSplitCounters();
          this.applyFilters();
          this.renderPersonaMatrix();
          this.renderArchetypesGuide();
          this.renderAnalytics();
          this.renderFileInventory();
          this.setupKeyboardShortcuts();
          this.loadTheme();
        }

        populateFilterDropdowns() {
          const personaSelect = document.getElementById("personaFilter");
          const personasPresent = new Set();
          this.allRecords.forEach(r => personasPresent.add(r.persona));

          // Sort personas alphabetically by display name
          const sorted = Array.from(personasPresent).sort();
          sorted.forEach(pKey => {
            const opt = document.createElement("option");
            opt.value = pKey;
            const pInfo = PERSONA_LABELS[pKey] || { name: pKey, union: "" };
            opt.textContent = `${pInfo.icon || "•"} ${pInfo.name}`;
            personaSelect.appendChild(opt);
          });
        }

        updateSplitCounters() {
          const counts = { all: this.allRecords.length, train: 0, val: 0, test: 0, dpo: 0 };
          this.allRecords.forEach(r => {
            if (counts[r.split] !== undefined) counts[r.split]++;
          });
          document.getElementById("countSplitAll").textContent = (this.stats.total_records || counts.all).toLocaleString();
          document.getElementById("countSplitTrain").textContent = (this.stats.splits?.train || counts.train).toLocaleString();
          document.getElementById("countSplitVal").textContent = (this.stats.splits?.val || counts.val).toLocaleString();
          document.getElementById("countSplitTest").textContent = (this.stats.splits?.test || counts.test).toLocaleString();
          document.getElementById("countSplitDpo").textContent = (this.stats.splits?.dpo || counts.dpo).toLocaleString();
        }

        setSplitFilter(splitName) {
          this.currentSplit = splitName;
          document.querySelectorAll(".split-pill").forEach(btn => {
            btn.classList.toggle("active", btn.dataset.split === splitName);
          });
          this.currentPage = 1;
          this.applyFilters();
        }

        handleFilterChange() {
          this.currentPersona = document.getElementById("personaFilter").value;
          this.currentArchetype = document.getElementById("archetypeFilter").value;
          this.currentPage = 1;
          this.applyFilters();
        }

        handleSearch(query) {
          this.searchQuery = query.trim().toLowerCase();
          this.currentPage = 1;
          this.applyFilters();
        }

        handleSortChange(val) {
          this.currentSort = val;
          this.applyFilters();
        }

        applyFilters() {
          let list = this.allRecords.filter(r => {
            if (this.currentSplit !== "all" && r.split !== this.currentSplit) return false;
            if (this.currentPersona !== "all" && r.persona !== this.currentPersona) return false;
            if (this.currentArchetype !== "all" && r.archetype !== this.currentArchetype) return false;
            if (this.searchQuery) {
              const q = this.searchQuery;
              const matchesQuery = (r.user_query || "").toLowerCase().includes(q);
              const matchesAns = (r.ground_truth_answer || "").toLowerCase().includes(q);
              const matchesUri = (r.entity_uri || "").toLowerCase().includes(q);
              const matchesId = (r.id || "").toLowerCase().includes(q);
              const matchesRej = (r.rejected || "").toLowerCase().includes(q);
              if (!matchesQuery && !matchesAns && !matchesUri && !matchesId && !matchesRej) return false;
            }
            return true;
          });

          // Sorting
          if (this.currentSort === "prompt_len_desc") {
            list.sort((a, b) => (b.user_query?.length || 0) - (a.user_query?.length || 0));
          } else if (this.currentSort === "prompt_len_asc") {
            list.sort((a, b) => (a.user_query?.length || 0) - (b.user_query?.length || 0));
          } else if (this.currentSort === "persona") {
            list.sort((a, b) => a.persona.localeCompare(b.persona));
          } else if (this.currentSort === "archetype") {
            list.sort((a, b) => a.archetype.localeCompare(b.archetype));
          }

          this.filteredRecords = list;
          document.getElementById("visibleSampleCount").textContent = list.length.toLocaleString();

          // Reset page if out of bounds
          const maxPages = Math.max(1, Math.ceil(this.filteredRecords.length / this.pageSize));
          if (this.currentPage > maxPages) this.currentPage = 1;

          this.renderSampleList();
          this.updatePagination();

          if (this.filteredRecords.length > 0) {
            if (this.currentIndex >= this.filteredRecords.length) this.currentIndex = 0;
            this.renderInspector();
          } else {
            this.renderEmptyInspector();
          }
        }

        renderSampleList() {
          const container = document.getElementById("sampleListContainer");
          container.innerHTML = "";

          const startIdx = (this.currentPage - 1) * this.pageSize;
          const pageItems = this.filteredRecords.slice(startIdx, startIdx + this.pageSize);

          if (pageItems.length === 0) {
            container.innerHTML = `<div style="padding:2rem 1rem; text-align:center; color:var(--text-muted); font-size:0.85rem;">No samples match current filter criteria.</div>`;
            return;
          }

          pageItems.forEach((rec, localIdx) => {
            const globalIdx = startIdx + localIdx;
            const card = document.createElement("div");
            card.className = `sample-card ${globalIdx === this.currentIndex ? "active" : ""}`;
            card.onclick = () => this.selectSample(globalIdx);

            const splitClass = `badge-${rec.split || "train"}`;
            const pInfo = PERSONA_LABELS[rec.persona] || { name: rec.persona };
            const isBookmarked = this.bookmarkedIds.has(rec.id);

            card.innerHTML = `
              <div class="sample-card-header">
                <div class="badge-row">
                  <span class="badge ${splitClass}">${rec.split || "TRAIN"}</span>
                  <span class="badge badge-persona">${pInfo.name}</span>
                </div>
                ${isBookmarked ? `<span class="badge badge-starred">★</span>` : ""}
              </div>
              <div class="sample-card-preview">${this.highlightMatch(rec.user_query || rec.prompt || "No query")}</div>
              <div class="sample-card-footer">
                <span>${rec.id}</span>
                <span>~${rec.approx_tokens || 0} tok</span>
              </div>
            `;
            container.appendChild(card);
          });
        }

        updatePagination() {
          const totalPages = Math.max(1, Math.ceil(this.filteredRecords.length / this.pageSize));
          document.getElementById("currentPageNum").textContent = this.currentPage;
          document.getElementById("totalPagesNum").textContent = totalPages;
          document.getElementById("btnPrevPage").disabled = this.currentPage <= 1;
          document.getElementById("btnNextPage").disabled = this.currentPage >= totalPages;
        }

        prevPage() {
          if (this.currentPage > 1) {
            this.currentPage--;
            this.applyFilters();
          }
        }

        nextPage() {
          const totalPages = Math.ceil(this.filteredRecords.length / this.pageSize);
          if (this.currentPage < totalPages) {
            this.currentPage++;
            this.applyFilters();
          }
        }

        selectSample(idx) {
          this.currentIndex = idx;
          this.renderSampleList();
          this.renderInspector();
          this.scrollActiveCardIntoView();
        }

        scrollActiveCardIntoView() {
          const activeCard = document.querySelector("#sampleListContainer .sample-card.active");
          if (activeCard) {
            activeCard.scrollIntoView({ block: "nearest", behavior: "smooth" });
          }
        }

        pickRandomSample() {
          if (this.filteredRecords.length === 0) return;
          const randIdx = Math.floor(Math.random() * this.filteredRecords.length);
          this.currentIndex = randIdx;
          this.currentPage = Math.floor(randIdx / this.pageSize) + 1;
          this.renderSampleList();
          this.updatePagination();
          this.renderInspector();
          this.scrollActiveCardIntoView();
          this.showToast(`Jumped to random sample #${randIdx + 1}`);
        }

        setInspectorFormat(fmt) {
          this.activeFormat = fmt;
          document.querySelectorAll(".format-btn").forEach(btn => btn.classList.remove("active"));
          const btnId = fmt === "rendered" ? "btnFmtRendered" : fmt === "openai" ? "btnFmtOpenAI" : fmt === "sharegpt" ? "btnFmtShareGPT" : "btnFmtRaw";
          document.getElementById(btnId)?.classList.add("active");
          this.renderInspector();
        }

        renderInspector() {
          const rec = this.filteredRecords[this.currentIndex];
          if (!rec) {
            this.renderEmptyInspector();
            return;
          }

          // Always reset scroll to top of inspector content container
          const container = document.getElementById("inspectorContent");
          if (container) {
            container.scrollTop = 0;
            container.innerHTML = "";
          }
          const pane = document.getElementById("inspectorPane");
          if (pane) pane.scrollTop = 0;

          // Header badges
          document.getElementById("insId").textContent = rec.id;
          const splitBadge = document.getElementById("insSplitBadge");
          splitBadge.className = `badge badge-${rec.split || "train"}`;
          splitBadge.textContent = (rec.split || "TRAIN").toUpperCase();

          const pInfo = PERSONA_LABELS[rec.persona] || { name: rec.persona, union: "" };
          document.getElementById("insPersonaBadge").textContent = `${pInfo.icon || "•"} ${pInfo.name}`;
          document.getElementById("insArchetypeBadge").textContent = (rec.archetype || "").replace(/_/g, " ");

          const isBookmarked = this.bookmarkedIds.has(rec.id);
          const bookmarkBtn = document.getElementById("btnBookmark");
          bookmarkBtn.style.color = isBookmarked ? "#f59e0b" : "inherit";

          if (this.activeFormat === "rendered") {
            this.renderRenderedView(container, rec);
          } else if (this.activeFormat === "openai") {
            this.renderOpenAIView(container, rec);
          } else if (this.activeFormat === "sharegpt") {
            this.renderShareGPTView(container, rec);
          } else {
            this.renderRawJSONView(container, rec);
          }

          // Trigger KaTeX rendering
          if (window.renderMathInElement) {
            renderMathInElement(container, {
              delimiters: [
                { left: "$$", right: "$$", display: true },
                { left: "$", right: "$", display: false },
                { left: "\\\\[", right: "\\\\]", display: true },
                { left: "\\\\(", right: "\\\\)", display: false }
              ],
              throwOnError: false
            });
          }
        }

        renderRenderedView(container, rec) {
          // Provenance Bar
          const provCard = document.createElement("div");
          provCard.className = "provenance-card";
          provCard.innerHTML = `
            <div class="provenance-item">
              <span class="provenance-label">Scientific Union / Domain</span>
              <span class="provenance-val" style="color:var(--accent-cyan);">${PERSONA_LABELS[rec.persona]?.union || rec.persona}</span>
            </div>
            <div class="provenance-item">
              <span class="provenance-label">Archetype Objective</span>
              <span class="provenance-val">${(rec.archetype || "").replace(/_/g, " ").toUpperCase()}</span>
            </div>
            <div class="provenance-item">
              <span class="provenance-label">Entity Canonical URI</span>
              <span class="provenance-val"><a href="${rec.entity_uri}" target="_blank">${rec.entity_uri ? rec.entity_uri.split("/").pop() : "N/A"}</a></span>
            </div>
            <div class="provenance-item">
              <span class="provenance-label">Estimated Length</span>
              <span class="provenance-val">~${rec.approx_tokens} tokens (${(rec.user_query?.length || 0) + (rec.ground_truth_answer?.length || 0)} chars)</span>
            </div>
          `;
          container.appendChild(provCard);

          if (rec.is_dpo || rec.split === "dpo") {
            // DPO Side-by-Side Arena
            if (rec.rejection_reason) {
              const banner = document.createElement("div");
              banner.className = "dpo-reason-banner";
              banner.innerHTML = `<strong>Audit Gate Diagnostic:</strong> ${rec.rejection_reason}`;
              container.appendChild(banner);
            }

            // User Prompt
            const promptCard = document.createElement("div");
            promptCard.className = "turn-card turn-user";
            promptCard.innerHTML = `
              <div class="turn-header">
                <div class="turn-title">Prompt (${PERSONA_LABELS[rec.persona]?.name || "User"})</div>
                <button class="btn" style="padding:0.2rem 0.5rem; font-size:0.75rem;" onclick="app.copyText('${this.escapeQuotes(rec.user_query)}')">Copy</button>
              </div>
              <div class="turn-body">${this.formatMarkdown(rec.user_query)}</div>
            `;
            container.appendChild(promptCard);

            // Side-by-Side Compare
            const dpoGrid = document.createElement("div");
            dpoGrid.className = "dpo-grid";

            const chosenCard = document.createElement("div");
            chosenCard.className = "turn-card turn-assistant dpo-card-chosen";
            chosenCard.innerHTML = `
              <div class="turn-header">
                <div class="turn-title" style="color:var(--accent-emerald);">✓ Chosen (Ground Truth Approved)</div>
                <button class="btn" style="padding:0.2rem 0.5rem; font-size:0.75rem;" onclick="app.copyText('${this.escapeQuotes(rec.ground_truth_answer)}')">Copy</button>
              </div>
              <div class="turn-body">${this.formatMarkdown(rec.ground_truth_answer)}</div>
            `;

            const rejectedCard = document.createElement("div");
            rejectedCard.className = "turn-card turn-rejected dpo-card-rejected";
            rejectedCard.innerHTML = `
              <div class="turn-header">
                <div class="turn-title" style="color:var(--accent-rose);">✗ Rejected (Audit Gate Failure)</div>
                <button class="btn" style="padding:0.2rem 0.5rem; font-size:0.75rem;" onclick="app.copyText('${this.escapeQuotes(rec.rejected)}')">Copy</button>
              </div>
              <div class="turn-body">${this.formatMarkdown(rec.rejected || "None provided")}</div>
            `;

            dpoGrid.appendChild(chosenCard);
            dpoGrid.appendChild(rejectedCard);
            container.appendChild(dpoGrid);

          } else {
            // Standard User + Assistant Conversation
            if (rec.system_prompt) {
              const sysCard = document.createElement("div");
              sysCard.className = "turn-card turn-system";
              sysCard.innerHTML = `
                <div class="turn-header">
                  <div class="turn-title" style="color:var(--accent-purple);">System Instruction</div>
                </div>
                <div class="turn-body" style="font-size:0.85rem; color:var(--text-secondary);">${rec.system_prompt}</div>
              `;
              container.appendChild(sysCard);
            }

            const userCard = document.createElement("div");
            userCard.className = "turn-card turn-user";
            userCard.innerHTML = `
              <div class="turn-header">
                <div class="turn-title" style="color:var(--accent-cyan);">${PERSONA_LABELS[rec.persona]?.name || "User"} Prompt</div>
                <button class="btn" style="padding:0.2rem 0.5rem; font-size:0.75rem;" onclick="app.copyText('${this.escapeQuotes(rec.user_query)}')">Copy</button>
              </div>
              <div class="turn-body">${this.formatMarkdown(rec.user_query)}</div>
            `;
            container.appendChild(userCard);

            const asstCard = document.createElement("div");
            asstCard.className = "turn-card turn-assistant";
            asstCard.innerHTML = `
              <div class="turn-header">
                <div class="turn-title" style="color:var(--accent-emerald);">SI Metrology Assistant Ground Truth</div>
                <button class="btn" style="padding:0.2rem 0.5rem; font-size:0.75rem;" onclick="app.copyText('${this.escapeQuotes(rec.ground_truth_answer)}')">Copy</button>
              </div>
              <div class="turn-body">${this.formatMarkdown(rec.ground_truth_answer)}</div>
            `;
            container.appendChild(asstCard);
          }
        }

        renderOpenAIView(container, rec) {
          const messages = [
            { role: "system", content: rec.system_prompt },
            { role: "user", content: rec.user_query },
            { role: "assistant", content: rec.ground_truth_answer }
          ];
          const obj = {
            id: rec.id,
            archetype: rec.archetype,
            entity_uri: rec.entity_uri,
            messages: messages,
            metadata: { persona: rec.persona, category: rec.category, ...rec.metadata }
          };

          const pre = document.createElement("pre");
          pre.className = "code-view";
          pre.textContent = JSON.stringify(obj, null, 2);
          container.appendChild(pre);
        }

        renderShareGPTView(container, rec) {
          const obj = {
            id: rec.id,
            conversations: [
              { from: "system", value: rec.system_prompt },
              { from: "human", value: rec.user_query },
              { from: "gpt", value: rec.ground_truth_answer }
            ],
            metadata: { persona: rec.persona, archetype: rec.archetype, entity_uri: rec.entity_uri }
          };

          const pre = document.createElement("pre");
          pre.className = "code-view";
          pre.textContent = JSON.stringify(obj, null, 2);
          container.appendChild(pre);
        }

        renderRawJSONView(container, rec) {
          const pre = document.createElement("pre");
          pre.className = "code-view";
          pre.textContent = JSON.stringify(rec, null, 2);
          container.appendChild(pre);
        }

        renderEmptyInspector() {
          const container = document.getElementById("inspectorContent");
          container.innerHTML = `
            <div style="padding:4rem 2rem; text-align:center; color:var(--text-muted);">
              <h3>No Record Selected</h3>
              <p style="margin-top:0.5rem; font-size:0.85rem;">Adjust search or filter criteria in the left sidebar to display records.</p>
            </div>
          `;
        }

        renderPersonaMatrix() {
          const grid = document.getElementById("personaMatrixGrid");
          grid.innerHTML = "";

          const personaCounts = {};
          this.allRecords.forEach(r => {
            personaCounts[r.persona] = (personaCounts[r.persona] || 0) + 1;
          });

          Object.keys(PERSONA_LABELS).forEach(pKey => {
            const count = this.stats.personas?.[pKey] || personaCounts[pKey] || 0;
            const pInfo = PERSONA_LABELS[pKey];
            const card = document.createElement("div");
            card.className = "matrix-card";
            card.onclick = () => {
              this.closeModal("modalPersonaMatrix");
              document.getElementById("personaFilter").value = pKey;
              this.handleFilterChange();
            };
            card.innerHTML = `
              <div style="font-size:1.5rem;">${pInfo.icon}</div>
              <div class="matrix-card-title">${pInfo.name}</div>
              <div class="matrix-card-union">${pInfo.union}</div>
              <div class="matrix-card-count">${count.toLocaleString()} samples</div>
            `;
            grid.appendChild(card);
          });
        }

        renderArchetypesGuide() {
          const container = document.getElementById("archetypesGuideContainer");
          container.innerHTML = "";

          Object.keys(ARCHETYPE_INFO).forEach(archKey => {
            const info = ARCHETYPE_INFO[archKey];
            const count = this.stats.archetypes?.[archKey] || this.allRecords.filter(r => r.archetype === archKey).length;
            const card = document.createElement("div");
            card.className = "turn-card";
            card.style.borderLeft = `4px solid ${info.color}`;
            card.style.cursor = "pointer";
            card.onclick = () => {
              this.closeModal("modalArchetypes");
              document.getElementById("archetypeFilter").value = archKey;
              this.handleFilterChange();
            };
            card.innerHTML = `
              <div class="turn-header">
                <div class="turn-title" style="color:${info.color};">${info.title}</div>
                <span class="badge" style="background:rgba(255,255,255,0.08);">${count.toLocaleString()} samples</span>
              </div>
              <p style="font-size:0.85rem; color:var(--text-secondary);">${info.desc}</p>
            `;
            container.appendChild(card);
          });
        }

        renderAnalytics() {
          const grid = document.getElementById("analyticsMetricGrid");
          const totalRecords = this.stats.total_records || this.allRecords.length;
          const totalTokens = this.stats.total_tokens_est || this.allRecords.reduce((acc, r) => acc + (r.approx_tokens || 0), 0);
          const avgTokens = totalRecords > 0 ? Math.round(totalTokens / totalRecords) : 0;
          const uniquePersonas = Object.keys(this.stats.personas || {}).length || 35;
          const uniqueEntities = new Set(this.allRecords.map(r => r.entity_uri)).size;

          grid.innerHTML = `
            <div class="stat-metric-card">
              <span class="stat-metric-label">Total Corpus Records</span>
              <span class="stat-metric-val" style="color:var(--accent-cyan);">${totalRecords.toLocaleString()}</span>
            </div>
            <div class="stat-metric-card">
              <span class="stat-metric-label">Estimated Total Tokens</span>
              <span class="stat-metric-val" style="color:var(--accent-purple);">${totalTokens.toLocaleString()}</span>
            </div>
            <div class="stat-metric-card">
              <span class="stat-metric-label">Avg Tokens / Record</span>
              <span class="stat-metric-val">${avgTokens}</span>
            </div>
            <div class="stat-metric-card">
              <span class="stat-metric-label">Scientific Personas</span>
              <span class="stat-metric-val" style="color:var(--accent-emerald);">${uniquePersonas}</span>
            </div>
            <div class="stat-metric-card">
              <span class="stat-metric-label">Entities Covered</span>
              <span class="stat-metric-val">${uniqueEntities.toLocaleString()}</span>
            </div>
          `;

          // Split bars
          const splitContainer = document.getElementById("splitDistBars");
          splitContainer.innerHTML = "";
          const splits = ["train", "val", "test", "dpo"];
          splits.forEach(sp => {
            const count = this.stats.splits?.[sp] || this.allRecords.filter(r => r.split === sp).length;
            const pct = totalRecords > 0 ? ((count / totalRecords) * 100).toFixed(1) : 0;
            const color = sp === "train" ? "var(--split-train)" : sp === "val" ? "var(--split-val)" : sp === "test" ? "var(--split-test)" : "var(--split-dpo)";
            splitContainer.appendChild(this.createDistRow(sp.toUpperCase(), count, pct, color));
          });

          // Archetype bars
          const archContainer = document.getElementById("archetypeDistBars");
          archContainer.innerHTML = "";
          Object.keys(ARCHETYPE_INFO).forEach(archKey => {
            const count = this.stats.archetypes?.[archKey] || this.allRecords.filter(r => r.archetype === archKey).length;
            const pct = totalRecords > 0 ? ((count / totalRecords) * 100).toFixed(1) : 0;
            archContainer.appendChild(this.createDistRow(archKey.replace(/_/g, " "), count, pct, ARCHETYPE_INFO[archKey].color));
          });

          // Persona bars
          const personaContainer = document.getElementById("personaDistBars");
          personaContainer.innerHTML = "";
          const pEntries = Object.entries(this.stats.personas || {}).sort((a, b) => b[1] - a[1]).slice(0, 10);
          pEntries.forEach(([pKey, count]) => {
            const pInfo = PERSONA_LABELS[pKey] || { name: pKey };
            const pct = totalRecords > 0 ? ((count / totalRecords) * 100).toFixed(1) : 0;
            personaContainer.appendChild(this.createDistRow(`${pInfo.icon || "•"} ${pInfo.name}`, count, pct, "var(--accent-cyan)"));
          });
        }

        createDistRow(label, count, pct, color) {
          const row = document.createElement("div");
          row.className = "dist-row";
          row.innerHTML = `
            <div class="dist-header">
              <span>${label}</span>
              <span><strong>${count.toLocaleString()}</strong> (${pct}%)</span>
            </div>
            <div class="dist-track">
              <div class="dist-fill" style="width:${pct}%; background:${color};"></div>
            </div>
          `;
          return row;
        }

        renderFileInventory() {
          const tableContainer = document.getElementById("fileInventoryTable");
          const inv = this.stats.file_inventory || [];
          if (inv.length === 0) {
            tableContainer.innerHTML = `<div style="color:var(--text-muted); font-size:0.82rem;">Standard partitions: train.jsonl, val.jsonl, test.jsonl, dpo_preferences.jsonl, by_persona/, by_archetype/</div>`;
            return;
          }

          let html = `<table style="width:100%; border-collapse:collapse; font-size:0.8rem; font-family:var(--font-mono);">
            <thead>
              <tr style="border-bottom:1px solid var(--border-color); text-align:left; color:var(--text-muted);">
                <th style="padding:0.4rem;">File Name</th>
                <th style="padding:0.4rem;">Split</th>
                <th style="padding:0.4rem; text-align:right;">Records</th>
                <th style="padding:0.4rem; text-align:right;">Size</th>
              </tr>
            </thead>
            <tbody>`;
          inv.forEach(f => {
            const sizeStr = f.bytes > 1048576 ? `${(f.bytes / 1048576).toFixed(1)} MB` : `${Math.round(f.bytes / 1024)} KB`;
            html += `<tr style="border-bottom:1px solid var(--border-subtle);">
              <td style="padding:0.4rem; color:var(--text-link);">${f.path}</td>
              <td style="padding:0.4rem;"><span class="badge badge-${f.split}">${f.split}</span></td>
              <td style="padding:0.4rem; text-align:right;">${(f.count || 0).toLocaleString()}</td>
              <td style="padding:0.4rem; text-align:right; color:var(--text-muted);">${sizeStr}</td>
            </tr>`;
          });
          html += `</tbody></table>`;
          tableContainer.innerHTML = html;
        }

        handleCustomFileUpload(event) {
          const file = event.target.files[0];
          if (!file) return;

          const reader = new FileReader();
          reader.onload = (e) => {
            const content = e.target.result;
            const lines = content.split("\\n");
            const parsed = [];
            lines.forEach((line, idx) => {
              line = line.strip ? line.strip() : line.trim();
              if (line) {
                try {
                  const raw = JSON.parse(line);
                  parsed.push(normalizeRecord_JS(raw, file.name));
                } catch(err) {}
              }
            });

            if (parsed.length > 0) {
              this.allRecords = parsed;
              this.stats = { total_records: parsed.length, splits: {} };
              this.updateSplitCounters();
              this.applyFilters();
              this.closeModal("modalFiles");
              this.showToast(`Loaded ${parsed.length.toLocaleString()} records from ${file.name}`);
            } else {
              this.showToast("Failed to parse JSONL records from file.");
            }
          };
          reader.readAsText(file);
        }

        exportFilteredJSONL() {
          if (this.filteredRecords.length === 0) {
            this.showToast("No records to export.");
            return;
          }
          const lines = this.filteredRecords.map(r => {
            const out = {
              id: r.id,
              archetype: r.archetype,
              entity_uri: r.entity_uri,
              messages: [
                { role: "system", content: r.system_prompt },
                { role: "user", content: r.user_query },
                { role: "assistant", content: r.ground_truth_answer }
              ],
              metadata: { persona: r.persona, split: r.split, category: r.category }
            };
            return JSON.stringify(out);
          });
          const blob = new Blob([lines.join("\\n")], { type: "application/jsonl" });
          const url = URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = url;
          a.download = `drum_ml_export_${this.currentSplit}_${this.currentPersona}.jsonl`;
          a.click();
          URL.revokeObjectURL(url);
          this.showToast(`Exported ${this.filteredRecords.length.toLocaleString()} records to JSONL`);
        }

        copyCurrentRecordJSON() {
          const rec = this.filteredRecords[this.currentIndex];
          if (!rec) return;
          this.copyText(JSON.stringify(rec, null, 2), "Copied record JSON to clipboard!");
        }

        toggleBookmark() {
          const rec = this.filteredRecords[this.currentIndex];
          if (!rec) return;
          if (this.bookmarkedIds.has(rec.id)) {
            this.bookmarkedIds.delete(rec.id);
            this.showToast("Removed bookmark");
          } else {
            this.bookmarkedIds.add(rec.id);
            this.showToast("Bookmarked sample!");
          }
          localStorage.setItem("drum_ml_bookmarks", JSON.stringify(Array.from(this.bookmarkedIds)));
          this.renderSampleList();
          this.renderInspector();
        }

        setupKeyboardShortcuts() {
          window.addEventListener("keydown", (e) => {
            if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" || e.target.tagName === "SELECT") {
              if (e.key === "Escape") e.target.blur();
              return;
            }

            if (e.key === "j" || e.key === "ArrowDown") {
              e.preventDefault();
              if (this.currentIndex < this.filteredRecords.length - 1) {
                this.currentIndex++;
                this.renderSampleList();
                this.renderInspector();
                this.scrollActiveCardIntoView();
              }
            } else if (e.key === "k" || e.key === "ArrowUp") {
              e.preventDefault();
              if (this.currentIndex > 0) {
                this.currentIndex--;
                this.renderSampleList();
                this.renderInspector();
                this.scrollActiveCardIntoView();
              }
            } else if (e.key === "/") {
              e.preventDefault();
              document.getElementById("searchInput")?.focus();
            } else if (e.key === "r") {
              this.pickRandomSample();
            } else if (e.key === "t") {
              this.toggleTheme();
            } else if (e.key === "1") {
              this.setInspectorFormat("rendered");
            } else if (e.key === "2") {
              this.setInspectorFormat("openai");
            } else if (e.key === "3") {
              this.setInspectorFormat("sharegpt");
            } else if (e.key === "4") {
              this.setInspectorFormat("raw");
            } else if (e.key === "Escape") {
              document.querySelectorAll(".modal-backdrop").forEach(m => m.classList.remove("active"));
            } else if (e.key === "?") {
              this.openModal("modalShortcuts");
            }
          });
        }

        toggleTheme() {
          const current = document.documentElement.getAttribute("data-theme") || "dark";
          const next = current === "dark" ? "light" : "dark";
          document.documentElement.setAttribute("data-theme", next);
          localStorage.setItem("drum_ml_theme", next);
          this.showToast(`Switched to ${next} theme`);
        }

        loadTheme() {
          const saved = localStorage.getItem("drum_ml_theme");
          if (saved) {
            document.documentElement.setAttribute("data-theme", saved);
          }
        }

        openModal(id) {
          document.getElementById(id)?.classList.add("active");
        }

        closeModal(id) {
          document.getElementById(id)?.classList.remove("active");
        }

        showToast(msg) {
          const container = document.getElementById("toastContainer");
          const toast = document.createElement("div");
          toast.className = "toast";
          toast.textContent = msg;
          container.appendChild(toast);
          setTimeout(() => toast.remove(), 2500);
        }

        copyText(text, successMsg = "Copied to clipboard!") {
          navigator.clipboard.writeText(text).then(() => {
            this.showToast(successMsg);
          }).catch(() => {
            this.showToast("Failed to copy");
          });
        }

        highlightMatch(text) {
          if (!this.searchQuery || !text) return this.escapeHtml(text);
          const escaped = this.escapeHtml(text);
          const q = this.escapeRegex(this.searchQuery);
          return escaped.replace(new RegExp(`(${q})`, "gi"), "<mark class='highlight'>$1</mark>");
        }

        formatMarkdown(md) {
          if (!md) return "";
          let html = this.escapeHtml(md);

          // Bold: **text**
          html = html.replace(/\\*\\*(.*?)\\*\\*/g, "<strong>$1</strong>");

          // Inline Code: `code`
          html = html.replace(/`([^`]+)`/g, "<code style='background:var(--bg-subtle); padding:0.1rem 0.3rem; border-radius:3px; font-family:var(--font-mono); font-size:0.85em;'>$1</code>");

          // Bullet points
          html = html.replace(/^- (.*)$/gm, "<li>$1</li>");
          html = html.replace(/(<li>.*<\\/li>)/s, "<ul>$1</ul>");

          // Paragraph breaks
          html = html.split("\\n\\n").map(p => p.startsWith("<ul>") ? p : `<p>${p}</p>`).join("");
          return html;
        }

        escapeHtml(str) {
          if (!str) return "";
          return String(str)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
        }

        escapeQuotes(str) {
          if (!str) return "";
          return String(str).replace(/'/g, "\\\\'").replace(/"/g, '\\\\"').replace(/\\n/g, "\\\\n");
        }

        escapeRegex(str) {
          return str.replace(/[.*+?^${}()|[\\]\\\\]/g, "\\\\$&");
        }
      }

      function normalizeRecord_JS(item, fileName) {
        return {
          id: item.id || `custom_${Math.random()}`,
          split: item.split || (fileName.includes("test") ? "test" : fileName.includes("val") ? "val" : fileName.includes("dpo") ? "dpo" : "train"),
          persona: item.persona || (item.metadata && item.metadata.persona) || "general_user",
          archetype: item.archetype || "direct_identification",
          category: item.category || "units",
          entity_uri: item.entity_uri || "",
          system_prompt: item.messages?.[0]?.content || "You are an authoritative SI and metrology assistant.",
          user_query: item.messages?.[1]?.content || item.user_query || item.prompt || "",
          ground_truth_answer: item.messages?.[2]?.content || item.ground_truth_answer || item.chosen || "",
          rejected: item.rejected || "",
          rejection_reason: item.rejection_reason || "",
          is_dpo: Boolean(item.rejected),
          approx_tokens: Math.max(1, Math.round(((item.user_query || "").length + (item.ground_truth_answer || "").length) / 3.8)),
          metadata: item.metadata || {}
        };
      }

      const app = new DatasetViewerApp();
      window.app = app;

      if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", () => app.init());
      } else {
        app.init();
      }

    })();
  </script>
</body>
</html>
"""


def generate_dataset_html(
    dataset_records: list[dict[str, Any]],
    dataset_stats: dict[str, Any] | None = None,
    dataset_title: str = "DRUM-ML Metrology Instruct Dataset",
) -> str:
    """Generates the complete self-contained HTML page for exploring training/test dataset splits."""
    data_json_str = json.dumps(dataset_records)
    stats_json_str = json.dumps(dataset_stats) if dataset_stats else "null"

    html = DATASET_VIEWER_HTML_TEMPLATE.replace("__DATASET_TITLE__", dataset_title)
    html = html.replace("__DATA_JSON__", data_json_str)
    html = html.replace("__STATS_JSON__", stats_json_str)
    return html


def save_dataset_viewer(
    output_html_path: str | Path,
    dataset_dir_or_file: str | Path = "./dataset",
    max_samples: int | None = 5000,
    open_browser: bool = False,
) -> Path:
    """Compiles and saves the standalone interactive dataset viewer HTML file."""
    records, stats = load_dataset_records(dataset_dir_or_file, max_samples=max_samples)

    html = generate_dataset_html(
        dataset_records=records,
        dataset_stats=stats,
        dataset_title="DRUM-ML Metrology Instruct Dataset",
    )

    out_p = Path(output_html_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        f.write(html)

    if open_browser:
        webbrowser.open(f"file://{out_p.resolve()}")

    return out_p


if __name__ == "__main__":
    import sys

    out_file = sys.argv[1] if len(sys.argv) > 1 else "dataset/dataset_viewer.html"
    src_dir = sys.argv[2] if len(sys.argv) > 2 else "dataset"
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
    p = save_dataset_viewer(out_file, src_dir, max_samples=limit)
    print(f"Generated dataset viewer HTML at: {p.resolve()}")
