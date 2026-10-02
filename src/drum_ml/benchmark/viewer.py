"""DRUM Metrology Benchmark (M-Eval) Interactive HTML Viewer Generator.

Generates a standalone, feature-rich, interactive HTML dashboard for browsing,
reviewing, testing, and auditing the DRUM Metrology Benchmark dataset.
"""

from __future__ import annotations

import json
import webbrowser
from pathlib import Path
from typing import Any


def load_jsonl_records(file_path: str | Path) -> list[dict[str, Any]]:
    """Load benchmark records from a JSONL file."""
    records = []
    p = Path(file_path)
    if not p.exists():
        return records
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def load_json_file(file_path: str | Path) -> dict[str, Any] | None:
    """Load a JSON file if it exists."""
    p = Path(file_path)
    if not p.exists():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>__DATASET_TITLE__ - Interactive Explorer & Review Tool</title>

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

      /* Dark Theme */
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

      /* Task Specific Colors */
      --task-constants: #a855f7;
      --task-dimensions: #0ea5e9;
      --task-conversions: #10b981;
      --task-homogeneity: #f59e0b;
      --task-conventions: #ec4899;
      --task-uncertainty: #6366f1;

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
      --text-muted: #94a3b8;
      --text-link: #0284c7;

      --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
      --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.08), 0 2px 4px -1px rgba(0, 0, 0, 0.04);
      --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      font-family: var(--font-sans);
      background-color: var(--bg-main);
      color: var(--text-primary);
      line-height: 1.5;
      -webkit-font-smoothing: antialiased;
      overflow-x: hidden;
      display: flex;
      flex-direction: column;
      height: 100vh;
    }

    /* Top Navigation Bar */
    header.app-header {
      background: var(--bg-sidebar);
      border-bottom: 1px solid var(--border-color);
      padding: 0.75rem 1.5rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
      z-index: 100;
      flex-shrink: 0;
    }

    .logo-container {
      display: flex;
      align-items: center;
      gap: 0.85rem;
      text-decoration: none;
      color: inherit;
    }

    .brand-icon {
      width: 36px;
      height: 36px;
      background: linear-gradient(135deg, #0ea5e9, #6366f1, #a855f7);
      border-radius: var(--radius-md);
      display: flex;
      align-items: center;
      justify-content: center;
      color: white;
      font-weight: 800;
      font-size: 1.1rem;
      box-shadow: 0 0 15px rgba(99, 102, 241, 0.4);
    }

    .brand-text h1 {
      font-family: var(--font-display);
      font-size: 1.15rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      background: linear-gradient(to right, #38bdf8, #818cf8, #c084fc);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .brand-text p {
      font-size: 0.75rem;
      color: var(--text-secondary);
      font-weight: 500;
    }

    .nav-tabs {
      display: flex;
      align-items: center;
      background: var(--bg-subtle);
      padding: 0.25rem;
      border-radius: var(--radius-md);
      border: 1px solid var(--border-color);
      gap: 0.25rem;
    }

    .nav-tab-btn {
      background: transparent;
      border: none;
      padding: 0.45rem 0.9rem;
      border-radius: var(--radius-sm);
      color: var(--text-secondary);
      font-size: 0.85rem;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 0.4rem;
      transition: all 0.15s ease;
    }

    .nav-tab-btn:hover {
      color: var(--text-primary);
      background: rgba(255, 255, 255, 0.05);
    }

    .nav-tab-btn.active {
      background: var(--bg-card);
      color: var(--text-primary);
      box-shadow: var(--shadow-sm);
    }

    .header-actions {
      display: flex;
      align-items: center;
      gap: 0.6rem;
    }

    .btn {
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      padding: 0.45rem 0.85rem;
      font-size: 0.825rem;
      font-weight: 600;
      border-radius: var(--radius-sm);
      border: 1px solid var(--border-color);
      background: var(--bg-card);
      color: var(--text-primary);
      cursor: pointer;
      transition: all 0.15s ease;
      white-space: nowrap;
    }

    .btn:hover {
      background: var(--bg-card-hover);
      border-color: var(--border-focus);
    }

    .btn-primary {
      background: linear-gradient(135deg, #0284c7, #2563eb);
      border-color: #38bdf8;
      color: white;
    }

    .btn-primary:hover {
      background: linear-gradient(135deg, #0369a1, #1d4ed8);
      box-shadow: 0 0 12px rgba(14, 165, 233, 0.4);
    }

    .btn-success {
      background: rgba(16, 185, 129, 0.15);
      border-color: #10b981;
      color: #34d399;
    }

    .btn-success:hover {
      background: rgba(16, 185, 129, 0.25);
    }

    .btn-warning {
      background: rgba(245, 158, 11, 0.15);
      border-color: #f59e0b;
      color: #fbbf24;
    }

    .btn-warning:hover {
      background: rgba(245, 158, 11, 0.25);
    }

    .icon-btn {
      padding: 0.45rem;
      width: 34px;
      height: 34px;
      display: flex;
      align-items: center;
      justify-content: center;
      border-radius: var(--radius-sm);
      background: var(--bg-subtle);
      border: 1px solid var(--border-color);
      color: var(--text-secondary);
      cursor: pointer;
      transition: all 0.15s ease;
    }

    .icon-btn:hover {
      color: var(--text-primary);
      border-color: var(--border-focus);
    }

    /* Main App Layout */
    .app-body {
      display: flex;
      flex: 1;
      overflow: hidden;
      position: relative;
    }

    /* View Containers */
    .view-panel {
      display: none;
      width: 100%;
      height: 100%;
      overflow: hidden;
    }

    .view-panel.active {
      display: flex;
    }

    /* Explorer Split View */
    .explorer-sidebar {
      width: 410px;
      min-width: 340px;
      max-width: 520px;
      background: var(--bg-sidebar);
      border-right: 1px solid var(--border-color);
      display: flex;
      flex-direction: column;
      flex-shrink: 0;
    }

    .sidebar-header {
      padding: 1rem;
      border-bottom: 1px solid var(--border-color);
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }

    .search-box {
      position: relative;
      width: 100%;
    }

    .search-input {
      width: 100%;
      background: var(--bg-input);
      border: 1px solid var(--border-color);
      color: var(--text-primary);
      padding: 0.55rem 2rem 0.55rem 2.2rem;
      border-radius: var(--radius-sm);
      font-size: 0.85rem;
      outline: none;
      transition: border-color 0.15s ease;
    }

    .search-input:focus {
      border-color: var(--border-focus);
      box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.15);
    }

    .search-icon {
      position: absolute;
      left: 0.75rem;
      top: 50%;
      transform: translateY(-50%);
      color: var(--text-muted);
      pointer-events: none;
      font-size: 0.85rem;
    }

    .search-clear {
      position: absolute;
      right: 0.6rem;
      top: 50%;
      transform: translateY(-50%);
      background: none;
      border: none;
      color: var(--text-muted);
      cursor: pointer;
      padding: 0.2rem;
      display: none;
    }

    .filter-group {
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }

    .filter-chips-scroll {
      display: flex;
      flex-wrap: wrap;
      gap: 0.35rem;
    }

    .filter-chip {
      background: var(--bg-subtle);
      border: 1px solid var(--border-color);
      padding: 0.25rem 0.6rem;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 500;
      color: var(--text-secondary);
      cursor: pointer;
      user-select: none;
      transition: all 0.12s ease;
      display: flex;
      align-items: center;
      gap: 0.3rem;
    }

    .filter-chip:hover {
      color: var(--text-primary);
      border-color: var(--text-muted);
    }

    .filter-chip.active {
      background: var(--accent-blue);
      border-color: #60a5fa;
      color: white;
      font-weight: 600;
    }

    .filter-chip[data-task="constants"].active { background: var(--task-constants); border-color: #c084fc; }
    .filter-chip[data-task="dimensions"].active { background: var(--task-dimensions); border-color: #38bdf8; }
    .filter-chip[data-task="conversions"].active { background: var(--task-conversions); border-color: #34d399; }
    .filter-chip[data-task="homogeneity"].active { background: var(--task-homogeneity); border-color: #fbbf24; }
    .filter-chip[data-task="conventions"].active { background: var(--task-conventions); border-color: #f472b6; }
    .filter-chip[data-task="uncertainty"].active { background: var(--task-uncertainty); border-color: #818cf8; }

    .filter-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.5rem;
      font-size: 0.8rem;
    }

    .select-sm {
      background: var(--bg-input);
      border: 1px solid var(--border-color);
      color: var(--text-primary);
      padding: 0.3rem 0.5rem;
      border-radius: var(--radius-sm);
      font-size: 0.775rem;
      outline: none;
    }

    .list-meta-bar {
      padding: 0.45rem 1rem;
      background: rgba(0, 0, 0, 0.15);
      border-bottom: 1px solid var(--border-color);
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.75rem;
      color: var(--text-muted);
    }

    /* Sidebar Sample List */
    .sample-list {
      flex: 1;
      overflow-y: auto;
      display: block;
    }

    .sample-item {
      padding: 0.85rem 1rem;
      border-bottom: 1px solid var(--border-subtle);
      cursor: pointer;
      transition: background 0.12s ease;
      display: flex;
      flex-direction: column;
      gap: 0.45rem;
      position: relative;
      flex-shrink: 0;
      box-sizing: border-box;
      width: 100%;
    }

    .sample-item:hover {
      background: var(--bg-card-hover);
    }

    .sample-item.active {
      background: rgba(56, 189, 248, 0.08);
      border-left: 3px solid var(--accent-cyan);
    }

    .sample-item-top {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.5rem;
      width: 100%;
    }

    .sample-item-header-left {
      display: flex;
      align-items: center;
      gap: 0.4rem;
      min-width: 0;
      flex: 1;
    }

    .sample-item-id {
      font-family: var(--font-mono);
      font-size: 0.725rem;
      color: var(--text-secondary);
      font-weight: 600;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .sample-tags {
      display: flex;
      align-items: center;
      gap: 0.35rem;
      flex-shrink: 0;
      white-space: nowrap;
    }

    .sample-item-text {
      font-size: 0.825rem;
      color: var(--text-primary);
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
      line-height: 1.4;
      word-break: break-word;
    }

    .sample-item-footer {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.4rem;
      font-size: 0.7rem;
      color: var(--text-muted);
      margin-top: 0.1rem;
    }

    .sample-item-footer-left {
      display: flex;
      align-items: center;
      gap: 0.3rem;
      white-space: nowrap;
      flex-shrink: 0;
    }

    .sample-item-full-id {
      font-family: var(--font-mono);
      font-size: 0.675rem;
      color: var(--text-muted);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      text-align: right;
      flex: 1;
      min-width: 0;
    }

    .tag {
      display: inline-flex;
      align-items: center;
      gap: 0.2rem;
      padding: 0.15rem 0.45rem;
      border-radius: 4px;
      font-size: 0.68rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.03em;
      white-space: nowrap;
      flex-shrink: 0;
      line-height: 1.2;
    }

    .tag-task {
      background: rgba(255, 255, 255, 0.08);
      color: var(--text-secondary);
    }

    .tag-task[data-task="constants"] { background: rgba(168, 85, 247, 0.18); color: #d8b4fe; }
    .tag-task[data-task="dimensions"] { background: rgba(14, 165, 233, 0.18); color: #7dd3fc; }
    .tag-task[data-task="conversions"] { background: rgba(16, 185, 129, 0.18); color: #6ee7b7; }
    .tag-task[data-task="homogeneity"] { background: rgba(245, 158, 11, 0.18); color: #fde68a; }
    .tag-task[data-task="conventions"] { background: rgba(236, 72, 153, 0.18); color: #fbcfe8; }
    .tag-task[data-task="uncertainty"] { background: rgba(99, 102, 241, 0.18); color: #c7d2fe; }

    .tag-diff-introductory { background: rgba(16, 185, 129, 0.12); color: #34d399; }
    .tag-diff-intermediate { background: rgba(14, 165, 233, 0.12); color: #38bdf8; }
    .tag-diff-advanced { background: rgba(245, 158, 11, 0.12); color: #fbbf24; }

    .tag-format {
      background: var(--bg-subtle);
      color: var(--text-muted);
    }

    .tag-eval-pass {
      background: rgba(16, 185, 129, 0.2) !important;
      color: #34d399 !important;
      border: 1px solid rgba(16, 185, 129, 0.4) !important;
    }

    .tag-eval-fail {
      background: rgba(239, 68, 68, 0.2) !important;
      color: #f87171 !important;
      border: 1px solid rgba(239, 68, 68, 0.4) !important;
    }

    /* Scorecard sidebar banner & quick filter pills */
    .scorecard-sidebar-bar {
      background: rgba(30, 41, 59, 0.6);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-md);
      padding: 0.6rem 0.75rem;
      display: flex;
      flex-direction: column;
      gap: 0.45rem;
    }

    .scorecard-sidebar-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 0.75rem;
      color: var(--text-secondary);
    }

    .eval-filter-pills {
      display: flex;
      gap: 0.3rem;
    }

    .eval-filter-pill {
      flex: 1;
      text-align: center;
      padding: 0.25rem 0.4rem;
      border-radius: var(--radius-sm);
      font-size: 0.725rem;
      font-weight: 600;
      cursor: pointer;
      border: 1px solid var(--border-color);
      background: var(--bg-input);
      color: var(--text-secondary);
      transition: all 0.15s ease;
      white-space: nowrap;
    }

    .eval-filter-pill:hover {
      color: var(--text-primary);
      border-color: var(--border-focus);
    }

    .eval-filter-pill.active {
      background: var(--accent-blue);
      border-color: #60a5fa;
      color: white;
    }

    .eval-filter-pill.eval-pill-pass.active {
      background: rgba(16, 185, 129, 0.25);
      border-color: #10b981;
      color: #34d399;
    }

    .eval-filter-pill.eval-pill-fail.active {
      background: rgba(239, 68, 68, 0.25);
      border-color: #ef4444;
      color: #f87171;
    }

    /* Model Evaluation Result Banner */
    .eval-banner {
      border-radius: var(--radius-md);
      padding: 1rem 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 0.6rem;
      box-shadow: var(--shadow-sm);
      animation: fadeIn 0.2s ease;
    }

    .eval-banner-pass {
      background: linear-gradient(135deg, rgba(16, 185, 129, 0.12), rgba(6, 182, 212, 0.06));
      border: 1px solid rgba(16, 185, 129, 0.4);
    }

    .eval-banner-fail {
      background: linear-gradient(135deg, rgba(239, 68, 68, 0.12), rgba(245, 158, 11, 0.06));
      border: 1px solid rgba(239, 68, 68, 0.4);
    }

    .eval-banner-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.75rem;
      flex-wrap: wrap;
    }

    .eval-banner-badge {
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      font-size: 0.8rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }

    .eval-banner-pass .eval-banner-badge {
      color: #34d399;
    }

    .eval-banner-fail .eval-banner-badge {
      color: #f87171;
    }

    .eval-banner-model {
      font-family: var(--font-mono);
      font-size: 0.775rem;
      color: var(--text-secondary);
      background: var(--bg-input);
      padding: 0.2rem 0.5rem;
      border-radius: 4px;
      border: 1px solid var(--border-color);
    }

    .eval-banner-body {
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
      font-size: 0.875rem;
      color: var(--text-primary);
    }

    .eval-pred-key {
      display: inline-block;
      font-family: var(--font-mono);
      font-weight: 700;
      padding: 0.1rem 0.45rem;
      border-radius: 4px;
      background: rgba(239, 68, 68, 0.2);
      color: #f87171;
      border: 1px solid rgba(239, 68, 68, 0.4);
    }

    .eval-exp-key {
      display: inline-block;
      font-family: var(--font-mono);
      font-weight: 700;
      padding: 0.1rem 0.45rem;
      border-radius: 4px;
      background: rgba(16, 185, 129, 0.2);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.4);
    }

    .eval-error-msg {
      font-size: 0.8rem;
      color: var(--accent-amber);
      background: rgba(245, 158, 11, 0.08);
      border-left: 3px solid var(--accent-amber);
      padding: 0.4rem 0.6rem;
      border-radius: 0 4px 4px 0;
    }

    .eval-raw-toggle {
      margin-top: 0.25rem;
      font-size: 0.775rem;
    }

    .eval-raw-toggle summary {
      cursor: pointer;
      color: var(--text-link);
      user-select: none;
    }

    .eval-raw-toggle pre {
      margin-top: 0.4rem;
      background: var(--bg-input);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-sm);
      padding: 0.6rem;
      font-family: var(--font-mono);
      font-size: 0.75rem;
      white-space: pre-wrap;
      word-break: break-all;
      max-height: 180px;
      overflow-y: auto;
      color: var(--text-secondary);
    }

    .option-card.model-selected-correct {
      border: 2px solid #10b981 !important;
      box-shadow: 0 0 10px rgba(16, 185, 129, 0.2);
    }

    .option-card.model-selected-wrong {
      border: 2px solid #ef4444 !important;
      box-shadow: 0 0 10px rgba(239, 68, 68, 0.2);
    }

    .badge-model-pred {
      font-size: 0.7rem;
      font-weight: 700;
      text-transform: uppercase;
      padding: 0.2rem 0.5rem;
      border-radius: 9999px;
      letter-spacing: 0.03em;
    }

    .badge-model-pred.pass {
      background: rgba(16, 185, 129, 0.25);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.5);
    }

    .badge-model-pred.fail {
      background: rgba(239, 68, 68, 0.25);
      color: #f87171;
      border: 1px solid rgba(239, 68, 68, 0.5);
    }

    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      display: inline-block;
    }
    .status-approved { background: #10b981; box-shadow: 0 0 6px rgba(16, 185, 129, 0.6); }
    .status-flagged { background: #f59e0b; box-shadow: 0 0 6px rgba(245, 158, 11, 0.6); }
    .status-pending { background: #475569; }

    .sample-item-text {
      font-size: 0.825rem;
      color: var(--text-primary);
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
      line-height: 1.35;
    }

    /* Main Inspector Panel */
    .explorer-content {
      flex: 1;
      display: flex;
      flex-direction: column;
      overflow-y: auto;
      background: var(--bg-main);
      padding: 1.5rem 2rem;
      gap: 1.5rem;
    }

    .inspector-header {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-lg);
      padding: 1.25rem 1.5rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
      flex-wrap: wrap;
      box-shadow: var(--shadow-sm);
    }

    .inspector-title-area {
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
    }

    .inspector-id-row {
      display: flex;
      align-items: center;
      gap: 0.6rem;
    }

    .inspector-id {
      font-family: var(--font-mono);
      font-size: 1.1rem;
      font-weight: 700;
      color: var(--accent-cyan);
    }

    .inspector-badges {
      display: flex;
      align-items: center;
      gap: 0.4rem;
      flex-wrap: wrap;
    }

    .inspector-actions {
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    /* Question & Content Cards */
    .card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-lg);
      padding: 1.5rem;
      box-shadow: var(--shadow-sm);
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }

    .card-title {
      font-family: var(--font-display);
      font-size: 0.95rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-secondary);
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .question-box {
      font-size: 1.125rem;
      font-weight: 500;
      color: var(--text-primary);
      line-height: 1.6;
    }

    .context-box {
      background: var(--bg-subtle);
      border-left: 3px solid var(--accent-blue);
      padding: 0.75rem 1rem;
      border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
      font-size: 0.875rem;
      color: var(--text-secondary);
    }

    /* Options / Distractor Cards */
    .options-grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 0.75rem;
    }

    .option-card {
      background: var(--bg-input);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-md);
      padding: 1rem 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
      transition: all 0.15s ease;
      position: relative;
    }

    .option-card.correct {
      background: rgba(16, 185, 129, 0.08);
      border-color: #10b981;
    }

    .option-card.distractor {
      background: rgba(239, 68, 68, 0.03);
      border-color: var(--border-subtle);
    }

    .option-main {
      display: flex;
      align-items: flex-start;
      gap: 0.85rem;
    }

    .option-key-badge {
      width: 28px;
      height: 28px;
      border-radius: 6px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-family: var(--font-mono);
      font-weight: 700;
      font-size: 0.85rem;
      flex-shrink: 0;
      background: var(--bg-subtle);
      border: 1px solid var(--border-color);
      color: var(--text-secondary);
    }

    .option-card.correct .option-key-badge {
      background: #10b981;
      border-color: #34d399;
      color: white;
    }

    .option-text {
      font-size: 0.95rem;
      color: var(--text-primary);
      flex: 1;
      line-height: 1.5;
    }

    .option-status-badge {
      font-size: 0.7rem;
      font-weight: 700;
      text-transform: uppercase;
      padding: 0.2rem 0.5rem;
      border-radius: 9999px;
      letter-spacing: 0.03em;
    }

    .badge-correct {
      background: rgba(16, 185, 129, 0.2);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.4);
    }

    .distractor-pill {
      background: rgba(245, 158, 11, 0.08);
      border: 1px solid rgba(245, 158, 11, 0.2);
      color: #fbbf24;
      padding: 0.35rem 0.75rem;
      border-radius: var(--radius-sm);
      font-size: 0.775rem;
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }

    /* Free Form Answer Card */
    .ground-truth-box {
      background: rgba(14, 165, 233, 0.08);
      border: 1px solid rgba(14, 165, 233, 0.3);
      border-radius: var(--radius-md);
      padding: 1rem 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }

    .ground-truth-val {
      font-size: 1.1rem;
      font-weight: 600;
      color: #38bdf8;
    }

    /* Explanation Box */
    .explanation-content {
      font-size: 0.95rem;
      line-height: 1.6;
      color: var(--text-primary);
    }

    /* Metrology Anatomy Grid */
    .meta-anatomy-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 1rem;
    }

    .meta-cell {
      background: var(--bg-input);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-md);
      padding: 0.75rem 1rem;
      display: flex;
      flex-direction: column;
      gap: 0.3rem;
    }

    .meta-cell-label {
      font-size: 0.725rem;
      text-transform: uppercase;
      color: var(--text-muted);
      font-weight: 600;
    }

    .meta-cell-value {
      font-family: var(--font-mono);
      font-size: 0.85rem;
      color: var(--text-primary);
      word-break: break-all;
    }

    /* Dimension Vector Visualizer */
    .dimension-chips {
      display: flex;
      flex-wrap: wrap;
      gap: 0.4rem;
    }

    .dim-chip {
      padding: 0.2rem 0.5rem;
      border-radius: 4px;
      font-family: var(--font-mono);
      font-size: 0.75rem;
      background: var(--bg-subtle);
      color: var(--text-muted);
      border: 1px solid var(--border-color);
    }

    .dim-chip.active-dim {
      background: rgba(14, 165, 233, 0.15);
      border-color: #38bdf8;
      color: #7dd3fc;
      font-weight: 600;
    }

    /* Reviewer Workspace Area */
    .reviewer-box {
      background: var(--bg-subtle);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-lg);
      padding: 1.25rem 1.5rem;
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }

    .reviewer-textarea {
      width: 100%;
      background: var(--bg-input);
      border: 1px solid var(--border-color);
      color: var(--text-primary);
      padding: 0.75rem;
      border-radius: var(--radius-sm);
      font-family: var(--font-sans);
      font-size: 0.875rem;
      resize: vertical;
      min-height: 80px;
      outline: none;
    }

    .reviewer-textarea:focus {
      border-color: var(--border-focus);
    }

    /* Raw JSON Collapsible */
    details.raw-json {
      background: var(--bg-input);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-md);
      overflow: hidden;
    }

    details.raw-json summary {
      padding: 0.75rem 1rem;
      font-size: 0.825rem;
      font-weight: 600;
      color: var(--text-secondary);
      cursor: pointer;
      user-select: none;
      background: var(--bg-subtle);
    }

    details.raw-json pre {
      padding: 1rem;
      font-family: var(--font-mono);
      font-size: 0.78rem;
      overflow-x: auto;
      color: #7dd3fc;
    }

    /* Quiz Mode Styles */
    .quiz-container {
      max-width: 860px;
      margin: 0 auto;
      padding: 2rem 1.5rem;
      display: flex;
      flex-direction: column;
      gap: 1.5rem;
      width: 100%;
      overflow-y: auto;
    }

    .quiz-progress-bar-container {
      width: 100%;
      height: 6px;
      background: var(--bg-subtle);
      border-radius: 9999px;
      overflow: hidden;
    }

    .quiz-progress-fill {
      height: 100%;
      background: linear-gradient(to right, #06b6d4, #3b82f6);
      width: 0%;
      transition: width 0.3s ease;
    }

    .quiz-stats-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-lg);
      padding: 1rem 1.5rem;
      flex-wrap: wrap;
    }

    .quiz-stat-pill {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 0.85rem;
      color: var(--text-secondary);
    }

    .quiz-stat-pill strong {
      color: var(--text-primary);
      font-size: 1.05rem;
      font-family: var(--font-mono);
    }

    .quiz-card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-xl);
      padding: 2rem;
      box-shadow: var(--shadow-lg);
      display: flex;
      flex-direction: column;
      gap: 1.5rem;
    }

    .quiz-option-btn {
      background: var(--bg-input);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-md);
      padding: 1rem 1.25rem;
      display: flex;
      align-items: flex-start;
      gap: 1rem;
      text-align: left;
      cursor: pointer;
      color: var(--text-primary);
      font-size: 0.95rem;
      transition: all 0.15s ease;
      position: relative;
    }

    .quiz-option-btn:hover:not(:disabled) {
      background: var(--bg-card-hover);
      border-color: var(--border-focus);
      transform: translateY(-1px);
    }

    .quiz-option-btn.selected {
      background: rgba(56, 189, 248, 0.12);
      border-color: #38bdf8;
    }

    .quiz-option-btn.revealed-correct {
      background: rgba(16, 185, 129, 0.15) !important;
      border-color: #10b981 !important;
    }

    .quiz-option-btn.revealed-incorrect {
      background: rgba(239, 68, 68, 0.15) !important;
      border-color: #ef4444 !important;
    }

    .quiz-feedback-box {
      padding: 1.25rem 1.5rem;
      border-radius: var(--radius-md);
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
      animation: fadeIn 0.25s ease;
    }

    .quiz-feedback-box.correct {
      background: rgba(16, 185, 129, 0.12);
      border: 1px solid rgba(16, 185, 129, 0.3);
      color: #34d399;
    }

    .quiz-feedback-box.incorrect {
      background: rgba(239, 68, 68, 0.12);
      border: 1px solid rgba(239, 68, 68, 0.3);
      color: #f87171;
    }

    /* Analytics View */
    .analytics-container {
      max-width: 1200px;
      margin: 0 auto;
      padding: 2rem;
      display: flex;
      flex-direction: column;
      gap: 2rem;
      width: 100%;
      overflow-y: auto;
    }

    .stats-overview-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 1rem;
    }

    .stat-card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-lg);
      padding: 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
      box-shadow: var(--shadow-sm);
    }

    .stat-card-title {
      font-size: 0.8rem;
      text-transform: uppercase;
      color: var(--text-muted);
      font-weight: 600;
      letter-spacing: 0.05em;
    }

    .stat-card-value {
      font-family: var(--font-display);
      font-size: 1.85rem;
      font-weight: 700;
      color: var(--text-primary);
    }

    .stat-card-sub {
      font-size: 0.75rem;
      color: var(--text-secondary);
    }

    .analytics-grid-2col {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
      gap: 1.5rem;
    }

    .table-container {
      overflow-x: auto;
    }

    table.data-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
      text-align: left;
    }

    table.data-table th {
      padding: 0.65rem 1rem;
      background: var(--bg-subtle);
      color: var(--text-secondary);
      font-weight: 600;
      border-bottom: 1px solid var(--border-color);
    }

    table.data-table td {
      padding: 0.65rem 1rem;
      border-bottom: 1px solid var(--border-subtle);
      color: var(--text-primary);
    }

    .progress-bar-inline {
      width: 100%;
      height: 8px;
      background: var(--bg-subtle);
      border-radius: 9999px;
      overflow: hidden;
      margin-top: 0.25rem;
    }

    .progress-fill-inline {
      height: 100%;
      background: var(--accent-cyan);
      border-radius: 9999px;
    }

    /* Triage / Review Queue */
    .triage-container {
      max-width: 1100px;
      margin: 0 auto;
      padding: 2rem;
      display: flex;
      flex-direction: column;
      gap: 1.5rem;
      width: 100%;
      overflow-y: auto;
    }

    /* Modal dialog */
    .modal-overlay {
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.7);
      backdrop-filter: blur(4px);
      z-index: 1000;
      align-items: center;
      justify-content: center;
    }

    .modal-overlay.active {
      display: flex;
    }

    .modal-card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-xl);
      max-width: 550px;
      width: 90%;
      padding: 1.75rem;
      box-shadow: var(--shadow-lg);
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
    }

    .modal-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .modal-title {
      font-family: var(--font-display);
      font-size: 1.15rem;
      font-weight: 700;
    }

    .dropzone {
      border: 2px dashed var(--border-color);
      border-radius: var(--radius-md);
      padding: 2.5rem 1.5rem;
      text-align: center;
      background: var(--bg-input);
      cursor: pointer;
      transition: all 0.15s ease;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 0.5rem;
    }

    .dropzone:hover, .dropzone.dragover {
      border-color: var(--accent-cyan);
      background: rgba(14, 165, 233, 0.05);
    }

    /* Toast Notification */
    .toast {
      position: fixed;
      bottom: 2rem;
      right: 2rem;
      background: var(--bg-card);
      border: 1px solid var(--border-focus);
      color: var(--text-primary);
      padding: 0.75rem 1.25rem;
      border-radius: var(--radius-md);
      box-shadow: var(--shadow-lg);
      font-size: 0.85rem;
      font-weight: 500;
      z-index: 2000;
      display: flex;
      align-items: center;
      gap: 0.6rem;
      transform: translateY(100px);
      opacity: 0;
      transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
      pointer-events: none;
    }

    .toast.show {
      transform: translateY(0);
      opacity: 1;
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }

    /* Responsive Design */
    @media (max-width: 900px) {
      .explorer-sidebar {
        width: 100%;
        max-width: none;
        height: 45vh;
      }
      .app-body {
        flex-direction: column;
      }
      .view-panel.active {
        flex-direction: column;
      }
    }
  </style>
</head>
<body>

  <!-- Top App Header -->
  <header class="app-header">
    <div class="logo-container">
      <div class="brand-icon">M</div>
      <div class="brand-text">
        <h1>DRUM Metrology Benchmark</h1>
        <p>CODATA Digital Representation of Units of Measurement (M-Eval)</p>
      </div>
    </div>

    <!-- View Navigation Tabs -->
    <nav class="nav-tabs" role="tablist">
      <button class="nav-tab-btn active" data-view="explorer" id="tabExplorer">
        <span>🔍</span> Explorer & Review
      </button>
      <button class="nav-tab-btn" data-view="quiz" id="tabQuiz">
        <span>🎯</span> Test Yourself
      </button>
      <button class="nav-tab-btn" data-view="analytics" id="tabAnalytics">
        <span>📊</span> Analytics & Scorecards
      </button>
      <button class="nav-tab-btn" data-view="triage" id="tabTriage">
        <span>🚩</span> Review Queue (<span id="triageBadgeCount">0</span>)
      </button>
    </nav>

    <!-- Header Actions -->
    <div class="header-actions">
      <button class="btn" id="btnLoadFile" title="Load custom benchmark JSONL / JSON">
        <span>📁</span> Load Data
      </button>
      <button class="btn" id="btnExport" title="Export review annotations & data">
        <span>💾</span> Export
      </button>
      <button class="icon-btn" id="btnShortcuts" title="Keyboard Shortcuts">
        <span>⌨️</span>
      </button>
      <button class="icon-btn" id="btnThemeToggle" title="Toggle Dark/Light Mode">
        <span id="themeIcon">🌓</span>
      </button>
    </div>
  </header>

  <!-- Main Body -->
  <main class="app-body">

    <!-- VIEW 1: EXPLORER & REVIEW INSPECTOR -->
    <div class="view-panel active" id="viewExplorer">

      <!-- Left Sidebar: Filters & List -->
      <aside class="explorer-sidebar">
        <div class="sidebar-header">
          <!-- Search Box -->
          <div class="search-box">
            <span class="search-icon">🔍</span>
            <input type="text" id="searchInput" class="search-input" placeholder="Search questions, IDs, entities, math... (/ to focus)" />
            <button class="search-clear" id="searchClear" title="Clear search">✕</button>
          </div>

          <!-- Task Filter Chips -->
          <div class="filter-group">
            <div class="filter-chips-scroll" id="taskFilters">
              <button class="filter-chip active" data-task="all">All Tasks</button>
              <button class="filter-chip" data-task="constants">⚡ Constants</button>
              <button class="filter-chip" data-task="dimensions">📐 Dimensions</button>
              <button class="filter-chip" data-task="conversions">🔄 Conversions</button>
              <button class="filter-chip" data-task="homogeneity">⚠️ Homogeneity</button>
              <button class="filter-chip" data-task="conventions">📜 Conventions</button>
              <button class="filter-chip" data-task="uncertainty">🎯 Uncertainty</button>
            </div>
          </div>

          <!-- Scorecard / Model Grade Filter Bar (Shown when scorecard is loaded) -->
          <div id="scorecardSidebarBar" class="scorecard-sidebar-bar" style="display:none;">
            <div class="scorecard-sidebar-title">
              <span>🤖 <strong id="sidebarModelName">Model Evaluation</strong></span>
              <span id="sidebarScoreBadge" class="tag tag-eval-pass">0/0</span>
            </div>
            <div class="eval-filter-pills" id="evalFilterPills">
              <button class="eval-filter-pill active" data-eval="all">All (<span id="countEvalAll">0</span>)</button>
              <button class="eval-filter-pill eval-pill-pass" data-eval="passed">✅ Correct (<span id="countEvalPassed">0</span>)</button>
              <button class="eval-filter-pill eval-pill-fail" data-eval="failed">❌ Missed (<span id="countEvalFailed">0</span>)</button>
            </div>
          </div>

          <!-- Sub-filters: Difficulty, Format, Grade, Status -->
          <div class="filter-row">
            <select id="diffFilter" class="select-sm">
              <option value="all">All Difficulties</option>
              <option value="introductory">Introductory</option>
              <option value="intermediate">Intermediate</option>
              <option value="advanced">Advanced</option>
            </select>

            <select id="formatFilter" class="select-sm">
              <option value="all">All Formats</option>
              <option value="mcq">MCQ (Track A)</option>
              <option value="free_form">Free-Form (Track B)</option>
            </select>

            <select id="evalFilter" class="select-sm" style="display:none;">
              <option value="all">All Grades</option>
              <option value="passed">✅ Passed / Correct</option>
              <option value="failed">❌ Failed / Missed</option>
            </select>

            <select id="statusFilter" class="select-sm">
              <option value="all">All Status</option>
              <option value="approved">Approved</option>
              <option value="flagged">Flagged</option>
              <option value="pending">Pending Review</option>
            </select>
          </div>
        </div>

        <!-- Meta bar: Counts & Sort -->
        <div class="list-meta-bar">
          <span id="filteredCountLabel">Showing 0 samples</span>
          <div style="display:flex; align-items:center; gap:0.4rem;">
            <span>Sort:</span>
            <select id="sortSelect" class="select-sm" style="padding:0.15rem 0.3rem;">
              <option value="default">Default</option>
              <option value="id">ID</option>
              <option value="task">Task</option>
              <option value="difficulty">Difficulty</option>
              <option value="status">Review Status</option>
            </select>
          </div>
        </div>

        <!-- Sample List Items -->
        <div class="sample-list" id="sampleListContainer">
          <!-- Dynamically populated -->
        </div>
      </aside>

      <!-- Right Inspector Panel -->
      <section class="explorer-content" id="inspectorPanel">
        <!-- Sticky Sample Header -->
        <div class="inspector-header">
          <div class="inspector-title-area">
            <div class="inspector-id-row">
              <span class="status-dot status-pending" id="inspectStatusDot"></span>
              <span class="inspector-id" id="inspectId">-</span>
              <button class="btn" id="btnCopyId" style="padding:0.2rem 0.5rem; font-size:0.75rem;" title="Copy ID">📋</button>
            </div>
            <div class="inspector-badges" id="inspectBadges">
              <!-- Task, Difficulty, Format tags dynamically injected -->
            </div>
          </div>

          <div class="inspector-actions">
            <button class="btn btn-success" id="btnApprove" title="Mark sample as verified & accurate (V)">
              <span>✅</span> Approve
            </button>
            <button class="btn btn-warning" id="btnFlag" title="Flag sample for review or correction (F)">
              <span>🚩</span> Flag
            </button>
            <button class="btn" id="btnRandomSample" title="Jump to random sample">
              <span>🔀</span> Random
            </button>
          </div>
        </div>

        <!-- Model Evaluation Grade Banner -->
        <div id="inspectEvalBanner" style="display: none;"></div>

        <!-- Question Card -->
        <div class="card">
          <div class="card-title">
            <span>❓</span> Question & Prompt
          </div>
          <div class="question-box render-math" id="inspectQuestion">
            Select a benchmark sample from the sidebar to inspect its full metrological anatomy.
          </div>
          <div class="context-box render-math" id="inspectContext" style="display: none;">
            <!-- Optional Context -->
          </div>
        </div>

        <!-- Options & Answers Card -->
        <div class="card" id="inspectAnswersCard">
          <div class="card-title">
            <span>🎯</span> Options & Ground Truth Validation
          </div>
          <div id="inspectOptionsContainer">
            <!-- MCQ Options or Free-form Ground Truth injected here -->
          </div>
        </div>

        <!-- Metrological Explanation & Standards Citation -->
        <div class="card">
          <div class="card-title">
            <span>💡</span> Metrological Derivation & Standards Reference
          </div>
          <div class="explanation-content render-math" id="inspectExplanation">
            -
          </div>
        </div>

        <!-- Provenance, Dimensions & Metadata Anatomy -->
        <div class="card">
          <div class="card-title">
            <span>📐</span> Physical Dimensions & Provenance Metadata
          </div>
          <div class="meta-anatomy-grid">
            <div class="meta-cell">
              <span class="meta-cell-label">Provenance Entity URI</span>
              <span class="meta-cell-value" id="inspectEntityUri">-</span>
            </div>
            <div class="meta-cell">
              <span class="meta-cell-label">ISQ 7-Base Dimension Vector</span>
              <div class="dimension-chips" id="inspectDimVector">
                <!-- Dimension vector chips -->
              </div>
            </div>
            <div class="meta-cell">
              <span class="meta-cell-label">Standard Reference</span>
              <span class="meta-cell-value" id="inspectStandard">-</span>
            </div>
            <div class="meta-cell">
              <span class="meta-cell-label">Unit / Constant Symbol</span>
              <span class="meta-cell-value render-math" id="inspectSymbol">-</span>
            </div>
          </div>
        </div>

        <!-- Reviewer Notes & Feedback -->
        <div class="reviewer-box">
          <div class="card-title">
            <span>📝</span> Reviewer Annotation & Notes
          </div>
          <textarea id="inspectReviewNotes" class="reviewer-textarea" placeholder="Add notes, domain feedback, or justification for flags... (auto-saved to local storage)"></textarea>
          <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.75rem; color:var(--text-muted);">
            <span id="notesSaveStatus">Saved</span>
            <span>Shortcut: [J/K] Navigate samples | [V] Approve | [F] Flag</span>
          </div>
        </div>

        <!-- Raw JSON Inspector Card -->
        <div class="card" id="jsonInspectorCard">
          <div class="card-title" style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;">
            <div style="display:flex; align-items:center; gap:0.5rem;">
              <span>🔧</span>
              <span>Raw JSON Inspector</span>
            </div>
            <div style="display:flex; align-items:center; gap:0.4rem; flex-wrap:wrap;">
              <div class="eval-filter-pills" id="jsonTabPills" style="margin:0;">
                <button class="eval-filter-pill active" data-jsontab="eval">Scorecard Result JSON</button>
                <button class="eval-filter-pill" data-jsontab="bench">Benchmark Record JSON</button>
                <button class="eval-filter-pill" data-jsontab="combined">Combined JSON</button>
              </div>
              <button class="btn" id="btnCopyActiveJson" style="padding:0.25rem 0.6rem; font-size:0.75rem;" title="Copy displayed JSON">📋 Copy JSON</button>
            </div>
          </div>
          <pre id="inspectRawJson" style="background:var(--bg-input); border:1px solid var(--border-color); border-radius:var(--radius-md); padding:1rem; font-family:var(--font-mono); font-size:0.78rem; overflow-x:auto; max-height:350px; color:#7dd3fc; line-height:1.45;">{}</pre>
        </div>
      </section>
    </div>

    <!-- VIEW 2: TEST YOURSELF / QUIZ MODE -->
    <div class="view-panel" id="viewQuiz">
      <div class="quiz-container">
        <!-- Progress & Live Score Bar -->
        <div class="quiz-progress-bar-container">
          <div class="quiz-progress-fill" id="quizProgressFill"></div>
        </div>

        <div class="quiz-stats-header">
          <div class="quiz-stat-pill">
            <span>Task:</span>
            <strong id="quizTaskBadge">Constants</strong>
          </div>
          <div class="quiz-stat-pill">
            <span>Progress:</span>
            <strong><span id="quizCurrentIndex">1</span> / <span id="quizTotalQuestions">50</span></strong>
          </div>
          <div class="quiz-stat-pill">
            <span>Score:</span>
            <strong><span id="quizScoreCorrect">0</span> / <span id="quizScoreAttempted">0</span> (<span id="quizAccuracyPct">0%</span>)</strong>
          </div>
          <div class="quiz-stat-pill">
            <span>Streak:</span>
            <strong>🔥 <span id="quizStreak">0</span></strong>
          </div>
          <button class="btn" id="btnQuizRestart" title="Restart Quiz">🔄 Reset</button>
        </div>

        <!-- Quiz Question Card -->
        <div class="quiz-card" id="quizCard">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <span class="tag tag-task" id="quizTaskTag">Dimensions</span>
            <span class="tag tag-diff-intermediate" id="quizDiffTag">Intermediate</span>
          </div>

          <div class="question-box render-math" id="quizQuestionText">
            Loading quiz question...
          </div>

          <!-- Options Buttons -->
          <div class="options-grid" id="quizOptionsGrid">
            <!-- Buttons injected here -->
          </div>

          <!-- Feedback Box -->
          <div class="quiz-feedback-box" id="quizFeedbackBox" style="display: none;">
            <div id="quizFeedbackHeading" style="font-weight:700; font-size:1.05rem;"></div>
            <div id="quizFeedbackExplanation" class="render-math" style="font-size:0.925rem; line-height:1.5;"></div>
          </div>

          <!-- Navigation / Submit Button -->
          <div style="display:flex; justify-content:flex-end; gap:0.75rem; margin-top:1rem;">
            <button class="btn" id="btnQuizSkip">Skip</button>
            <button class="btn btn-primary" id="btnQuizSubmit">Submit Answer</button>
            <button class="btn btn-primary" id="btnQuizNext" style="display: none;">Next Question ➔</button>
          </div>
        </div>
      </div>
    </div>

    <!-- VIEW 3: ANALYTICS & SCORECARD VIEW -->
    <div class="view-panel" id="viewAnalytics">
      <div class="analytics-container">
        <div>
          <h2 style="font-family:var(--font-display); font-size:1.5rem; font-weight:700;">Benchmark Metrics & Scorecards</h2>
          <p style="color:var(--text-secondary); font-size:0.9rem;">Dataset composition, distribution breakdown, and baseline evaluation performance.</p>
        </div>

        <!-- Global Stat Cards -->
        <div class="stats-overview-grid">
          <div class="stat-card">
            <span class="stat-card-title">Total Samples</span>
            <span class="stat-card-value" id="statTotalSamples">252</span>
            <span class="stat-card-sub" id="statFormatBreakdown">234 MCQ / 18 Open</span>
          </div>
          <div class="stat-card">
            <span class="stat-card-title">Core Tasks</span>
            <span class="stat-card-value">6</span>
            <span class="stat-card-sub">SI, Dim, Conv, Homog, Rule, GUM</span>
          </div>
          <div class="stat-card">
            <span class="stat-card-title">Reviewed Progress</span>
            <span class="stat-card-value" id="statReviewedPct">0%</span>
            <span class="stat-card-sub" id="statReviewedCount">0 of 252 annotated</span>
          </div>
          <div class="stat-card">
            <span class="stat-card-title">Flagged Items</span>
            <span class="stat-card-value" style="color:var(--accent-amber);" id="statFlaggedCount">0</span>
            <span class="stat-card-sub">Requires revision</span>
          </div>
        </div>

        <!-- 2 Column Analytics Grids -->
        <div class="analytics-grid-2col">
          <!-- Task Distribution Table -->
          <div class="card">
            <div class="card-title"><span>⚡</span> Task Distribution</div>
            <div class="table-container">
              <table class="data-table" id="taskStatsTable">
                <thead>
                  <tr>
                    <th>Task</th>
                    <th>Count</th>
                    <th>Share</th>
                    <th>Progress</th>
                  </tr>
                </thead>
                <tbody id="taskStatsBody">
                  <!-- Injected -->
                </tbody>
              </table>
            </div>
          </div>

          <!-- Difficulty & Format Breakdown -->
          <div class="card">
            <div class="card-title"><span>📊</span> Difficulty & Format Breakdown</div>
            <div class="table-container">
              <table class="data-table" id="diffStatsTable">
                <thead>
                  <tr>
                    <th>Tier / Format</th>
                    <th>Count</th>
                    <th>Share</th>
                    <th>Bar</th>
                  </tr>
                </thead>
                <tbody id="diffStatsBody">
                  <!-- Injected -->
                </tbody>
              </table>
            </div>
          </div>
        </div>

        <!-- Baseline Evaluation Scorecard -->
        <div class="card" id="scorecardCard">
          <div class="card-title" style="display:flex; justify-content:space-between; align-items:center;">
            <span>🏆 Model Evaluation Baseline Scorecard</span>
            <span class="tag tag-task" id="scorecardModelName">ground_truth_baseline</span>
          </div>
          <p style="color:var(--text-secondary); font-size:0.875rem;">
            Evaluation pass rates across tasks and difficulty tiers based on deterministic rule-checking and symbolic verification.
          </p>
          <div class="table-container" style="margin-top:0.75rem;">
            <table class="data-table" id="scorecardTable">
              <thead>
                <tr>
                  <th>Category</th>
                  <th>Total Tested</th>
                  <th>Passed</th>
                  <th>Accuracy</th>
                  <th>Review Actions</th>
                </tr>
              </thead>
              <tbody id="scorecardBody">
                <!-- Injected from scorecard JSON -->
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>

    <!-- VIEW 4: REVIEW QUEUE & FLAGS VIEW -->
    <div class="view-panel" id="viewTriage">
      <div class="triage-container">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:1rem;">
          <div>
            <h2 style="font-family:var(--font-display); font-size:1.5rem; font-weight:700;">Review Queue & Flagged Samples</h2>
            <p style="color:var(--text-secondary); font-size:0.9rem;">Audit samples marked as flagged or containing reviewer feedback.</p>
          </div>
          <div style="display:flex; gap:0.5rem;">
            <button class="btn btn-primary" id="btnExportReviews">Export Reviews JSON</button>
            <button class="btn" id="btnClearAllReviews">Clear All Annotations</button>
          </div>
        </div>

        <div class="card">
          <div class="table-container">
            <table class="data-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Task</th>
                  <th>Difficulty</th>
                  <th>Status</th>
                  <th>Question Preview</th>
                  <th>Notes</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody id="triageTableBody">
                <!-- Injected dynamically -->
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>

  </main>

  <!-- Load File Modal -->
  <div class="modal-overlay" id="loadFileModal">
    <div class="modal-card">
      <div class="modal-header">
        <h3 class="modal-title">Load Benchmark Dataset</h3>
        <button class="icon-btn" id="btnCloseModal">✕</button>
      </div>
      <p style="font-size:0.85rem; color:var(--text-secondary);">
        Select a local JSONL or JSON benchmark file to browse and evaluate in the interactive inspector.
      </p>

      <div class="dropzone" id="fileDropzone">
        <span style="font-size:2rem;">📄</span>
        <strong style="font-size:0.95rem;">Drag & drop benchmark .jsonl or .json file here</strong>
        <span style="font-size:0.75rem; color:var(--text-muted);">or click to browse from disk</span>
        <input type="file" id="fileInput" accept=".jsonl,.json" style="display:none;" />
      </div>

      <div style="display:flex; justify-content:flex-end; gap:0.5rem;">
        <button class="btn" id="btnCancelModal">Cancel</button>
      </div>
    </div>
  </div>

  <!-- Shortcuts Modal -->
  <div class="modal-overlay" id="shortcutsModal">
    <div class="modal-card">
      <div class="modal-header">
        <h3 class="modal-title">Keyboard Shortcuts</h3>
        <button class="icon-btn" id="btnCloseShortcuts">✕</button>
      </div>
      <div class="table-container">
        <table class="data-table">
          <thead>
            <tr><th>Shortcut</th><th>Action</th></tr>
          </thead>
          <tbody>
            <tr><td><kbd>J</kbd> / <kbd>↓</kbd></td><td>Next sample</td></tr>
            <tr><td><kbd>K</kbd> / <kbd>↑</kbd></td><td>Previous sample</td></tr>
            <tr><td><kbd>V</kbd></td><td>Toggle Verify / Approved</td></tr>
            <tr><td><kbd>F</kbd></td><td>Toggle Flag for Revision</td></tr>
            <tr><td><kbd>/</kbd></td><td>Focus search bar</td></tr>
            <tr><td><kbd>1</kbd> - <kbd>4</kbd></td><td>Select Option A-D in Quiz Mode</td></tr>
            <tr><td><kbd>Esc</kbd></td><td>Close modals</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- Toast Notification Container -->
  <div class="toast" id="toast">
    <span id="toastIcon">✓</span>
    <span id="toastMsg">Operation successful</span>
  </div>

  <!-- Embedded Benchmark Data -->
  <script id="embeddedBenchmarkData" type="application/json">
    __DATA_JSON__
  </script>

  <!-- Embedded Scorecard Data -->
  <script id="embeddedScorecardData" type="application/json">
    __SCORECARD_JSON__
  </script>

  <!-- Core Interactive Application Logic -->
  <script>
    (function() {
      "use strict";

      // --- STATE & PERSISTENCE ---
      let rawData = [];
      let currentRecords = [];
      let filteredRecords = [];
      let selectedIndex = 0;
      let activeView = "explorer";
      let scorecardData = null;
      let evalResultsMap = {}; // sample.id -> grade object
      let evalFilter = "all";  // 'all' | 'passed' | 'failed'

      // LocalStorage Keys
      const STORAGE_KEY_REVIEWS = "drum_benchmark_reviews_v1";
      const STORAGE_KEY_THEME = "drum_benchmark_theme_v1";

      let reviews = {}; // id -> { status: 'approved'|'flagged'|'pending', notes: string }

      // --- INITIALIZATION ---
      function init() {
        loadSavedTheme();
        loadSavedReviews();
        loadEmbeddedData();
        setupEventListeners();
        renderActiveView();
      }

      function loadSavedTheme() {
        const saved = localStorage.getItem(STORAGE_KEY_THEME);
        if (saved) {
          document.documentElement.setAttribute("data-theme", saved);
          updateThemeIcon(saved);
        }
      }

      function loadSavedReviews() {
        try {
          const saved = localStorage.getItem(STORAGE_KEY_REVIEWS);
          if (saved) {
            reviews = JSON.parse(saved);
          }
        } catch (e) {
          reviews = {};
        }
      }

      function saveReviews() {
        try {
          localStorage.setItem(STORAGE_KEY_REVIEWS, JSON.stringify(reviews));
          updateTriageBadge();
        } catch (e) {
          console.error("Failed to save reviews to localStorage", e);
        }
      }

      function loadEmbeddedData() {
        try {
          const rawEl = document.getElementById("embeddedBenchmarkData");
          if (rawEl && rawEl.textContent.trim()) {
            rawData = JSON.parse(rawEl.textContent);
            currentRecords = [...rawData];
          }
        } catch (e) {
          console.error("Failed to parse embedded benchmark data", e);
          rawData = [];
          currentRecords = [];
        }

        try {
          const scEl = document.getElementById("embeddedScorecardData");
          if (scEl && scEl.textContent.trim() && scEl.textContent.trim() !== "null") {
            scorecardData = JSON.parse(scEl.textContent);
          }
        } catch (e) {
          console.error("Failed to parse scorecard data", e);
        }

        updateEvalResultsMap();
        applyFilters();
      }

      function updateEvalResultsMap() {
        evalResultsMap = {};
        if (scorecardData && Array.isArray(scorecardData.detailed_results)) {
          scorecardData.detailed_results.forEach(res => {
            if (res && res.id) {
              evalResultsMap[res.id] = res.grade || res;
            }
          });
        }
        updateScorecardSidebarUI();
      }

      function updateScorecardSidebarUI() {
        const bar = document.getElementById("scorecardSidebarBar");
        const select = document.getElementById("evalFilter");
        if (!bar || !select) return;

        const totalEval = Object.keys(evalResultsMap).length;
        if (totalEval > 0) {
          bar.style.display = "flex";
          select.style.display = "inline-block";

          const modelName = (scorecardData && scorecardData.model_name) || "Model";
          document.getElementById("sidebarModelName").textContent = modelName;

          let passed = 0;
          let failed = 0;
          Object.values(evalResultsMap).forEach(g => {
            if (g.passed) passed++;
            else failed++;
          });

          const pct = totalEval > 0 ? Math.round((passed / totalEval) * 100) : 0;
          const badgeEl = document.getElementById("sidebarScoreBadge");
          badgeEl.textContent = `${passed}/${totalEval} (${pct}%)`;
          badgeEl.className = `tag ${pct >= 70 ? "tag-eval-pass" : (pct >= 30 ? "tag-diff-intermediate" : "tag-eval-fail")}`;

          document.getElementById("countEvalAll").textContent = totalEval;
          document.getElementById("countEvalPassed").textContent = passed;
          document.getElementById("countEvalFailed").textContent = failed;
        } else {
          bar.style.display = "none";
          select.style.display = "none";
        }
      }

      function updateEvalPills() {
        document.querySelectorAll("#evalFilterPills .eval-filter-pill").forEach(p => {
          p.classList.toggle("active", p.getAttribute("data-eval") === evalFilter);
        });
        const select = document.getElementById("evalFilter");
        if (select) select.value = evalFilter;
      }

      // --- EVENT LISTENERS ---
      function setupEventListeners() {
        // View Navigation Tabs
        document.querySelectorAll(".nav-tab-btn").forEach(btn => {
          btn.addEventListener("click", () => {
            const view = btn.getAttribute("data-view");
            switchView(view);
          });
        });

        // Theme Toggle
        document.getElementById("btnThemeToggle").addEventListener("click", toggleTheme);

        // Search Input
        const searchInput = document.getElementById("searchInput");
        const searchClear = document.getElementById("searchClear");

        searchInput.addEventListener("input", () => {
          searchClear.style.display = searchInput.value ? "block" : "none";
          applyFilters();
        });

        searchClear.addEventListener("click", () => {
          searchInput.value = "";
          searchClear.style.display = "none";
          searchInput.focus();
          applyFilters();
        });

        // Task Filter Chips
        document.querySelectorAll("#taskFilters .filter-chip").forEach(chip => {
          chip.addEventListener("click", () => {
            document.querySelectorAll("#taskFilters .filter-chip").forEach(c => c.classList.remove("active"));
            chip.classList.add("active");
            applyFilters();
          });
        });

        // Eval Filter Pills
        document.querySelectorAll("#evalFilterPills .eval-filter-pill").forEach(pill => {
          pill.addEventListener("click", () => {
            evalFilter = pill.getAttribute("data-eval") || "all";
            updateEvalPills();
            applyFilters();
          });
        });

        // JSON Tab pills
        document.querySelectorAll("#jsonTabPills .eval-filter-pill").forEach(pill => {
          pill.addEventListener("click", () => {
            activeJsonTab = pill.getAttribute("data-jsontab") || "eval";
            updateJsonInspectorView();
          });
        });

        document.getElementById("btnCopyActiveJson")?.addEventListener("click", () => {
          const rawEl = document.getElementById("inspectRawJson");
          if (rawEl) {
            copyToClipboard(rawEl.textContent, "JSON copied to clipboard!");
          }
        });

        // Sub-filters & Sorting
        document.getElementById("diffFilter").addEventListener("change", applyFilters);
        document.getElementById("formatFilter").addEventListener("change", applyFilters);
        document.getElementById("evalFilter").addEventListener("change", (e) => {
          evalFilter = e.target.value;
          updateEvalPills();
          applyFilters();
        });
        document.getElementById("statusFilter").addEventListener("change", applyFilters);
        document.getElementById("sortSelect").addEventListener("change", applyFilters);

        // Actions in Inspector Header
        document.getElementById("btnApprove").addEventListener("click", () => toggleReviewStatus("approved"));
        document.getElementById("btnFlag").addEventListener("click", () => toggleReviewStatus("flagged"));
        document.getElementById("btnRandomSample").addEventListener("click", selectRandomSample);
        document.getElementById("btnCopyId").addEventListener("click", () => {
          if (filteredRecords[selectedIndex]) {
            copyToClipboard(filteredRecords[selectedIndex].id, "Sample ID copied to clipboard!");
          }
        });

        // Notes textarea
        const notesArea = document.getElementById("inspectReviewNotes");
        notesArea.addEventListener("input", () => {
          const item = filteredRecords[selectedIndex];
          if (!item) return;
          if (!reviews[item.id]) {
            reviews[item.id] = { status: "pending", notes: "" };
          }
          reviews[item.id].notes = notesArea.value;
          saveReviews();
          const statusLbl = document.getElementById("notesSaveStatus");
          statusLbl.textContent = "Saved";
        });

        // Modal triggers
        document.getElementById("btnLoadFile").addEventListener("click", openLoadModal);
        document.getElementById("btnCloseModal").addEventListener("click", closeLoadModal);
        document.getElementById("btnCancelModal").addEventListener("click", closeLoadModal);
        document.getElementById("btnShortcuts").addEventListener("click", openShortcutsModal);
        document.getElementById("btnCloseShortcuts").addEventListener("click", closeShortcutsModal);

        // Export Actions
        document.getElementById("btnExport").addEventListener("click", exportDataMenu);
        document.getElementById("btnExportReviews")?.addEventListener("click", exportReviewsJson);
        document.getElementById("btnClearAllReviews")?.addEventListener("click", clearAllReviews);

        // File Drag & Drop
        setupFileLoader();

        // Quiz Interactions
        setupQuizListeners();

        // Keyboard Shortcuts
        window.addEventListener("keydown", handleGlobalKeyDown);
      }

      // --- FILTERING & LIST RENDERING ---
      function applyFilters() {
        const query = document.getElementById("searchInput").value.trim().toLowerCase();
        const activeTaskChip = document.querySelector("#taskFilters .filter-chip.active");
        const taskFilter = activeTaskChip ? activeTaskChip.getAttribute("data-task") : "all";
        const diffFilter = document.getElementById("diffFilter").value;
        const formatFilter = document.getElementById("formatFilter").value;
        const statusFilter = document.getElementById("statusFilter").value;
        const sortBy = document.getElementById("sortSelect").value;

        filteredRecords = currentRecords.filter(item => {
          // Eval filter (Correct / Missed)
          if (evalFilter !== "all") {
            const grade = evalResultsMap[item.id];
            if (!grade) return false;
            if (evalFilter === "passed" && !grade.passed) return false;
            if (evalFilter === "failed" && grade.passed) return false;
          }
          // Task filter
          if (taskFilter !== "all" && item.task !== taskFilter) return false;
          // Difficulty filter
          if (diffFilter !== "all" && item.difficulty !== diffFilter) return false;
          // Format filter
          if (formatFilter !== "all" && item.format !== formatFilter) return false;
          // Status filter
          if (statusFilter !== "all") {
            const st = (reviews[item.id] && reviews[item.id].status) || "pending";
            if (st !== statusFilter) return false;
          }
          // Text search query
          if (query) {
            const matchId = (item.id || "").toLowerCase().includes(query);
            const matchQ = (item.question || "").toLowerCase().includes(query);
            const matchAns = (item.ground_truth_answer || "").toLowerCase().includes(query);
            const matchExp = (item.explanation || "").toLowerCase().includes(query);
            const matchUri = (item.entity_uri || "").toLowerCase().includes(query);
            const matchMeta = JSON.stringify(item.metadata || {}).toLowerCase().includes(query);
            if (!matchId && !matchQ && !matchAns && !matchExp && !matchUri && !matchMeta) {
              return false;
            }
          }
          return true;
        });

        // Sort
        if (sortBy === "id") {
          filteredRecords.sort((a, b) => a.id.localeCompare(b.id));
        } else if (sortBy === "task") {
          filteredRecords.sort((a, b) => (a.task || "").localeCompare(b.task || ""));
        } else if (sortBy === "difficulty") {
          const diffOrder = { introductory: 1, intermediate: 2, advanced: 3 };
          filteredRecords.sort((a, b) => (diffOrder[a.difficulty] || 0) - (diffOrder[b.difficulty] || 0));
        } else if (sortBy === "status") {
          const getStatus = id => (reviews[id] && reviews[id].status) || "pending";
          filteredRecords.sort((a, b) => getStatus(a.id).localeCompare(getStatus(b.id)));
        }

        // Update list counter
        document.getElementById("filteredCountLabel").textContent = `Showing ${filteredRecords.length} of ${currentRecords.length} samples`;

        if (selectedIndex >= filteredRecords.length) {
          selectedIndex = 0;
        }

        renderSampleList();
        renderInspector();
        updateTriageBadge();
      }

      function formatSampleSlug(id) {
        if (!id) return "";
        let s = id.replace(/^drum_bench_/, "");
        const m = s.match(/^([a-z]+)_(?:mcq_|open_)?(\\d+)$/i);
        if (m) {
          return `${m[1]} #${m[2]}`;
        }
        return s;
      }

      function renderSampleList() {
        const container = document.getElementById("sampleListContainer");
        container.innerHTML = "";

        if (filteredRecords.length === 0) {
          container.innerHTML = `
            <div style="padding:2rem; text-align:center; color:var(--text-muted); font-size:0.875rem;">
              No matching benchmark samples found.<br/>
              Try clearing filters or search query.
            </div>`;
          return;
        }

        filteredRecords.forEach((item, idx) => {
          const itemEl = document.createElement("div");
          itemEl.className = `sample-item ${idx === selectedIndex ? "active" : ""}`;
          itemEl.addEventListener("click", () => {
            selectedIndex = idx;
            updateActiveListItem();
            renderInspector();
          });

          const status = (reviews[item.id] && reviews[item.id].status) || "pending";
          const statusClass = status === "approved" ? "status-approved" : (status === "flagged" ? "status-flagged" : "status-pending");

          const task = item.task || "general";
          const diff = item.difficulty || "intermediate";
          const format = item.format || "mcq";

          const grade = evalResultsMap[item.id];
          let evalBadgeHtml = "";
          if (grade) {
            if (grade.passed) {
              evalBadgeHtml = `<span class="tag tag-eval-pass">✓ Correct</span>`;
            } else {
              const pred = grade.predicted_key ? `Pred: ${grade.predicted_key}` : "Missed";
              evalBadgeHtml = `<span class="tag tag-eval-fail">✗ ${escapeHtml(pred)}</span>`;
            }
          }

          const shortSlug = formatSampleSlug(item.id);

          itemEl.innerHTML = `
            <div class="sample-item-top">
              <div class="sample-item-header-left">
                <span class="status-dot ${statusClass}"></span>
                <span class="sample-item-id" title="${escapeHtml(item.id)}">${escapeHtml(shortSlug)}</span>
              </div>
              <div class="sample-tags">
                <span class="tag tag-task" data-task="${task}">${task}</span>
                ${evalBadgeHtml}
              </div>
            </div>
            <div class="sample-item-text">${escapeHtml(cleanMathText(item.question))}</div>
            <div class="sample-item-footer">
              <div class="sample-item-footer-left">
                <span class="tag tag-diff-${diff}">${diff.slice(0, 3)}</span>
                <span class="tag tag-format">${format === "mcq" ? "MCQ" : "Open"}</span>
              </div>
              <span class="sample-item-full-id" title="${escapeHtml(item.id)}">${escapeHtml(item.id)}</span>
            </div>
          `;

          container.appendChild(itemEl);
        });
      }

      function updateActiveListItem() {
        const items = document.querySelectorAll("#sampleListContainer .sample-item");
        items.forEach((el, idx) => {
          if (idx === selectedIndex) {
            el.classList.add("active");
            el.scrollIntoView({ block: "nearest", behavior: "smooth" });
          } else {
            el.classList.remove("active");
          }
        });
      }

      // --- INSPECTOR PANEL RENDERING ---
      function renderInspector() {
        const item = filteredRecords[selectedIndex];
        if (!item) {
          document.getElementById("inspectId").textContent = "-";
          document.getElementById("inspectQuestion").textContent = "No sample selected.";
          document.getElementById("inspectBadges").innerHTML = "";
          document.getElementById("inspectEvalBanner").style.display = "none";
          document.getElementById("inspectOptionsContainer").innerHTML = "";
          document.getElementById("inspectExplanation").textContent = "";
          document.getElementById("inspectRawJson").textContent = "{}";
          return;
        }

        // Header ID and badges
        document.getElementById("inspectId").textContent = item.id;
        const status = (reviews[item.id] && reviews[item.id].status) || "pending";
        const dot = document.getElementById("inspectStatusDot");
        dot.className = "status-dot " + (status === "approved" ? "status-approved" : (status === "flagged" ? "status-flagged" : "status-pending"));

        const grade = evalResultsMap[item.id];
        let evalTagInBadges = "";
        if (grade) {
          evalTagInBadges = grade.passed
            ? '<span class="tag tag-eval-pass">✓ Model Correct</span>'
            : `<span class="tag tag-eval-fail">✗ Model Missed (Predicted: ${escapeHtml(grade.predicted_key || "None")})</span>`;
        }

        const badgesEl = document.getElementById("inspectBadges");
        badgesEl.innerHTML = `
          ${evalTagInBadges}
          <span class="tag tag-task" data-task="${item.task}">${item.task}</span>
          <span class="tag tag-diff-${item.difficulty}">${item.difficulty}</span>
          <span class="tag tag-format">${item.format === "mcq" ? "Multiple Choice (Track A)" : "Free-Form Symbolic (Track B)"}</span>
          ${status === "approved" ? '<span class="tag badge-correct">✓ Verified Approved</span>' : ''}
          ${status === "flagged" ? '<span class="tag" style="background:rgba(245,158,11,0.2); color:#fbbf24;">🚩 Flagged for Revision</span>' : ''}
        `;

        // Model Evaluation Grade Banner
        const evalBannerEl = document.getElementById("inspectEvalBanner");
        if (grade) {
          evalBannerEl.style.display = "flex";
          const modelName = (scorecardData && scorecardData.model_name) || "Evaluated Model";
          const rawText = (grade.predicted_raw !== undefined && grade.predicted_raw !== null) ? String(grade.predicted_raw) : "";
          const hasRaw = rawText.trim().length > 0;

          const evalRecord = {
            id: item.id,
            task: item.task,
            format: item.format,
            difficulty: item.difficulty,
            grade: grade
          };
          const evalJsonStr = JSON.stringify(evalRecord, null, 2);

          let rawSectionHtml = "";
          if (hasRaw && rawText !== evalJsonStr) {
            rawSectionHtml = `
              <details class="eval-raw-toggle" style="margin-top:0.4rem;">
                <summary style="display:flex; justify-content:space-between; align-items:center; cursor:pointer; font-weight:600; color:var(--text-secondary);">
                  <span>📄 Raw Text Response from Model (${rawText.length} chars)</span>
                  <span style="font-size:0.725rem; color:var(--text-link);">Click to toggle</span>
                </summary>
                <pre style="margin-top:0.4rem; background:var(--bg-input); border:1px solid var(--border-color); border-radius:var(--radius-sm); padding:0.75rem; font-family:var(--font-mono); font-size:0.78rem; white-space:pre-wrap; word-break:break-all; max-height:220px; overflow-y:auto; color:var(--text-primary); line-height:1.45;">${escapeHtml(rawText)}</pre>
              </details>
            `;
          }

          const jsonSectionHtml = `
            <details class="eval-raw-toggle" open style="margin-top:0.4rem;">
              <summary style="display:flex; justify-content:space-between; align-items:center; cursor:pointer; font-weight:600; color:var(--text-secondary);">
                <span>📊 Scorecard Evaluation JSON Result</span>
                <span style="font-size:0.725rem; color:var(--text-link);">Click to toggle</span>
              </summary>
              <pre style="margin-top:0.4rem; background:var(--bg-input); border:1px solid var(--border-color); border-radius:var(--radius-sm); padding:0.75rem; font-family:var(--font-mono); font-size:0.78rem; white-space:pre-wrap; word-break:break-all; max-height:220px; overflow-y:auto; color:#7dd3fc; line-height:1.45;">${escapeHtml(evalJsonStr)}</pre>
            </details>
          `;

          const expHtml = (grade.predicted_explanation)
            ? `<div style="margin-top:0.45rem; padding:0.45rem 0.65rem; background:rgba(255,255,255,0.04); border-left:3px solid var(--primary-color); border-radius:var(--radius-sm); font-size:0.82rem; color:var(--text-secondary); line-height:1.45;">
                <strong style="color:var(--text-primary);">Model Stated Rationale:</strong> ${escapeHtml(grade.predicted_explanation)}
               </div>`
            : '';

          if (grade.passed) {
            evalBannerEl.className = "eval-banner eval-banner-pass";
            evalBannerEl.innerHTML = `
              <div class="eval-banner-header">
                <span class="eval-banner-badge">✅ MODEL PASSED (CORRECT)</span>
                <span class="eval-banner-model">Model: ${escapeHtml(modelName)}</span>
              </div>
              <div class="eval-banner-body">
                <div>
                  <strong>Model Prediction:</strong> Option <span class="eval-exp-key">${escapeHtml(grade.predicted_key || "Correct")}</span> matches canonical ground truth!
                </div>
                ${expHtml}
                ${jsonSectionHtml}
                ${rawSectionHtml}
              </div>
            `;
          } else {
            evalBannerEl.className = "eval-banner eval-banner-fail";
            const expKey = grade.expected_key || item.correct_option_key || item.ground_truth_answer || "-";
            const predKey = grade.predicted_key || "None / Malformed";
            evalBannerEl.innerHTML = `
              <div class="eval-banner-header">
                <span class="eval-banner-badge">❌ MODEL MISSED (INCORRECT)</span>
                <span class="eval-banner-model">Model: ${escapeHtml(modelName)}</span>
              </div>
              <div class="eval-banner-body">
                <div>
                  <strong>Model Predicted:</strong> <span class="eval-pred-key">${escapeHtml(predKey)}</span>
                  &nbsp;|&nbsp;
                  <strong>Expected Ground Truth:</strong> <span class="eval-exp-key">${escapeHtml(expKey)}</span>
                </div>
                ${grade.error ? `<div class="eval-error-msg">⚠️ ${escapeHtml(grade.error)}</div>` : ''}
                ${expHtml}
                ${jsonSectionHtml}
                ${rawSectionHtml}
              </div>
            `;
          }
        } else {
          evalBannerEl.style.display = "none";
          evalBannerEl.innerHTML = "";
        }

        // Question & Context
        document.getElementById("inspectQuestion").innerHTML = formatMathText(item.question);

        const ctxBox = document.getElementById("inspectContext");
        if (item.context) {
          ctxBox.style.display = "block";
          ctxBox.innerHTML = `<strong>Context:</strong> ` + formatMathText(item.context);
        } else {
          ctxBox.style.display = "none";
        }

        // Options & Answers
        const optContainer = document.getElementById("inspectOptionsContainer");
        optContainer.innerHTML = "";

        if (item.format === "mcq" && item.options && item.options.length > 0) {
          const grid = document.createElement("div");
          grid.className = "options-grid";

          item.options.forEach(opt => {
            const isCorrect = opt.is_correct || opt.key === item.correct_option_key;
            const isModelChosen = grade && grade.predicted_key === opt.key;

            let cardExtraClass = "";
            if (isModelChosen) {
              cardExtraClass = grade.passed ? " model-selected-correct" : " model-selected-wrong";
            }

            const optCard = document.createElement("div");
            optCard.className = `option-card ${isCorrect ? "correct" : "distractor"}${cardExtraClass}`;

            let badgesHtml = "";
            if (isCorrect) {
              badgesHtml += '<span class="option-status-badge badge-correct">Correct Ground Truth</span>';
            }
            if (isModelChosen) {
              if (grade.passed) {
                badgesHtml += ' <span class="badge-model-pred pass">🤖 Model Selected (Correct)</span>';
              } else {
                badgesHtml += ' <span class="badge-model-pred fail">🤖 Model Selected (Missed)</span>';
              }
            }

            let distractorHtml = "";
            if (!isCorrect && opt.distractor_rationale) {
              distractorHtml = `
                <div class="distractor-pill">
                  <span>⚠️ Distractor Flaw:</span>
                  <span>${escapeHtml(opt.distractor_rationale)}</span>
                </div>
              `;
            } else if (isCorrect) {
              distractorHtml = `
                <div class="distractor-pill" style="background:rgba(16,185,129,0.1); border-color:rgba(16,185,129,0.3); color:#34d399;">
                  <span>✓ Ground Truth Canonical Reference</span>
                </div>
              `;
            }

            optCard.innerHTML = `
              <div class="option-main">
                <div class="option-key-badge">${opt.key}</div>
                <div class="option-text">${formatMathText(opt.text)}</div>
                ${badgesHtml}
              </div>
              ${distractorHtml}
            `;
            grid.appendChild(optCard);
          });
          optContainer.appendChild(grid);
        } else {
          // Free form / Open representation
          const ffBox = document.createElement("div");
          ffBox.className = "ground-truth-box";
          ffBox.innerHTML = `
            <span style="font-size:0.75rem; text-transform:uppercase; font-weight:700; color:#38bdf8;">Canonical Ground Truth Target</span>
            <div class="ground-truth-val">${formatMathText(item.ground_truth_answer)}</div>
          `;
          optContainer.appendChild(ffBox);
        }

        // Explanation
        const expEl = document.getElementById("inspectExplanation");
        expEl.innerHTML = item.explanation ? formatMathText(item.explanation) : "<em>No derivation explanation provided.</em>";

        // Provenance & Metadata Anatomy
        const entityUriEl = document.getElementById("inspectEntityUri");
        if (item.entity_uri) {
          entityUriEl.innerHTML = `<a href="${escapeHtml(item.entity_uri)}" target="_blank" rel="noopener noreferrer" style="color:var(--text-link); text-decoration:underline;">${escapeHtml(item.entity_uri)}</a>`;
        } else {
          entityUriEl.textContent = "N/A";
        }

        // Dimension Vector Chips
        const dimVectorEl = document.getElementById("inspectDimVector");
        dimVectorEl.innerHTML = "";
        const dims = (item.metadata && item.metadata.dim_vector) || null;
        if (dims) {
          const baseNames = ["L", "M", "T", "I", "Theta", "N", "J"];
          baseNames.forEach(dim => {
            const val = dims[dim] || 0;
            const chip = document.createElement("span");
            chip.className = `dim-chip ${val !== 0 ? "active-dim" : ""}`;
            chip.textContent = `${dim === "Theta" ? "Θ" : dim}: ${val}`;
            dimVectorEl.appendChild(chip);
          });
        } else {
          dimVectorEl.innerHTML = `<span style="font-size:0.75rem; color:var(--text-muted);">No dimension vector specified</span>`;
        }

        // Standard Reference & Symbol
        const standardEl = document.getElementById("inspectStandard");
        const standardVal = (item.metadata && (item.metadata.standard || item.metadata.category)) || "SI Digital Framework (9th Ed)";
        standardEl.textContent = standardVal;

        const symbolEl = document.getElementById("inspectSymbol");
        const symVal = (item.metadata && (item.metadata.unit_symbol || item.metadata.constant_symbol)) || item.ground_truth_answer || "-";
        symbolEl.innerHTML = formatMathText(symVal);

        // Reviewer Notes
        const notesArea = document.getElementById("inspectReviewNotes");
        notesArea.value = (reviews[item.id] && reviews[item.id].notes) || "";

        // Raw JSON Views
        updateJsonInspectorView();

        // Trigger LaTeX Math Render
        renderMathInDOM();
      }

      let activeJsonTab = "eval"; // 'eval' | 'bench' | 'combined'

      function updateJsonInspectorView() {
        const item = filteredRecords[selectedIndex];
        const rawEl = document.getElementById("inspectRawJson");
        if (!item || !rawEl) return;

        const grade = evalResultsMap[item.id];
        const evalRecord = grade ? {
          id: item.id,
          task: item.task,
          format: item.format,
          difficulty: item.difficulty,
          grade: grade
        } : null;

        let tabToUse = activeJsonTab;
        if (tabToUse === "eval" && !evalRecord) {
          tabToUse = "bench";
        }

        document.querySelectorAll("#jsonTabPills .eval-filter-pill").forEach(pill => {
          pill.classList.toggle("active", pill.getAttribute("data-jsontab") === tabToUse);
        });

        let jsonToDisplay = {};
        if (tabToUse === "eval") {
          jsonToDisplay = evalRecord || { note: "No evaluation scorecard loaded for this item." };
        } else if (tabToUse === "bench") {
          jsonToDisplay = item;
        } else if (tabToUse === "combined") {
          jsonToDisplay = {
            ...item,
            evaluation: evalRecord || { note: "Not evaluated in loaded scorecard." }
          };
        }

        rawEl.textContent = JSON.stringify(jsonToDisplay, null, 2);
      }

      function toggleReviewStatus(newStatus) {
        const item = filteredRecords[selectedIndex];
        if (!item) return;

        if (!reviews[item.id]) {
          reviews[item.id] = { status: "pending", notes: "" };
        }

        if (reviews[item.id].status === newStatus) {
          reviews[item.id].status = "pending";
          showToast(`Status cleared for ${item.id}`);
        } else {
          reviews[item.id].status = newStatus;
          showToast(`Marked ${item.id} as ${newStatus.toUpperCase()}`);
        }

        saveReviews();
        updateActiveListItem();
        renderInspector();
        renderAnalytics();
      }

      function selectRandomSample() {
        if (filteredRecords.length === 0) return;
        selectedIndex = Math.floor(Math.random() * filteredRecords.length);
        updateActiveListItem();
        renderInspector();
      }

      // --- MATH RENDERING HELPER ---
      function formatMathText(str) {
        if (!str) return "";
        let s = escapeHtml(str);
        // Replace LaTeX inline formulas
        return s.replace(/\\\\text\\{([^}]+)\\}/g, "<strong>$1</strong>")
                .replace(/\\\\cdot/g, " · ")
                .replace(/\\\\Theta/g, "Θ")
                .replace(/\\\\Delta/g, "Δ")
                .replace(/\\\\nu/g, "ν")
                .replace(/\\\\pm/g, "±")
                .replace(/\\\\circ/g, "°");
      }

      function cleanMathText(str) {
        if (!str) return "";
        return str.replace(/\\text\\{([^}]+)\\}/g, "$1")
                  .replace(/\\cdot/g, "·")
                  .replace(/\\Theta/g, "Θ")
                  .replace(/\\Delta/g, "Δ")
                  .replace(/\\nu/g, "ν")
                  .replace(/\\pm/g, "±")
                  .replace(/\\circ/g, "°");
      }

      function renderMathInDOM() {
        if (window.renderMathInElement) {
          try {
            renderMathInElement(document.getElementById("inspectorPanel"), {
              delimiters: [
                { left: "$$", right: "$$", display: true },
                { left: "$", right: "$", display: false },
                { left: "\\(", right: "\\)", display: false },
                { left: "\\[", right: "\\]", display: true }
              ],
              throwOnError: false
            });
          } catch (e) {
            // KaTeX rendering fallback
          }
        }
      }

      // --- VIEW 2: QUIZ / TEST MODE ---
      let quizState = {
        queue: [],
        currentIndex: 0,
        correctCount: 0,
        attemptedCount: 0,
        streak: 0,
        selectedOption: null,
        isAnswered: false
      };

      function setupQuizListeners() {
        document.getElementById("btnQuizSubmit").addEventListener("click", submitQuizAnswer);
        document.getElementById("btnQuizNext").addEventListener("click", nextQuizQuestion);
        document.getElementById("btnQuizSkip").addEventListener("click", nextQuizQuestion);
        document.getElementById("btnQuizRestart").addEventListener("click", startQuiz);
      }

      function startQuiz() {
        const mcqItems = currentRecords.filter(r => r.format === "mcq" && r.options && r.options.length > 0);
        quizState.queue = shuffleArray([...mcqItems]);
        quizState.currentIndex = 0;
        quizState.correctCount = 0;
        quizState.attemptedCount = 0;
        quizState.streak = 0;
        quizState.selectedOption = null;
        quizState.isAnswered = false;

        document.getElementById("quizTotalQuestions").textContent = quizState.queue.length;
        renderQuizQuestion();
      }

      function renderQuizQuestion() {
        const item = quizState.queue[quizState.currentIndex];
        if (!item) {
          document.getElementById("quizQuestionText").textContent = "Quiz complete! Great job!";
          document.getElementById("quizOptionsGrid").innerHTML = "";
          return;
        }

        quizState.selectedOption = null;
        quizState.isAnswered = false;

        document.getElementById("quizCurrentIndex").textContent = quizState.currentIndex + 1;
        document.getElementById("quizScoreCorrect").textContent = quizState.correctCount;
        document.getElementById("quizScoreAttempted").textContent = quizState.attemptedCount;
        const acc = quizState.attemptedCount > 0 ? Math.round((quizState.correctCount / quizState.attemptedCount) * 100) : 0;
        document.getElementById("quizAccuracyPct").textContent = acc + "%";
        document.getElementById("quizStreak").textContent = quizState.streak;

        const fillPct = (quizState.currentIndex / Math.max(1, quizState.queue.length)) * 100;
        document.getElementById("quizProgressFill").style.width = fillPct + "%";

        document.getElementById("quizTaskBadge").textContent = item.task;
        document.getElementById("quizTaskTag").textContent = item.task;
        document.getElementById("quizTaskTag").setAttribute("data-task", item.task);
        document.getElementById("quizDiffTag").textContent = item.difficulty;
        document.getElementById("quizDiffTag").className = `tag tag-diff-${item.difficulty}`;

        document.getElementById("quizQuestionText").innerHTML = formatMathText(item.question);

        const optionsGrid = document.getElementById("quizOptionsGrid");
        optionsGrid.innerHTML = "";

        item.options.forEach((opt, idx) => {
          const btn = document.createElement("button");
          btn.className = "quiz-option-btn";
          btn.innerHTML = `
            <div class="option-key-badge">${opt.key}</div>
            <div style="flex:1;">${formatMathText(opt.text)}</div>
          `;
          btn.addEventListener("click", () => {
            if (quizState.isAnswered) return;
            document.querySelectorAll(".quiz-option-btn").forEach(b => b.classList.remove("selected"));
            btn.classList.add("selected");
            quizState.selectedOption = opt.key;
          });
          optionsGrid.appendChild(btn);
        });

        document.getElementById("quizFeedbackBox").style.display = "none";
        document.getElementById("btnQuizSubmit").style.display = "inline-flex";
        document.getElementById("btnQuizNext").style.display = "none";
        document.getElementById("btnQuizSkip").style.display = "inline-flex";

        renderMathInDOM();
      }

      function submitQuizAnswer() {
        if (!quizState.selectedOption) {
          showToast("Please select an option first!");
          return;
        }

        const item = quizState.queue[quizState.currentIndex];
        const isCorrect = quizState.selectedOption === item.correct_option_key;

        quizState.isAnswered = true;
        quizState.attemptedCount++;
        if (isCorrect) {
          quizState.correctCount++;
          quizState.streak++;
        } else {
          quizState.streak = 0;
        }

        const buttons = document.querySelectorAll(".quiz-option-btn");
        item.options.forEach((opt, idx) => {
          const btn = buttons[idx];
          if (btn) {
            if (opt.key === item.correct_option_key) {
              btn.classList.add("revealed-correct");
            } else if (opt.key === quizState.selectedOption) {
              btn.classList.add("revealed-incorrect");
            }
          }
        });

        const fbBox = document.getElementById("quizFeedbackBox");
        fbBox.style.display = "flex";
        fbBox.className = `quiz-feedback-box ${isCorrect ? "correct" : "incorrect"}`;

        const heading = document.getElementById("quizFeedbackHeading");
        heading.innerHTML = isCorrect ? "🎉 Correct! Metrologically Exact!" : "❌ Incorrect Option";

        const explanation = document.getElementById("quizFeedbackExplanation");
        let expText = item.explanation || "";
        if (!isCorrect) {
          const chosenOpt = item.options.find(o => o.key === quizState.selectedOption);
          if (chosenOpt && chosenOpt.distractor_rationale) {
            expText = `<strong>Why this distractor is flawed:</strong> ${chosenOpt.distractor_rationale}<br/><br/>` + expText;
          }
        }
        explanation.innerHTML = formatMathText(expText);

        document.getElementById("btnQuizSubmit").style.display = "none";
        document.getElementById("btnQuizNext").style.display = "inline-flex";
        document.getElementById("btnQuizSkip").style.display = "none";

        renderMathInDOM();
      }

      function nextQuizQuestion() {
        quizState.currentIndex++;
        if (quizState.currentIndex < quizState.queue.length) {
          renderQuizQuestion();
        } else {
          showToast("Quiz finished! Final score: " + quizState.correctCount + "/" + quizState.attemptedCount);
          startQuiz();
        }
      }

      // --- VIEW 3: ANALYTICS & SCORECARDS ---
      function renderAnalytics() {
        document.getElementById("statTotalSamples").textContent = currentRecords.length;
        const mcqCount = currentRecords.filter(r => r.format === "mcq").length;
        const openCount = currentRecords.filter(r => r.format === "free_form").length;
        document.getElementById("statFormatBreakdown").textContent = `${mcqCount} MCQ / ${openCount} Open`;

        const reviewedIds = Object.keys(reviews).filter(id => reviews[id].status === "approved" || reviews[id].status === "flagged");
        const flaggedCount = Object.keys(reviews).filter(id => reviews[id].status === "flagged").length;
        const pct = currentRecords.length > 0 ? Math.round((reviewedIds.length / currentRecords.length) * 100) : 0;

        document.getElementById("statReviewedPct").textContent = pct + "%";
        document.getElementById("statReviewedCount").textContent = `${reviewedIds.length} of ${currentRecords.length} annotated`;
        document.getElementById("statFlaggedCount").textContent = flaggedCount;

        // Task Stats Table
        const taskCounts = {};
        currentRecords.forEach(r => {
          taskCounts[r.task] = (taskCounts[r.task] || 0) + 1;
        });

        const taskTbody = document.getElementById("taskStatsBody");
        taskTbody.innerHTML = "";
        Object.keys(taskCounts).sort().forEach(task => {
          const count = taskCounts[task];
          const share = Math.round((count / currentRecords.length) * 100);
          const tr = document.createElement("tr");
          tr.innerHTML = `
            <td><span class="tag tag-task" data-task="${task}">${task}</span></td>
            <td><strong>${count}</strong></td>
            <td>${share}%</td>
            <td style="width:140px;">
              <div class="progress-bar-inline">
                <div class="progress-fill-inline" style="width:${share}%;"></div>
              </div>
            </td>
          `;
          taskTbody.appendChild(tr);
        });

        // Difficulty Stats
        const diffCounts = { introductory: 0, intermediate: 0, advanced: 0 };
        currentRecords.forEach(r => {
          if (diffCounts[r.difficulty] !== undefined) diffCounts[r.difficulty]++;
        });

        const diffTbody = document.getElementById("diffStatsBody");
        diffTbody.innerHTML = "";
        ["introductory", "intermediate", "advanced"].forEach(tier => {
          const count = diffCounts[tier] || 0;
          const share = Math.round((count / currentRecords.length) * 100);
          const tr = document.createElement("tr");
          tr.innerHTML = `
            <td><span class="tag tag-diff-${tier}">${tier}</span></td>
            <td><strong>${count}</strong></td>
            <td>${share}%</td>
            <td style="width:140px;">
              <div class="progress-bar-inline">
                <div class="progress-fill-inline" style="width:${share}%; background:var(--accent-blue);"></div>
              </div>
            </td>
          `;
          diffTbody.appendChild(tr);
        });

        renderScorecardTable();
      }

      function renderScorecardTable() {
        const scBody = document.getElementById("scorecardBody");
        scBody.innerHTML = "";

        if (!scorecardData || !scorecardData.task_breakdown) {
          scBody.innerHTML = `<tr><td colspan="5" style="text-align:center; color:var(--text-muted);">No evaluation baseline scorecard data found. Run <code>drum-ml evaluate</code> to generate a report.</td></tr>`;
          return;
        }

        document.getElementById("scorecardModelName").textContent = scorecardData.model_name || "baseline";

        const total = scorecardData.total_samples || 0;
        const passed = scorecardData.passed_samples || 0;
        const overallAcc = scorecardData.overall_accuracy_pct || (total > 0 ? (passed / total) * 100 : 0);
        const missedTotal = total - passed;

        const summaryTr = document.createElement("tr");
        summaryTr.style.fontWeight = "700";
        summaryTr.style.background = "rgba(16,185,129,0.08)";
        summaryTr.innerHTML = `
          <td>OVERALL SUITE</td>
          <td>${total}</td>
          <td>${passed}</td>
          <td style="color:${overallAcc >= 70 ? '#10b981' : (overallAcc >= 30 ? '#fbbf24' : '#f87171')};">${overallAcc.toFixed(1)}%</td>
          <td>
            <div style="display:flex; gap:0.4rem; flex-wrap:wrap;">
              ${missedTotal > 0 ? `<button class="btn btn-warning" style="padding:0.2rem 0.5rem; font-size:0.75rem;" onclick="window.filterByEvalMissed('all')">🔍 Review All Missed (${missedTotal})</button>` : ''}
              ${passed > 0 ? `<button class="btn btn-success" style="padding:0.2rem 0.5rem; font-size:0.75rem;" onclick="window.filterByEvalPassed('all')">✓ Review Passed (${passed})</button>` : ''}
            </div>
          </td>
        `;
        scBody.appendChild(summaryTr);

        for (const [task, data] of Object.entries(scorecardData.task_breakdown)) {
          const tr = document.createElement("tr");
          const acc = data.accuracy_pct !== undefined ? data.accuracy_pct : (data.total > 0 ? (data.passed / data.total) * 100 : 0);
          const missed = data.total - data.passed;
          tr.innerHTML = `
            <td><span class="tag tag-task" data-task="${task}">${task}</span></td>
            <td>${data.total}</td>
            <td>${data.passed}</td>
            <td><strong style="color:${acc >= 70 ? '#10b981' : (acc >= 30 ? '#fbbf24' : '#f87171')};">${acc.toFixed(1)}%</strong></td>
            <td>
              <div style="display:flex; gap:0.4rem; flex-wrap:wrap;">
                ${missed > 0 ? `<button class="btn btn-warning" style="padding:0.2rem 0.5rem; font-size:0.75rem;" onclick="window.filterByEvalMissed('${task}')">🔍 Review Missed (${missed})</button>` : ''}
                ${data.passed > 0 ? `<button class="btn btn-success" style="padding:0.2rem 0.5rem; font-size:0.75rem;" onclick="window.filterByEvalPassed('${task}')">✓ Passed (${data.passed})</button>` : ''}
              </div>
            </td>
          `;
          scBody.appendChild(tr);
        }
      }

      // Global window helpers for jumping from scorecard to explorer with filters applied
      window.filterByEvalMissed = function(task) {
        switchView("explorer");
        if (task && task !== "all") {
          document.querySelectorAll("#taskFilters .filter-chip").forEach(c => {
            c.classList.toggle("active", c.getAttribute("data-task") === task);
          });
        } else {
          document.querySelectorAll("#taskFilters .filter-chip").forEach(c => {
            c.classList.toggle("active", c.getAttribute("data-task") === "all");
          });
        }
        evalFilter = "failed";
        updateEvalPills();
        applyFilters();
      };

      window.filterByEvalPassed = function(task) {
        switchView("explorer");
        if (task && task !== "all") {
          document.querySelectorAll("#taskFilters .filter-chip").forEach(c => {
            c.classList.toggle("active", c.getAttribute("data-task") === task);
          });
        } else {
          document.querySelectorAll("#taskFilters .filter-chip").forEach(c => {
            c.classList.toggle("active", c.getAttribute("data-task") === "all");
          });
        }
        evalFilter = "passed";
        updateEvalPills();
        applyFilters();
      };

      // --- VIEW 4: TRIAGE / REVIEW QUEUE ---
      function renderTriage() {
        const tbody = document.getElementById("triageTableBody");
        tbody.innerHTML = "";

        const annotated = currentRecords.filter(r => {
          const s = reviews[r.id];
          return s && (s.status === "flagged" || (s.notes && s.notes.trim().length > 0));
        });

        if (annotated.length === 0) {
          tbody.innerHTML = `
            <tr>
              <td colspan="7" style="text-align:center; padding:2rem; color:var(--text-muted);">
                No flagged questions or notes yet. Use the Inspector to flag items that need review.
              </td>
            </tr>
          `;
          return;
        }

        annotated.forEach(item => {
          const rev = reviews[item.id] || {};
          const tr = document.createElement("tr");
          tr.innerHTML = `
            <td><strong style="font-family:var(--font-mono); font-size:0.75rem;">${escapeHtml(item.id)}</strong></td>
            <td><span class="tag tag-task" data-task="${item.task}">${item.task}</span></td>
            <td><span class="tag tag-diff-${item.difficulty}">${item.difficulty}</span></td>
            <td><span class="status-dot ${rev.status === 'flagged' ? 'status-flagged' : 'status-approved'}"></span> ${rev.status}</td>
            <td style="max-width:300px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${escapeHtml(cleanMathText(item.question))}</td>
            <td style="font-size:0.8rem; color:var(--accent-amber);">${escapeHtml(rev.notes || "-")}</td>
            <td>
              <button class="btn" style="padding:0.2rem 0.5rem; font-size:0.75rem;" data-jump="${item.id}">Inspect</button>
            </td>
          `;
          tr.querySelector("button").addEventListener("click", () => {
            jumpToSampleById(item.id);
          });
          tbody.appendChild(tr);
        });
      }

      function jumpToSampleById(id) {
        switchView("explorer");
        const idx = filteredRecords.findIndex(r => r.id === id);
        if (idx !== -1) {
          selectedIndex = idx;
          updateActiveListItem();
          renderInspector();
        }
      }

      function updateTriageBadge() {
        const flaggedCount = Object.keys(reviews).filter(id => reviews[id].status === "flagged").length;
        document.getElementById("triageBadgeCount").textContent = flaggedCount;
      }

      // --- EXPORT TOOLS ---
      function exportDataMenu() {
        const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(reviews, null, 2));
        const downloadAnchor = document.createElement("a");
        downloadAnchor.setAttribute("href", dataStr);
        downloadAnchor.setAttribute("download", "drum_benchmark_reviews.json");
        document.body.appendChild(downloadAnchor);
        downloadAnchor.click();
        downloadAnchor.remove();
        showToast("Exported review annotations JSON!");
      }

      function exportReviewsJson() {
        exportDataMenu();
      }

      function clearAllReviews() {
        if (confirm("Are you sure you want to clear all review annotations and flags?")) {
          reviews = {};
          saveReviews();
          renderTriage();
          renderAnalytics();
          renderInspector();
          renderSampleList();
          showToast("All review annotations cleared.");
        }
      }

      // --- FILE LOADER & DROPZONE ---
      function setupFileLoader() {
        const dropzone = document.getElementById("fileDropzone");
        const fileInput = document.getElementById("fileInput");

        dropzone.addEventListener("click", () => fileInput.click());

        dropzone.addEventListener("dragover", e => {
          e.preventDefault();
          dropzone.classList.add("dragover");
        });

        dropzone.addEventListener("dragleave", () => {
          dropzone.classList.remove("dragover");
        });

        dropzone.addEventListener("drop", e => {
          e.preventDefault();
          dropzone.classList.remove("dragover");
          if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
          }
        });

        fileInput.addEventListener("change", e => {
          if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
          }
        });
      }

      function handleFile(file) {
        const reader = new FileReader();
        reader.onload = e => {
          try {
            const text = e.target.result;
            const parsed = [];
            if (file.name.endsWith(".jsonl")) {
              const lines = text.split("\\n");
              for (const line of lines) {
                if (line.trim()) {
                  parsed.push(JSON.parse(line.trim()));
                }
              }
            } else {
              const j = JSON.parse(text);
              if (Array.isArray(j)) {
                parsed.push(...j);
              } else if (j.detailed_results || j.task_breakdown) {
                scorecardData = j;
                updateEvalResultsMap();
                renderAnalytics();
                applyFilters();
                showToast(`Loaded evaluation scorecard for '${j.model_name || "model"}'!`);
                closeLoadModal();
                return;
              }
            }

            if (parsed.length > 0) {
              currentRecords = parsed;
              selectedIndex = 0;
              applyFilters();
              closeLoadModal();
              showToast(`Loaded ${parsed.length} samples from ${file.name}`);
            } else {
              showToast("No valid records found in file.");
            }
          } catch (err) {
            console.error(err);
            showToast("Failed to parse file: " + err.message);
          }
        };
        reader.readAsText(file);
      }

      function openLoadModal() {
        document.getElementById("loadFileModal").classList.add("active");
      }
      function closeLoadModal() {
        document.getElementById("loadFileModal").classList.remove("active");
      }
      function openShortcutsModal() {
        document.getElementById("shortcutsModal").classList.add("active");
      }
      function closeShortcutsModal() {
        document.getElementById("shortcutsModal").classList.remove("active");
      }

      // --- VIEW SWITCHING ---
      function switchView(viewName) {
        activeView = viewName;
        document.querySelectorAll(".nav-tab-btn").forEach(btn => {
          btn.classList.toggle("active", btn.getAttribute("data-view") === viewName);
        });
        document.querySelectorAll(".view-panel").forEach(p => {
          p.classList.remove("active");
        });

        if (viewName === "explorer") {
          document.getElementById("viewExplorer").classList.add("active");
          renderInspector();
        } else if (viewName === "quiz") {
          document.getElementById("viewQuiz").classList.add("active");
          if (quizState.queue.length === 0) startQuiz();
        } else if (viewName === "analytics") {
          document.getElementById("viewAnalytics").classList.add("active");
          renderAnalytics();
        } else if (viewName === "triage") {
          document.getElementById("viewTriage").classList.add("active");
          renderTriage();
        }
      }

      function renderActiveView() {
        switchView(activeView);
      }

      // --- KEYBOARD SHORTCUTS ---
      function handleGlobalKeyDown(e) {
        if (["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName)) {
          if (e.key === "Escape") {
            e.target.blur();
          }
          return;
        }

        if (e.key === "/" && activeView === "explorer") {
          e.preventDefault();
          document.getElementById("searchInput").focus();
          return;
        }

        if (e.key === "Escape") {
          closeLoadModal();
          closeShortcutsModal();
          return;
        }

        if (activeView === "explorer") {
          if (e.key === "j" || e.key === "ArrowDown") {
            e.preventDefault();
            if (selectedIndex < filteredRecords.length - 1) {
              selectedIndex++;
              updateActiveListItem();
              renderInspector();
            }
          } else if (e.key === "k" || e.key === "ArrowUp") {
            e.preventDefault();
            if (selectedIndex > 0) {
              selectedIndex--;
              updateActiveListItem();
              renderInspector();
            }
          } else if (e.key === "v" || e.key === "V") {
            e.preventDefault();
            toggleReviewStatus("approved");
          } else if (e.key === "f" || e.key === "F") {
            e.preventDefault();
            toggleReviewStatus("flagged");
          }
        } else if (activeView === "quiz") {
          if (["1", "2", "3", "4"].includes(e.key)) {
            const idx = parseInt(e.key) - 1;
            const item = quizState.queue[quizState.currentIndex];
            if (item && item.options && item.options[idx]) {
              quizState.selectedOption = item.options[idx].key;
              const buttons = document.querySelectorAll(".quiz-option-btn");
              buttons.forEach((b, i) => b.classList.toggle("selected", i === idx));
            }
          } else if (e.key === "Enter") {
            if (!quizState.isAnswered) {
              submitQuizAnswer();
            } else {
              nextQuizQuestion();
            }
          }
        }
      }

      // --- UTILITIES ---
      function toggleTheme() {
        const current = document.documentElement.getAttribute("data-theme") || "dark";
        const target = current === "dark" ? "light" : "dark";
        document.documentElement.setAttribute("data-theme", target);
        localStorage.setItem(STORAGE_KEY_THEME, target);
        updateThemeIcon(target);
      }

      function updateThemeIcon(theme) {
        document.getElementById("themeIcon").textContent = theme === "dark" ? "🌙" : "☀️";
      }

      function showToast(msg) {
        const toast = document.getElementById("toast");
        document.getElementById("toastMsg").textContent = msg;
        toast.classList.add("show");
        setTimeout(() => toast.classList.remove("show"), 2800);
      }

      function copyToClipboard(text, successMsg) {
        navigator.clipboard.writeText(text).then(() => {
          showToast(successMsg || "Copied to clipboard!");
        }).catch(() => {
          showToast("Failed to copy to clipboard.");
        });
      }

      function escapeHtml(str) {
        if (!str) return "";
        return String(str)
          .replace(/&/g, "&amp;")
          .replace(/</g, "&lt;")
          .replace(/>/g, "&gt;")
          .replace(/"/g, "&quot;")
          .replace(/'/g, "&#039;");
      }

      function shuffleArray(arr) {
        for (let i = arr.length - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1));
          [arr[i], arr[j]] = [arr[j], arr[i]];
        }
        return arr;
      }

      if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
      } else {
        init();
      }

    })();
  </script>
</body>
</html>
"""


def generate_benchmark_html(
    benchmark_records: list[dict[str, Any]],
    scorecard: dict[str, Any] | None = None,
    dataset_title: str = "DRUM Metrology Benchmark (M-Eval)",
) -> str:
    """Generate the complete self-contained HTML page."""
    data_json_str = json.dumps(benchmark_records)
    scorecard_json_str = json.dumps(scorecard) if scorecard else "null"

    html = HTML_TEMPLATE.replace("__DATASET_TITLE__", dataset_title)
    html = html.replace("__DATA_JSON__", data_json_str)
    html = html.replace("__SCORECARD_JSON__", scorecard_json_str)
    return html


def save_benchmark_viewer(
    output_html_path: str | Path,
    benchmark_file: str | Path = "./dataset/benchmark/drum_benchmark_all.jsonl",
    scorecard_file: str | Path | None = "./dataset/benchmark/benchmark_report.json",
    open_browser: bool = False,
) -> Path:
    """Compile and save the standalone benchmark viewer HTML file."""
    records = load_jsonl_records(benchmark_file)
    scorecard = load_json_file(scorecard_file) if scorecard_file else None

    html = generate_benchmark_html(
        benchmark_records=records,
        scorecard=scorecard,
        dataset_title="DRUM Metrology Benchmark (M-Eval)",
    )

    out_p = Path(output_html_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        f.write(html)

    if open_browser:
        webbrowser.open(f"file://{out_p.resolve()}")

    return out_p


if __name__ == "__main__":
    import sys

    out_file = sys.argv[1] if len(sys.argv) > 1 else "dataset/benchmark/benchmark_viewer.html"
    bench_file = sys.argv[2] if len(sys.argv) > 2 else "dataset/benchmark/drum_benchmark_all.jsonl"
    report_file = sys.argv[3] if len(sys.argv) > 3 else "dataset/benchmark/benchmark_report.json"
    p = save_benchmark_viewer(out_file, bench_file, report_file)
    print(f"Generated benchmark viewer HTML at: {p.resolve()}")
