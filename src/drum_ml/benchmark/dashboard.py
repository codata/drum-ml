"""DRUM Metrology Benchmark (M-Eval) Multi-Model Leaderboard & Overall Results Dashboard.

Generates a standalone, zero-dependency interactive HTML dashboard comparing all
evaluated models and environment profiles across the 6 metrological sub-disciplines.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()


def collect_scorecards_from_dir(benchmark_dir: Path | str) -> list[dict[str, Any]]:
    """Scans a directory for all evaluation scorecard JSON files."""
    b_dir = Path(benchmark_dir)
    if not b_dir.exists():
        return []

    scorecards: list[dict[str, Any]] = []
    seen_models: set[str] = set()

    for p in sorted(b_dir.glob("*.json")):
        # Skip manifests or non-scorecard files
        if p.name in ["manifest.json", "entities.json", "scaffolds.json"]:
            continue
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "model_name" in data and ("task_breakdown" in data or "overall_accuracy_pct" in data or "passed_samples" in data):
                    data["_file_name"] = p.name
                    data["_file_path"] = str(p)
                    # Use unique identifier if duplicate model names
                    m_key = f"{data.get('model_name')}_{p.name}"
                    if m_key not in seen_models:
                        seen_models.add(m_key)
                        scorecards.append(data)
        except Exception:
            continue

    return scorecards


def generate_leaderboard_html(
    scorecards: list[dict[str, Any]],
    benchmark_samples: list[dict[str, Any]] | None = None,
    title: str = "DRUM Metrology Benchmark — Multi-Model Leaderboard & Overall Profiles Dashboard",
) -> str:
    """Generates a self-contained, interactive HTML dashboard for all evaluated models and profiles."""
    scorecards_json = json.dumps(scorecards, ensure_ascii=False)
    samples_json = json.dumps(benchmark_samples or [], ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&family=Outfit:wght@400;600;700;800&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>

  <style>
    :root {{
      --bg-base: #090d16;
      --bg-surface: #0f172a;
      --bg-card: rgba(30, 41, 59, 0.7);
      --bg-card-hover: rgba(51, 65, 85, 0.85);
      --bg-input: #0b1120;
      --border-color: rgba(255, 255, 255, 0.09);
      --border-active: #38bdf8;

      --text-primary: #f8fafc;
      --text-secondary: #94a3b8;
      --text-muted: #64748b;
      --text-link: #38bdf8;

      --primary-color: #38bdf8;
      --primary-gradient: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
      --accent-green: #10b981;
      --accent-purple: #a855f7;
      --accent-amber: #f59e0b;
      --accent-rose: #f43f5e;

      --radius-sm: 6px;
      --radius-md: 10px;
      --radius-lg: 16px;
      --radius-full: 9999px;

      --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.3);
      --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.4);
      --shadow-lg: 0 10px 30px -5px rgba(0, 0, 0, 0.6);
      --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      --font-display: 'Outfit', var(--font-sans);
      --font-mono: 'JetBrains Mono', monospace;
    }}

    * {{ box-sizing: border-box; margin: 0; padding: 0; }}

    body {{
      font-family: var(--font-sans);
      background-color: var(--bg-base);
      color: var(--text-primary);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      overflow-x: hidden;
      line-height: 1.5;
    }}

    /* Top Navbar */
    .navbar {{
      background: rgba(15, 23, 42, 0.85);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border-bottom: 1px solid var(--border-color);
      position: sticky;
      top: 0;
      z-index: 100;
      padding: 0.75rem 1.5rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
    }}

    .navbar-brand {{
      display: flex;
      align-items: center;
      gap: 0.75rem;
      text-decoration: none;
      color: inherit;
    }}

    .brand-icon {{
      width: 36px;
      height: 36px;
      border-radius: var(--radius-md);
      background: var(--primary-gradient);
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 800;
      font-size: 1.1rem;
      color: #0f172a;
      box-shadow: 0 0 16px rgba(56, 189, 248, 0.4);
    }}

    .brand-title {{
      font-family: var(--font-display);
      font-size: 1.15rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }}

    .brand-badge {{
      font-size: 0.7rem;
      padding: 0.15rem 0.5rem;
      background: rgba(56, 189, 248, 0.15);
      border: 1px solid rgba(56, 189, 248, 0.3);
      color: var(--primary-color);
      border-radius: var(--radius-full);
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}

    .nav-tabs {{
      display: flex;
      background: var(--bg-input);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-full);
      padding: 0.25rem;
      gap: 0.25rem;
    }}

    .nav-tab {{
      padding: 0.4rem 0.9rem;
      font-size: 0.825rem;
      font-weight: 600;
      color: var(--text-secondary);
      border: none;
      background: transparent;
      border-radius: var(--radius-full);
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }}

    .nav-tab:hover {{
      color: var(--text-primary);
    }}

    .nav-tab.active {{
      background: var(--primary-gradient);
      color: #090d16;
      font-weight: 700;
      box-shadow: var(--shadow-sm);
    }}

    .nav-actions {{
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }}

    /* Buttons */
    .btn {{
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      padding: 0.45rem 0.85rem;
      font-size: 0.8rem;
      font-weight: 600;
      border-radius: var(--radius-sm);
      border: 1px solid var(--border-color);
      background: var(--bg-surface);
      color: var(--text-primary);
      cursor: pointer;
      transition: all 0.2s ease;
      text-decoration: none;
    }}

    .btn:hover {{
      background: var(--bg-card-hover);
      border-color: var(--border-active);
    }}

    .btn-primary {{
      background: var(--primary-color);
      color: #090d16;
      border-color: transparent;
    }}

    .btn-primary:hover {{
      background: #7dd3fc;
      box-shadow: 0 0 12px rgba(56, 189, 248, 0.4);
    }}

    .btn-success {{
      background: rgba(16, 185, 129, 0.2);
      border-color: rgba(16, 185, 129, 0.4);
      color: #34d399;
    }}

    .btn-success:hover {{
      background: rgba(16, 185, 129, 0.3);
    }}

    /* Layout */
    .main-container {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 1.5rem;
      width: 100%;
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 1.5rem;
    }}

    /* Header Hero */
    .hero-banner {{
      background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.95) 100%);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-lg);
      padding: 1.5rem 2rem;
      display: flex;
      flex-direction: column;
      gap: 1rem;
      box-shadow: var(--shadow-md);
      position: relative;
      overflow: hidden;
    }}

    .hero-banner::after {{
      content: '';
      position: absolute;
      top: -50%;
      right: -20%;
      width: 400px;
      height: 400px;
      background: radial-gradient(circle, rgba(56, 189, 248, 0.12) 0%, rgba(0,0,0,0) 70%);
      pointer-events: none;
    }}

    .hero-title-row {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      flex-wrap: wrap;
      gap: 1rem;
    }}

    .hero-title {{
      font-family: var(--font-display);
      font-size: 1.65rem;
      font-weight: 800;
      letter-spacing: -0.02em;
      background: linear-gradient(135deg, #ffffff 30%, #94a3b8 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}

    .hero-subtitle {{
      color: var(--text-secondary);
      font-size: 0.9rem;
      max-width: 800px;
      margin-top: 0.25rem;
    }}

    /* Stats KPI Grid */
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 1rem;
    }}

    .kpi-card {{
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-md);
      padding: 1rem 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
      transition: all 0.2s ease;
    }}

    .kpi-card:hover {{
      border-color: rgba(56, 189, 248, 0.4);
      background: var(--bg-card-hover);
      transform: translateY(-2px);
    }}

    .kpi-label {{
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      font-weight: 700;
    }}

    .kpi-value {{
      font-family: var(--font-display);
      font-size: 1.6rem;
      font-weight: 800;
      color: var(--text-primary);
      display: flex;
      align-items: baseline;
      gap: 0.35rem;
    }}

    .kpi-subtext {{
      font-size: 0.75rem;
      color: var(--text-secondary);
    }}

    /* Filter Toolbar */
    .toolbar {{
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-md);
      padding: 0.75rem 1rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 0.75rem;
    }}

    .filter-group {{
      display: flex;
      align-items: center;
      gap: 0.5rem;
      flex-wrap: wrap;
    }}

    .filter-pill {{
      padding: 0.35rem 0.75rem;
      font-size: 0.75rem;
      font-weight: 600;
      border-radius: var(--radius-full);
      border: 1px solid var(--border-color);
      background: var(--bg-input);
      color: var(--text-secondary);
      cursor: pointer;
      transition: all 0.2s;
    }}

    .filter-pill:hover {{
      color: var(--text-primary);
      border-color: rgba(255, 255, 255, 0.2);
    }}

    .filter-pill.active {{
      background: rgba(56, 189, 248, 0.2);
      border-color: var(--primary-color);
      color: var(--primary-color);
      font-weight: 700;
    }}

    .search-box {{
      position: relative;
      min-width: 220px;
    }}

    .search-input {{
      width: 100%;
      background: var(--bg-input);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-sm);
      padding: 0.4rem 0.75rem 0.4rem 2rem;
      color: var(--text-primary);
      font-size: 0.8rem;
      outline: none;
      transition: border-color 0.2s;
    }}

    .search-input:focus {{
      border-color: var(--primary-color);
    }}

    .search-icon {{
      position: absolute;
      left: 0.6rem;
      top: 50%;
      transform: translateY(-50%);
      color: var(--text-muted);
      font-size: 0.8rem;
      pointer-events: none;
    }}

    /* View Panels */
    .view-panel {{
      display: none;
      flex-direction: column;
      gap: 1.5rem;
    }}

    .view-panel.active {{
      display: flex;
    }}

    /* Card Containers */
    .card {{
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-lg);
      padding: 1.25rem;
      box-shadow: var(--shadow-md);
    }}

    .card-title {{
      font-family: var(--font-display);
      font-size: 1.15rem;
      font-weight: 700;
      margin-bottom: 0.75rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.5rem;
    }}

    /* Leaderboard Table */
    .table-container {{
      overflow-x: auto;
      border-radius: var(--radius-md);
      border: 1px solid var(--border-color);
      background: var(--bg-surface);
    }}

    .data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.825rem;
      text-align: left;
    }}

    .data-table th {{
      background: #0b1220;
      padding: 0.75rem 0.85rem;
      font-weight: 700;
      color: var(--text-secondary);
      border-bottom: 1px solid var(--border-color);
      white-space: nowrap;
      cursor: pointer;
      user-select: none;
      transition: color 0.2s;
    }}

    .data-table th:hover {{
      color: var(--text-primary);
    }}

    .data-table th.sorted-asc::after {{
      content: " ▲";
      font-size: 0.65rem;
      color: var(--primary-color);
    }}

    .data-table th.sorted-desc::after {{
      content: " ▼";
      font-size: 0.65rem;
      color: var(--primary-color);
    }}

    .data-table td {{
      padding: 0.75rem 0.85rem;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
      color: var(--text-primary);
      vertical-align: middle;
    }}

    .data-table tr:hover td {{
      background: rgba(56, 189, 248, 0.04);
    }}

    .data-table tr.baseline-row td {{
      background: rgba(168, 85, 247, 0.06);
    }}

    /* Tags & Badges */
    .tag {{
      display: inline-flex;
      align-items: center;
      gap: 0.3rem;
      padding: 0.15rem 0.45rem;
      border-radius: var(--radius-sm);
      font-size: 0.7rem;
      font-weight: 600;
    }}

    .tag-local {{
      background: rgba(16, 185, 129, 0.15);
      border: 1px solid rgba(16, 185, 129, 0.3);
      color: #34d399;
    }}

    .tag-cloud {{
      background: rgba(56, 189, 248, 0.15);
      border: 1px solid rgba(56, 189, 248, 0.3);
      color: #38bdf8;
    }}

    .tag-baseline {{
      background: rgba(168, 85, 247, 0.15);
      border: 1px solid rgba(168, 85, 247, 0.3);
      color: #c084fc;
    }}

    .rank-badge {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 24px;
      height: 24px;
      border-radius: 50%;
      font-weight: 800;
      font-size: 0.75rem;
    }}

    .rank-1 {{ background: #eab308; color: #000; }}
    .rank-2 {{ background: #94a3b8; color: #000; }}
    .rank-3 {{ background: #b45309; color: #fff; }}
    .rank-other {{ background: var(--bg-input); color: var(--text-secondary); border: 1px solid var(--border-color); }}

    .score-cell {{
      font-weight: 700;
      display: flex;
      flex-direction: column;
      gap: 0.2rem;
    }}

    .progress-bar-wrap {{
      width: 100%;
      height: 5px;
      background: rgba(255, 255, 255, 0.1);
      border-radius: var(--radius-full);
      overflow: hidden;
    }}

    .progress-bar-fill {{
      height: 100%;
      border-radius: var(--radius-full);
    }}

    /* Radar & Chart Container */
    .chart-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 1.5rem;
    }}

    .chart-box {{
      background: var(--bg-surface);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-md);
      padding: 1.25rem;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      min-height: 380px;
      position: relative;
    }}

    .model-toggles {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      margin-bottom: 1rem;
      justify-content: center;
    }}

    .model-chip {{
      padding: 0.25rem 0.6rem;
      font-size: 0.725rem;
      font-weight: 600;
      border-radius: var(--radius-full);
      border: 1px solid var(--border-color);
      background: var(--bg-input);
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 0.35rem;
      transition: all 0.2s;
    }}

    .model-chip.selected {{
      border-color: currentColor;
    }}

    /* Side-by-Side Comparison Arena */
    .arena-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.5rem;
    }}

    @media (max-width: 900px) {{
      .arena-grid {{ grid-template-columns: 1fr; }}
      .chart-grid {{ grid-template-columns: 1fr; }}
    }}

    .arena-select-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 1rem;
      background: var(--bg-surface);
      padding: 1rem 1.25rem;
      border-radius: var(--radius-md);
      border: 1px solid var(--border-color);
    }}

    .model-select {{
      background: var(--bg-input);
      color: var(--text-primary);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-sm);
      padding: 0.45rem 0.75rem;
      font-size: 0.85rem;
      font-weight: 600;
      outline: none;
      min-width: 240px;
    }}

    .model-select:focus {{
      border-color: var(--primary-color);
    }}

    .comparison-card {{
      background: var(--bg-surface);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-md);
      padding: 1rem;
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }}

    .diff-item {{
      background: var(--bg-surface);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-md);
      padding: 1rem;
      margin-bottom: 0.75rem;
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }}

    .diff-item-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.75rem;
      color: var(--text-muted);
    }}

    .diff-responses {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem;
      margin-top: 0.5rem;
    }}

    @media (max-width: 768px) {{
      .diff-responses {{ grid-template-columns: 1fr; }}
    }}

    .resp-box {{
      background: var(--bg-input);
      border-radius: var(--radius-sm);
      padding: 0.75rem;
      font-size: 0.8rem;
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
      border-left: 3px solid var(--border-color);
    }}

    .resp-box.pass {{ border-left-color: var(--accent-green); }}
    .resp-box.fail {{ border-left-color: var(--accent-rose); }}

    /* Toast Notification */
    .toast {{
      position: fixed;
      bottom: 2rem;
      right: 2rem;
      background: #1e293b;
      border: 1px solid var(--primary-color);
      color: var(--text-primary);
      padding: 0.75rem 1.25rem;
      border-radius: var(--radius-md);
      box-shadow: var(--shadow-lg);
      font-size: 0.85rem;
      font-weight: 600;
      z-index: 999;
      opacity: 0;
      transform: translateY(10px);
      transition: all 0.25s ease;
      pointer-events: none;
    }}

    .toast.show {{
      opacity: 1;
      transform: translateY(0);
    }}
  </style>
</head>
<body>

  <!-- Top Navbar -->
  <header class="navbar">
    <div style="display:flex; align-items:center; gap:0.6rem;">
      <a href="../index.html" class="btn" title="Return to DRUM-ML Main Portal Home" style="text-decoration:none; display:inline-flex; align-items:center; gap:0.4rem; color:var(--text-secondary); font-size:0.75rem; font-weight:700; padding:0.35rem 0.65rem; border-radius:var(--radius-sm); background:var(--bg-subtle); border:1px solid var(--border-color); transition:all 0.2s;">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path><polyline points="9 22 9 12 15 12 15 22"></polyline></svg>
        <span>Home</span>
      </a>
      <a href="../index.html" class="navbar-brand" style="text-decoration:none;">
        <div class="brand-icon">D</div>
        <div class="brand-title">
          <span>DRUM-ML</span>
          <span class="brand-badge">Leaderboard & Profiles</span>
        </div>
      </a>
    </div>

    <!-- Navigation Tabs -->
    <nav class="nav-tabs">
      <button class="nav-tab active" data-tab="leaderboard">
        <span>🏆</span> Leaderboard Matrix
      </button>
      <button class="nav-tab" data-tab="visuals">
        <span>📊</span> Radar & Visuals
      </button>
      <button class="nav-tab" data-tab="arena">
        <span>⚔️</span> Model Arena
      </button>
      <button class="nav-tab" data-tab="hardware">
        <span>💻</span> Environment Profiles
      </button>
    </nav>

    <!-- Global Actions -->
    <div class="nav-actions">
      <a href="../dataset_viewer.html" class="btn" title="Open Dataset Browser & Explorer" style="text-decoration:none; display:inline-flex; align-items:center; gap:6px; font-size:0.75rem;">
        <span>📂</span> Dataset Browser
      </a>
      <a href="benchmark_viewer.html" class="btn" title="Open Single Model Benchmark Reviewer" style="text-decoration:none; display:inline-flex; align-items:center; gap:6px; font-size:0.75rem;">
        <span>🎯</span> Benchmark Viewer
      </a>
      <label class="btn" title="Upload additional scorecard JSON files" style="cursor:pointer; font-size:0.75rem;">
        <span>📂</span> Add Scorecard
        <input type="file" id="scorecardFileInput" multiple accept=".json" style="display:none;">
      </label>
      <button class="btn btn-primary" id="btnExportMarkdown" title="Export Leaderboard Markdown Table" style="font-size:0.75rem;">
        <span>📋</span> Copy Table
      </button>
    </div>
  </header>

  <!-- Main Container -->
  <main class="main-container">

    <!-- Hero Banner -->
    <section class="hero-banner">
      <div class="hero-title-row">
        <div>
          <h1 class="hero-title">Metrological Precision & Benchmark Leaderboard</h1>
          <p class="hero-subtitle">
            Autonomous multi-model evaluation across the 6 core metrology sub-disciplines (Physical Constants, Dimensional Decomposition, Unit Conversions, Homogeneity, SI Typography, and GUM Uncertainty).
          </p>
        </div>
      </div>

      <!-- Quick KPI Cards -->
      <div class="kpi-grid" id="kpiGrid">
        <div class="kpi-card">
          <div class="kpi-label">Models Evaluated</div>
          <div class="kpi-value" id="kpiModelCount">0</div>
          <div class="kpi-subtext" id="kpiModelBreakdown">0 Local | 0 Cloud</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Top Model Score</div>
          <div class="kpi-value" id="kpiTopScore" style="color:#34d399;">-%</div>
          <div class="kpi-subtext" id="kpiTopModelName">None</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Average Model Accuracy</div>
          <div class="kpi-value" id="kpiAvgScore" style="color:#38bdf8;">-%</div>
          <div class="kpi-subtext" id="kpiAvgSubtext">Across evaluated models</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Test Suite Size</div>
          <div class="kpi-value" id="kpiSampleCount">300</div>
          <div class="kpi-subtext" id="kpiSampleSubtext">Held-out Ground-Truth Items</div>
        </div>
      </div>
    </section>

    <!-- Toolbar Filters -->
    <div class="toolbar">
      <div style="display:flex; align-items:center; gap:1.25rem; flex-wrap:wrap;">
        <div class="filter-group">
          <span style="font-size:0.75rem; color:var(--text-muted); font-weight:700; text-transform:uppercase;">Format:</span>
          <button class="filter-pill active format-pill" data-format="all">All Formats</button>
          <button class="filter-pill format-pill" data-format="mcq">🔠 MCQ (Track A)</button>
          <button class="filter-pill format-pill" data-format="free_form">✍️ Free-Form (Track B)</button>
        </div>

        <div class="filter-group">
          <span style="font-size:0.75rem; color:var(--text-muted); font-weight:700; text-transform:uppercase;">Difficulty:</span>
          <button class="filter-pill active diff-pill" data-diff="all">All Difficulties</button>
          <button class="filter-pill diff-pill" data-diff="introductory">🟢 Introductory</button>
          <button class="filter-pill diff-pill" data-diff="intermediate">🟡 Intermediate</button>
          <button class="filter-pill diff-pill" data-diff="advanced">🔴 Advanced</button>
        </div>

        <div class="filter-group">
          <span style="font-size:0.75rem; color:var(--text-muted); font-weight:700; text-transform:uppercase;">Execution Target:</span>
          <button class="filter-pill active target-pill" data-filter="all">All Models</button>
          <button class="filter-pill target-pill" data-filter="local">💻 Local Models</button>
          <button class="filter-pill target-pill" data-filter="cloud">☁️ Cloud APIs</button>
          <button class="filter-pill target-pill" data-filter="baseline">🎯 Baseline Reference</button>
        </div>
      </div>

      <div class="search-box">
        <span class="search-icon">🔍</span>
        <input type="text" id="globalSearchInput" class="search-input" placeholder="Search models or hardware...">
      </div>
    </div>

    <!-- VIEW 1: LEADERBOARD MATRIX -->
    <section class="view-panel active" id="viewLeaderboard">
      <div class="card">
        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:1rem; border-bottom:1px solid var(--border-color); padding-bottom:0.75rem; flex-wrap:wrap; gap:0.75rem;">
          <div class="table-subtabs" style="display:flex; gap:0.5rem; flex-wrap:wrap;">
            <button class="filter-pill active table-tab-pill" data-table="subdiscipline">🏆 Sub-Discipline Matrix</button>
            <button class="filter-pill table-tab-pill" data-table="difficulty">🎯 Difficulty Tier Breakdown</button>
            <button class="filter-pill table-tab-pill" data-table="tracks">📑 Track A (MCQ) vs. Track B</button>
          </div>
          <span style="font-size:0.8rem; color:var(--text-muted); font-weight:500;">Click any header to sort</span>
        </div>

        <div class="table-container">
          <table class="data-table" id="leaderboardTable">
            <thead id="leaderboardHead">
              <!-- Injected dynamically by JS -->
            </thead>
            <tbody id="leaderboardBody">
              <!-- Injected dynamically by JS -->
            </tbody>
          </table>
        </div>
      </div>
    </section>

    <!-- VIEW 2: RADAR & CAPABILITY VISUALS -->
    <section class="view-panel" id="viewVisuals">
      <div class="card">
        <div class="card-title">
          <span>📊 Metrological Capability Radar Comparison</span>
        </div>
        <p style="color:var(--text-secondary); font-size:0.85rem; margin-bottom:1rem;">
          Select models to overlay and compare strengths across all 6 metrology sub-disciplines.
        </p>

        <div class="model-toggles" id="radarModelToggles">
          <!-- Model chips injected here -->
        </div>

        <div class="chart-grid">
          <div class="chart-box">
            <svg id="radarSvg" width="400" height="360" viewBox="0 0 400 360"></svg>
          </div>
          <div class="chart-box" style="align-items:stretch;">
            <div style="font-weight:700; font-size:0.9rem; margin-bottom:0.75rem; color:var(--text-primary);">Sub-Discipline Performance Breakdown</div>
            <div id="taskBarCharts" style="display:flex; flex-direction:column; gap:0.75rem; width:100%; overflow-y:auto; max-height:300px;">
              <!-- Injected by JS -->
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- VIEW 3: MODEL COMPARISON ARENA -->
    <section class="view-panel" id="viewArena">
      <div class="card">
        <div class="arena-select-row">
          <div>
            <label style="font-size:0.75rem; color:var(--text-muted); font-weight:700; text-transform:uppercase; display:block; margin-bottom:0.25rem;">Model A (Baseline)</label>
            <select id="arenaModelA" class="model-select"></select>
          </div>
          <div style="font-weight:800; font-size:1.25rem; color:var(--primary-color);">VS</div>
          <div>
            <label style="font-size:0.75rem; color:var(--text-muted); font-weight:700; text-transform:uppercase; display:block; margin-bottom:0.25rem;">Model B (Challenger)</label>
            <select id="arenaModelB" class="model-select"></select>
          </div>
          <div>
            <label style="font-size:0.75rem; color:var(--text-muted); font-weight:700; text-transform:uppercase; display:block; margin-bottom:0.25rem;">Differential Filter</label>
            <select id="arenaDiffFilter" class="model-select" style="min-width:180px;">
              <option value="all">All Differences</option>
              <option value="a_wins">Model A Passed / Model B Missed</option>
              <option value="b_wins">Model B Passed / Model A Missed</option>
              <option value="both_failed">Both Failed (Hard Questions)</option>
              <option value="both_passed">Both Passed</option>
            </select>
          </div>
        </div>

        <!-- Comparison Summary Cards -->
        <div class="kpi-grid" style="margin-top:1rem;" id="arenaKpiGrid">
          <!-- Injected by JS -->
        </div>

        <!-- Differential Questions List -->
        <div style="margin-top:1.5rem;">
          <h3 style="font-size:1rem; font-weight:700; margin-bottom:0.75rem;">Differential Question Analysis</h3>
          <div id="arenaDiffList">
            <!-- Injected by JS -->
          </div>
        </div>
      </div>
    </section>

    <!-- VIEW 4: HARDWARE & ENVIRONMENT PROFILES -->
    <section class="view-panel" id="viewHardware">
      <div class="card">
        <div class="card-title">
          <span>💻 Hardware, Accelerator & Environment Profiles</span>
        </div>
        <p style="color:var(--text-secondary); font-size:0.85rem; margin-bottom:1rem;">
          Full environment and hardware execution telemetry captured during evaluation runs.
        </p>

        <div class="table-container">
          <table class="data-table" id="hardwareTable">
            <thead>
              <tr>
                <th>Model</th>
                <th>Target / Execution</th>
                <th>Host Operating System</th>
                <th>CPU Cores</th>
                <th>System Memory</th>
                <th>GPU Accelerator & VRAM</th>
                <th>Python Version</th>
                <th>Evaluation Timestamp</th>
              </tr>
            </thead>
            <tbody id="hardwareBody">
              <!-- Injected by JS -->
            </tbody>
          </table>
        </div>
      </div>
    </section>

  </main>

  <!-- Toast Notification -->
  <div id="toast" class="toast">Action completed!</div>

  <!-- Embedded Data -->
  <script id="embeddedLeaderboardData" type="application/json">
{scorecards_json}
  </script>
  <script id="embeddedBenchmarkData" type="application/json">
{samples_json}
  </script>

  <script>
    (function() {{
      let scorecards = [];
      let benchmarkSamples = [];
      let activeFilter = 'all';
      let activeFormat = 'all';
      let activeDiff = 'all';
      let activeTable = 'subdiscipline';
      let searchQuery = '';
      let sortCol = 'overall';
      let sortDir = 'desc';

      const COLORS = ['#38bdf8', '#34d399', '#f43f5e', '#a855f7', '#fbbf24', '#ec4899', '#06b6d4'];

      // 1. Ingestion
      try {{
        const rawScorecards = document.getElementById("embeddedLeaderboardData").textContent;
        scorecards = JSON.parse(rawScorecards) || [];
      }} catch (e) {{
        console.error("Failed to parse embedded scorecards", e);
      }}

      try {{
        const rawSamples = document.getElementById("embeddedBenchmarkData").textContent;
        benchmarkSamples = JSON.parse(rawSamples) || [];
      }} catch (e) {{
        console.error("Failed to parse embedded benchmark samples", e);
      }}

      // Init UI
      initUI();

      function initUI() {{
        setupNavigation();
        setupFileUpload();
        setupFiltersAndSearch();
        setupExport();
        renderAll();
        checkHttpLiveSync();
      }}

      async function checkHttpLiveSync() {{
        if (!window.location.protocol.startsWith("http")) return;
        try {{
          const filesToCheck = [
            "gemma4_e2b_scorecard.json",
            "gemma4_12b_scorecard.json",
            "benchmark_report.json"
          ];
          let updated = false;
          for (const f of filesToCheck) {{
            try {{
              const resp = await fetch(f, {{ cache: "no-store" }});
              if (resp.ok) {{
                const data = await resp.json();
                if (data && data.model_name) {{
                  data._file_name = f;
                  const idx = scorecards.findIndex(s => s.model_name === data.model_name || s._file_name === f);
                  if (idx >= 0) {{
                    scorecards[idx] = data;
                  }} else {{
                    scorecards.push(data);
                  }}
                  updated = true;
                }}
              }}
            }} catch (e) {{}}
          }}
          if (updated) {{
            renderAll();
          }}
        }} catch (e) {{}}
      }}

      function renderAll() {{
        updateHeroKPIs();
        renderLeaderboardTable();
        renderRadarView();
        renderArenaView();
        renderHardwareTable();
      }}

      // 2. Navigation
      function setupNavigation() {{
        document.querySelectorAll(".nav-tab").forEach(tab => {{
          tab.addEventListener("click", () => {{
            document.querySelectorAll(".nav-tab").forEach(t => t.classList.remove("active"));
            tab.classList.add("active");
            const target = tab.dataset.tab;
            document.querySelectorAll(".view-panel").forEach(p => p.classList.remove("active"));
            if (target === "leaderboard") document.getElementById("viewLeaderboard").classList.add("active");
            if (target === "visuals") document.getElementById("viewVisuals").classList.add("active");
            if (target === "arena") document.getElementById("viewArena").classList.add("active");
            if (target === "hardware") document.getElementById("viewHardware").classList.add("active");
          }});
        }});
      }}

      // 3. File Upload
      function setupFileUpload() {{
        const input = document.getElementById("scorecardFileInput");
        input.addEventListener("change", async (e) => {{
          const files = Array.from(e.target.files);
          let loaded = 0;
          for (const file of files) {{
            try {{
              const text = await file.text();
              const data = JSON.parse(text);
              if (data && data.model_name) {{
                data._file_name = file.name;
                const idx = scorecards.findIndex(s => s.model_name === data.model_name);
                if (idx >= 0) {{
                  scorecards[idx] = data;
                }} else {{
                  scorecards.push(data);
                }}
                loaded++;
              }}
            }} catch (err) {{
              console.error("Error reading file", file.name, err);
            }}
          }}
          if (loaded > 0) {{
            showToast(`Loaded ${{loaded}} scorecard(s) successfully!`);
            renderAll();
          }}
        }});
      }}

      // Format & Difficulty Statistics Engine
      function getModelStats(sc, fmt, diff) {{
        if (!sc) return {{ has_data: false, overall_accuracy_pct: null, passed: 0, total: 0, task_breakdown: {{}}, difficulty_breakdown: {{}}, track_breakdown: {{}} }};

        // 1. Calculate from detailed_results if available
        if (Array.isArray(sc.detailed_results) && sc.detailed_results.length > 0) {{
          const isMcq = fmt === 'mcq';
          const isFree = fmt === 'free_form';

          const filtered = sc.detailed_results.filter(r => {{
            const f = (r.format || 'mcq').toLowerCase();
            const d = (r.difficulty || 'intermediate').toLowerCase();

            const matchesFmt = (fmt === 'all') || (isMcq ? f === 'mcq' : (f === 'free_form' || f === 'open'));
            const matchesDiff = (!diff || diff === 'all') || (d === diff.toLowerCase());
            return matchesFmt && matchesDiff;
          }});

          if (filtered.length > 0) {{
            let passed = 0;
            const tasks = ['constants', 'dimensions', 'conversions', 'homogeneity', 'conventions', 'uncertainty'];
            const tb = {{}};
            tasks.forEach(t => {{ tb[t] = {{ task: t, total: 0, passed: 0, accuracy_pct: 0 }}; }});

            const difficulties = ['introductory', 'intermediate', 'advanced'];
            const db = {{}};
            difficulties.forEach(d => {{ db[d] = {{ difficulty: d, total: 0, passed: 0, accuracy_pct: 0 }}; }});

            const trb = {{
              mcq: {{ total: 0, passed: 0, accuracy_pct: 0 }},
              free_form: {{ total: 0, passed: 0, accuracy_pct: 0 }},
              exact_matches: 0,
              symbolic_matches: 0
            }};

            filtered.forEach(r => {{
              const isPass = !!(r.grade && r.grade.passed);
              if (isPass) passed++;

              const t = r.task;
              if (t && tb[t]) {{
                tb[t].total++;
                if (isPass) tb[t].passed++;
              }}

              const d = (r.difficulty || 'intermediate').toLowerCase();
              if (d && db[d]) {{
                db[d].total++;
                if (isPass) db[d].passed++;
              }}

              const f = (r.format || 'mcq').toLowerCase();
              if (f === 'mcq') {{
                trb.mcq.total++;
                if (isPass) trb.mcq.passed++;
              }} else {{
                trb.free_form.total++;
                if (isPass) trb.free_form.passed++;
              }}

              const g = r.grade || {{}};
              if (isPass) {{
                if (g.match_type === 'exact_normalized' || g.exact_match || g.predicted_option || g.predicted_key) {{
                  trb.exact_matches++;
                }}
                if (g.match_type === 'sympy_algebraic' || g.match_type === 'pint_unit_conversion' || g.symbolic_match) {{
                  trb.symbolic_matches++;
                }}
              }}
            }});

            tasks.forEach(t => {{
              tb[t].accuracy_pct = tb[t].total > 0 ? (tb[t].passed / tb[t].total) * 100 : 0;
            }});

            difficulties.forEach(d => {{
              db[d].accuracy_pct = db[d].total > 0 ? (db[d].passed / db[d].total) * 100 : 0;
            }});

            trb.mcq.accuracy_pct = trb.mcq.total > 0 ? (trb.mcq.passed / trb.mcq.total) * 100 : 0;
            trb.free_form.accuracy_pct = trb.free_form.total > 0 ? (trb.free_form.passed / trb.free_form.total) * 100 : 0;

            return {{
              has_data: true,
              overall_accuracy_pct: (passed / filtered.length) * 100,
              passed: passed,
              total: filtered.length,
              task_breakdown: tb,
              difficulty_breakdown: db,
              track_breakdown: trb
            }};
          }}
        }}

        // 2. Fallback for summarized records
        let overallAcc = sc.overall_accuracy_pct || 0;
        let passedCount = sc.passed_samples || 0;
        let totalCount = sc.total_samples || 0;
        let tb = sc.task_breakdown || {{}};

        if (fmt !== 'all' && sc.format_breakdown) {{
          const key = fmt === 'mcq' ? 'mcq' : (sc.format_breakdown['free_form'] ? 'free_form' : 'open');
          const fb = sc.format_breakdown[key];
          if (fb && fb.total > 0) {{
            overallAcc = fb.accuracy_pct !== undefined ? fb.accuracy_pct : (fb.passed / fb.total) * 100;
            passedCount = fb.passed || 0;
            totalCount = fb.total || 0;
          }} else {{
            return {{ has_data: false, overall_accuracy_pct: null, passed: 0, total: 0, task_breakdown: {{}}, difficulty_breakdown: {{}}, track_breakdown: {{}} }};
          }}
        }}

        if (diff !== 'all' && sc.difficulty_breakdown) {{
          const dbObj = sc.difficulty_breakdown[diff];
          if (dbObj && dbObj.total > 0) {{
            overallAcc = dbObj.accuracy_pct !== undefined ? dbObj.accuracy_pct : (dbObj.passed / dbObj.total) * 100;
            passedCount = dbObj.passed || 0;
            totalCount = dbObj.total || 0;
          }} else {{
            return {{ has_data: false, overall_accuracy_pct: null, passed: 0, total: 0, task_breakdown: {{}}, difficulty_breakdown: {{}}, track_breakdown: {{}} }};
          }}
        }}

        return {{
          has_data: totalCount > 0,
          overall_accuracy_pct: totalCount > 0 ? overallAcc : null,
          passed: passedCount,
          total: totalCount,
          task_breakdown: tb,
          difficulty_breakdown: sc.difficulty_breakdown || {{}},
          track_breakdown: {{
            mcq: (sc.format_breakdown && sc.format_breakdown.mcq) || {{ total: totalCount, passed: passedCount, accuracy_pct: overallAcc }},
            free_form: (sc.format_breakdown && (sc.format_breakdown.free_form || sc.format_breakdown.open)) || {{ total: 0, passed: 0, accuracy_pct: 0 }},
            exact_matches: passedCount,
            symbolic_matches: 0
          }}
        }};
      }}

      // 4. Filters & Search Setup
      function setupFiltersAndSearch() {{
        // Target filter pills
        document.querySelectorAll(".target-pill").forEach(pill => {{
          pill.addEventListener("click", () => {{
            document.querySelectorAll(".target-pill").forEach(p => p.classList.remove("active"));
            pill.classList.add("active");
            activeFilter = pill.dataset.filter;
            updateHeroKPIs();
            renderLeaderboardTable();
            renderHardwareTable();
          }});
        }});

        // Format filter pills
        document.querySelectorAll(".format-pill").forEach(pill => {{
          pill.addEventListener("click", () => {{
            document.querySelectorAll(".format-pill").forEach(p => p.classList.remove("active"));
            pill.classList.add("active");
            activeFormat = pill.dataset.format;
            updateHeroKPIs();
            renderLeaderboardTable();
            renderRadarView();
            updateArenaComparison();
          }});
        }});

        // Difficulty filter pills
        document.querySelectorAll(".diff-pill").forEach(pill => {{
          pill.addEventListener("click", () => {{
            document.querySelectorAll(".diff-pill").forEach(p => p.classList.remove("active"));
            pill.classList.add("active");
            activeDiff = pill.dataset.diff;
            updateHeroKPIs();
            renderLeaderboardTable();
            renderRadarView();
            updateArenaComparison();
          }});
        }});

        // Table subtab pills
        document.querySelectorAll(".table-tab-pill").forEach(pill => {{
          pill.addEventListener("click", () => {{
            document.querySelectorAll(".table-tab-pill").forEach(p => p.classList.remove("active"));
            pill.classList.add("active");
            activeTable = pill.dataset.table;
            renderLeaderboardTable();
          }});
        }});

        const searchInput = document.getElementById("globalSearchInput");
        searchInput.addEventListener("input", (e) => {{
          searchQuery = e.target.value.toLowerCase().trim();
          renderLeaderboardTable();
          renderHardwareTable();
        }});
      }}

      // 5. KPIs
      function updateHeroKPIs() {{
        // Exclude ground truth baseline from competitive KPIs unless baseline filter is explicitly active
        const evalModels = scorecards.filter(sc => {{
          const env = sc.environment || {{}};
          const isBaseline = (sc.model_name && sc.model_name.toLowerCase().includes("baseline")) || env.execution_type === 'baseline';
          if (activeFilter === 'baseline') return isBaseline;
          return !isBaseline;
        }});

        document.getElementById("kpiModelCount").textContent = evalModels.length;

        let localCount = 0;
        let cloudCount = 0;
        let topScore = -1;
        let topModel = "None";
        let totalAcc = 0;
        let accCount = 0;

        evalModels.forEach(sc => {{
          const env = sc.environment || {{}};
          const isLoc = env.execution_type === 'local' || env.is_local_inference;
          if (isLoc) localCount++; else cloudCount++;

          const stats = getModelStats(sc, activeFormat, activeDiff);
          if (stats.has_data && stats.overall_accuracy_pct !== null) {{
            const acc = stats.overall_accuracy_pct;
            totalAcc += acc;
            accCount++;
            if (acc > topScore) {{
              topScore = acc;
              topModel = sc.model_name;
            }}
          }}
        }});

        const avgAcc = accCount > 0 ? (totalAcc / accCount) : -1;

        // Compute sample count matching active format and difficulty
        let filteredBenchmark = benchmarkSamples;
        if (activeFormat !== 'all') {{
          filteredBenchmark = filteredBenchmark.filter(s => {{
            const f = (s.format || 'mcq').toLowerCase();
            return activeFormat === 'mcq' ? f === 'mcq' : (f === 'free_form' || f === 'open');
          }});
        }}
        if (activeDiff !== 'all') {{
          filteredBenchmark = filteredBenchmark.filter(s => {{
            const d = (s.difficulty || 'intermediate').toLowerCase();
            return d === activeDiff.toLowerCase();
          }});
        }}
        const sampleCount = (filteredBenchmark && filteredBenchmark.length > 0) ? filteredBenchmark.length : (activeFormat === 'all' && activeDiff === 'all' ? 300 : 0);

        document.getElementById("kpiModelBreakdown").textContent = `${{localCount}} Local | ${{cloudCount}} Cloud`;
        document.getElementById("kpiTopScore").textContent = topScore >= 0 ? `${{topScore.toFixed(1)}}%` : "-%";
        document.getElementById("kpiTopModelName").textContent = topModel;
        document.getElementById("kpiAvgScore").textContent = avgAcc >= 0 ? `${{avgAcc.toFixed(1)}}%` : "-%";
        document.getElementById("kpiAvgSubtext").textContent = accCount > 0 ? `Avg across ${{accCount}} model${{accCount > 1 ? 's' : ''}}` : "Across evaluated models";
        document.getElementById("kpiSampleCount").textContent = sampleCount;
      }}

      // 6. Filter & Sort Helper
      function getFilteredScorecards() {{
        return scorecards.filter(sc => {{
          const env = sc.environment || {{}};
          const isBaseline = (sc.model_name && sc.model_name.toLowerCase().includes("baseline")) || env.execution_type === 'baseline';
          const isLocal = env.execution_type === 'local' || env.is_local_inference;

          // Exclude ground truth baseline from standard leaderboard views by default
          if (activeFilter === 'all' && isBaseline) return false;
          if (activeFilter === 'local' && (!isLocal || isBaseline)) return false;
          if (activeFilter === 'cloud' && (isLocal || isBaseline)) return false;
          if (activeFilter === 'baseline' && !isBaseline) return false;

          if (searchQuery) {{
            const name = (sc.model_name || "").toLowerCase();
            const prov = (env.provider || "").toLowerCase();
            const gpu = (env.gpu || "").toLowerCase();
            const os = (env.os || "").toLowerCase();
            if (!name.includes(searchQuery) && !prov.includes(searchQuery) && !gpu.includes(searchQuery) && !os.includes(searchQuery)) {{
              return false;
            }}
          }}

          return true;
        }}).sort((a, b) => {{
          let valA = 0;
          let valB = 0;

          const statsA = getModelStats(a, activeFormat, activeDiff);
          const statsB = getModelStats(b, activeFormat, activeDiff);

          if (sortCol === 'name') {{
            return sortDir === 'asc' ? (a.model_name || '').localeCompare(b.model_name || '') : (b.model_name || '').localeCompare(a.model_name || '');
          }} else if (sortCol === 'overall' || sortCol === 'rank') {{
            valA = (statsA.has_data && statsA.overall_accuracy_pct !== null) ? statsA.overall_accuracy_pct : -1;
            valB = (statsB.has_data && statsB.overall_accuracy_pct !== null) ? statsB.overall_accuracy_pct : -1;
          }} else if (sortCol.startsWith('task_')) {{
            const tKey = sortCol.replace('task_', '');
            valA = (statsA.task_breakdown && statsA.task_breakdown[tKey]) ? statsA.task_breakdown[tKey].accuracy_pct : -1;
            valB = (statsB.task_breakdown && statsB.task_breakdown[tKey]) ? statsB.task_breakdown[tKey].accuracy_pct : -1;
          }} else if (sortCol.startsWith('diff_')) {{
            const dKey = sortCol.replace('diff_', '');
            valA = (statsA.difficulty_breakdown && statsA.difficulty_breakdown[dKey]) ? statsA.difficulty_breakdown[dKey].accuracy_pct : -1;
            valB = (statsB.difficulty_breakdown && statsB.difficulty_breakdown[dKey]) ? statsB.difficulty_breakdown[dKey].accuracy_pct : -1;
          }} else if (sortCol === 'track_mcq') {{
            valA = (statsA.track_breakdown && statsA.track_breakdown.mcq) ? statsA.track_breakdown.mcq.accuracy_pct : -1;
            valB = (statsB.track_breakdown && statsB.track_breakdown.mcq) ? statsB.track_breakdown.mcq.accuracy_pct : -1;
          }} else if (sortCol === 'track_free_form') {{
            valA = (statsA.track_breakdown && statsA.track_breakdown.free_form) ? statsA.track_breakdown.free_form.accuracy_pct : -1;
            valB = (statsB.track_breakdown && statsB.track_breakdown.free_form) ? statsB.track_breakdown.free_form.accuracy_pct : -1;
          }} else if (sortCol === 'track_exact') {{
            valA = (statsA.track_breakdown) ? statsA.track_breakdown.exact_matches : -1;
            valB = (statsB.track_breakdown) ? statsB.track_breakdown.exact_matches : -1;
          }} else if (sortCol === 'track_symbolic') {{
            valA = (statsA.track_breakdown) ? statsA.track_breakdown.symbolic_matches : -1;
            valB = (statsB.track_breakdown) ? statsB.track_breakdown.symbolic_matches : -1;
          }} else if (sortCol === 'speed') {{
            valA = (a.metrics && a.metrics.avg_tokens_per_second) ? a.metrics.avg_tokens_per_second : 0;
            valB = (b.metrics && b.metrics.avg_tokens_per_second) ? b.metrics.avg_tokens_per_second : 0;
          }} else if (sortCol === 'latency') {{
            valA = (a.metrics && a.metrics.avg_latency_seconds) ? a.metrics.avg_latency_seconds : 9999;
            valB = (b.metrics && b.metrics.avg_latency_seconds) ? b.metrics.avg_latency_seconds : 9999;
            return sortDir === 'asc' ? valA - valB : valB - valA;
          }}

          return sortDir === 'asc' ? valA - valB : valB - valA;
        }});
      }}

      // 7. Render Leaderboard Table with Multiple Views
      function renderLeaderboardTable() {{
        const thead = document.getElementById("leaderboardHead");
        const tbody = document.getElementById("leaderboardBody");
        thead.innerHTML = "";
        tbody.innerHTML = "";

        // Build Table Header based on activeTable
        if (activeTable === 'subdiscipline') {{
          thead.innerHTML = `
            <tr>
              <th data-sort="rank" style="width:44px;">Rank</th>
              <th data-sort="name">Model & Execution Target</th>
              <th data-sort="overall">Overall Score</th>
              <th data-sort="task_constants">1. Constants & SI</th>
              <th data-sort="task_dimensions">2. Dimensions</th>
              <th data-sort="task_conversions">3. Conversions</th>
              <th data-sort="task_homogeneity">4. Homogeneity</th>
              <th data-sort="task_conventions">5. SI Rules</th>
              <th data-sort="task_uncertainty">6. GUM Uncertainty</th>
              <th data-sort="speed">Speed</th>
              <th data-sort="latency">Latency</th>
              <th style="width:85px;">Actions</th>
            </tr>
          `;
        }} else if (activeTable === 'difficulty') {{
          thead.innerHTML = `
            <tr>
              <th data-sort="rank" style="width:44px;">Rank</th>
              <th data-sort="name">Model & Execution Target</th>
              <th data-sort="overall">Overall Score</th>
              <th data-sort="diff_introductory">🟢 Introductory Tier</th>
              <th data-sort="diff_intermediate">🟡 Intermediate Tier</th>
              <th data-sort="diff_advanced">🔴 Advanced Tier</th>
              <th data-sort="speed">Speed</th>
              <th data-sort="latency">Latency</th>
              <th style="width:85px;">Actions</th>
            </tr>
          `;
        }} else if (activeTable === 'tracks') {{
          thead.innerHTML = `
            <tr>
              <th data-sort="rank" style="width:44px;">Rank</th>
              <th data-sort="name">Model & Execution Target</th>
              <th data-sort="overall">Overall Score</th>
              <th data-sort="track_mcq">🔠 Track A (MCQ Accuracy)</th>
              <th data-sort="track_free_form">✍️ Track B (Free-Form Accuracy)</th>
              <th data-sort="track_exact">🎯 Exact Value Matches</th>
              <th data-sort="track_symbolic">📐 Symbolic (SymPy/Pint) Passes</th>
              <th data-sort="speed">Speed</th>
              <th data-sort="latency">Latency</th>
              <th style="width:85px;">Actions</th>
            </tr>
          `;
        }}

        // Bind sorting headers
        thead.querySelectorAll("th[data-sort]").forEach(th => {{
          const col = th.dataset.sort;
          if (col === sortCol) {{
            th.classList.add(sortDir === 'asc' ? 'sorted-asc' : 'sorted-desc');
          }}
          th.addEventListener("click", () => {{
            if (sortCol === col) {{
              sortDir = sortDir === 'asc' ? 'desc' : 'asc';
            }} else {{
              sortCol = col;
              sortDir = col === 'name' ? 'asc' : 'desc';
            }}
            renderLeaderboardTable();
          }});
        }});

        const list = getFilteredScorecards();
        const totalCols = activeTable === 'subdiscipline' ? 12 : (activeTable === 'difficulty' ? 9 : 10);

        if (list.length === 0) {{
          tbody.innerHTML = `<tr><td colspan="${{totalCols}}" style="text-align:center; padding:2.5rem; color:var(--text-muted);">No scorecards match the selected filter criteria.</td></tr>`;
          return;
        }}

        list.forEach((sc, idx) => {{
          const isBaseline = (sc.model_name && sc.model_name.toLowerCase().includes("baseline")) || (sc.environment && sc.environment.execution_type === 'baseline');
          const env = sc.environment || {{}};
          const isLocal = env.execution_type === 'local' || env.is_local_inference;
          const tr = document.createElement("tr");
          if (isBaseline) tr.className = "baseline-row";

          const stats = getModelStats(sc, activeFormat, activeDiff);

          // Rank Badge
          const rank = idx + 1;
          const rankClass = rank === 1 ? 'rank-1' : (rank === 2 ? 'rank-2' : (rank === 3 ? 'rank-3' : 'rank-other'));

          // Target Badge
          let targetBadge = isBaseline
            ? `<span class="tag tag-baseline">🎯 Baseline</span>`
            : (isLocal ? `<span class="tag tag-local">💻 Local</span>` : `<span class="tag tag-cloud">☁️ Cloud</span>`);

          let gpuBadge = env.gpu ? `<span style="font-size:0.7rem; color:var(--text-muted); display:block;">⚡ ${{escapeHtml(env.gpu)}}</span>` : '';

          let overallHtml = '';
          if (stats.has_data && stats.overall_accuracy_pct !== null) {{
            const overall = stats.overall_accuracy_pct;
            const overallColor = overall >= 70 ? '#34d399' : (overall >= 30 ? '#fbbf24' : '#f87171');
            overallHtml = `
              <div class="score-cell">
                <span style="font-size:1.05rem; font-weight:800; color:${{overallColor}};">${{overall.toFixed(1)}}%</span>
                <span style="font-size:0.68rem; color:var(--text-muted);">${{stats.passed}} / ${{stats.total}} passed</span>
                <div class="progress-bar-wrap" style="height:5px;">
                  <div class="progress-bar-fill" style="width:${{Math.min(100, overall)}}%; background:${{overallColor}};"></div>
                </div>
              </div>
            `;
          }} else {{
            overallHtml = `
              <div class="score-cell">
                <span style="font-size:0.95rem; font-weight:700; color:var(--text-muted);">N/A</span>
                <span style="font-size:0.68rem; color:var(--text-muted);">No samples</span>
              </div>
            `;
          }}

          function renderTaskCell(taskKey) {{
            if (!stats.has_data) return `<td>-</td>`;
            const tb = stats.task_breakdown ? stats.task_breakdown[taskKey] : null;
            if (!tb || tb.total === 0) return `<td><span style="color:var(--text-muted); font-size:0.75rem;">-</span></td>`;
            const acc = tb.accuracy_pct !== undefined ? tb.accuracy_pct : 0;
            const barCol = acc >= 70 ? '#34d399' : (acc >= 30 ? '#fbbf24' : '#f87171');
            return `
              <td>
                <div class="score-cell">
                  <span><strong>${{acc.toFixed(1)}}%</strong> <span style="font-size:0.68rem; color:var(--text-muted);">(${{tb.passed}}/${{tb.total}})</span></span>
                  <div class="progress-bar-wrap">
                    <div class="progress-bar-fill" style="width:${{Math.min(100, acc)}}%; background:${{barCol}};"></div>
                  </div>
                </div>
              </td>
            `;
          }}

          function renderDiffCell(diffKey) {{
            if (!stats.has_data) return `<td>-</td>`;
            const db = stats.difficulty_breakdown ? stats.difficulty_breakdown[diffKey] : null;
            if (!db || db.total === 0) return `<td><span style="color:var(--text-muted); font-size:0.75rem;">-</span></td>`;
            const acc = db.accuracy_pct !== undefined ? db.accuracy_pct : 0;
            const barCol = acc >= 70 ? '#34d399' : (acc >= 30 ? '#fbbf24' : '#f87171');
            return `
              <td>
                <div class="score-cell">
                  <span><strong>${{acc.toFixed(1)}}%</strong> <span style="font-size:0.68rem; color:var(--text-muted);">(${{db.passed}}/${{db.total}})</span></span>
                  <div class="progress-bar-wrap">
                    <div class="progress-bar-fill" style="width:${{Math.min(100, acc)}}%; background:${{barCol}};"></div>
                  </div>
                </div>
              </td>
            `;
          }}

          function renderTrackBreakdownCells() {{
            const trb = stats.track_breakdown || {{}};
            const mcq = trb.mcq || {{ total: 0, passed: 0, accuracy_pct: 0 }};
            const free = trb.free_form || {{ total: 0, passed: 0, accuracy_pct: 0 }};

            const mcqCol = mcq.accuracy_pct >= 70 ? '#34d399' : (mcq.accuracy_pct >= 30 ? '#fbbf24' : '#f87171');
            const freeCol = free.accuracy_pct >= 70 ? '#34d399' : (free.accuracy_pct >= 30 ? '#fbbf24' : '#f87171');

            const mcqHtml = mcq.total > 0 ? `
              <td>
                <div class="score-cell">
                  <span><strong>${{mcq.accuracy_pct.toFixed(1)}}%</strong> <span style="font-size:0.68rem; color:var(--text-muted);">(${{mcq.passed}}/${{mcq.total}})</span></span>
                  <div class="progress-bar-wrap"><div class="progress-bar-fill" style="width:${{mcq.accuracy_pct}}%; background:${{mcqCol}};"></div></div>
                </div>
              </td>
            ` : `<td><span style="color:var(--text-muted); font-size:0.75rem;">-</span></td>`;

            const freeHtml = free.total > 0 ? `
              <td>
                <div class="score-cell">
                  <span><strong>${{free.accuracy_pct.toFixed(1)}}%</strong> <span style="font-size:0.68rem; color:var(--text-muted);">(${{free.passed}}/${{free.total}})</span></span>
                  <div class="progress-bar-wrap"><div class="progress-bar-fill" style="width:${{free.accuracy_pct}}%; background:${{freeCol}};"></div></div>
                </div>
              </td>
            ` : `<td><span style="color:var(--text-muted); font-size:0.75rem;">-</span></td>`;

            const exactHtml = `<td><strong style="color:#38bdf8;">${{trb.exact_matches || 0}}</strong> <span style="font-size:0.7rem; color:var(--text-muted);">verified</span></td>`;
            const symbHtml = `<td><strong style="color:#a855f7;">${{trb.symbolic_matches || 0}}</strong> <span style="font-size:0.7rem; color:var(--text-muted);">symbolic</span></td>`;

            return `${{mcqHtml}}${{freeHtml}}${{exactHtml}}${{symbHtml}}`;
          }}

          const m = sc.metrics || {{}};
          const speedStr = m.avg_tokens_per_second ? `<strong style="color:#38bdf8;">${{m.avg_tokens_per_second.toFixed(1)}} tok/s</strong>` : '-';
          const latStr = m.avg_latency_seconds ? `${{m.avg_latency_seconds.toFixed(2)}}s` : '-';

          let customCells = '';
          if (activeTable === 'subdiscipline') {{
            customCells = `
              ${{renderTaskCell('constants')}}
              ${{renderTaskCell('dimensions')}}
              ${{renderTaskCell('conversions')}}
              ${{renderTaskCell('homogeneity')}}
              ${{renderTaskCell('conventions')}}
              ${{renderTaskCell('uncertainty')}}
            `;
          }} else if (activeTable === 'difficulty') {{
            customCells = `
              ${{renderDiffCell('introductory')}}
              ${{renderDiffCell('intermediate')}}
              ${{renderDiffCell('advanced')}}
            `;
          }} else if (activeTable === 'tracks') {{
            customCells = renderTrackBreakdownCells();
          }}

          tr.innerHTML = `
            <td><span class="rank-badge ${{rankClass}}">${{rank}}</span></td>
            <td>
              <div style="font-weight:700; color:var(--text-primary); font-size:0.875rem;">${{escapeHtml(sc.model_name || "Unknown")}}</div>
              <div style="display:flex; align-items:center; gap:0.4rem; margin-top:0.2rem; flex-wrap:wrap;">
                ${{targetBadge}}
                <span style="font-size:0.7rem; color:var(--text-secondary);">${{escapeHtml(env.provider || "")}}</span>
              </div>
              ${{gpuBadge}}
            </td>
            <td>${{overallHtml}}</td>
            ${{customCells}}
            <td>${{speedStr}}</td>
            <td>${{latStr}}</td>
            <td>
              <button class="btn" style="padding:0.25rem 0.5rem; font-size:0.7rem;" onclick="window.inspectModelInArena('${{escapeHtml(sc.model_name)}}')">⚔️ Compare</button>
            </td>
          `;
          tbody.appendChild(tr);
        }});
      }}

      // 8. Radar View
      let selectedRadarModels = new Set();
      function renderRadarView() {{
        const toggles = document.getElementById("radarModelToggles");
        toggles.innerHTML = "";

        if (scorecards.length === 0) return;

        // Filter selectable models (exclude baseline by default)
        const radarCandidateModels = scorecards.filter(sc => {{
          const isBaseline = (sc.model_name && sc.model_name.toLowerCase().includes("baseline")) || (sc.environment && sc.environment.execution_type === 'baseline');
          return activeFilter === 'baseline' ? isBaseline : !isBaseline;
        }});

        if (selectedRadarModels.size === 0 && radarCandidateModels.length > 0) {{
          radarCandidateModels.slice(0, 3).forEach(s => selectedRadarModels.add(s.model_name));
        }}

        radarCandidateModels.forEach((sc, i) => {{
          const chip = document.createElement("button");
          const isSel = selectedRadarModels.has(sc.model_name);
          const col = COLORS[i % COLORS.length];
          chip.className = `model-chip ${{isSel ? 'selected' : ''}}`;
          chip.style.color = col;
          chip.innerHTML = `<span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:${{col}};"></span> ${{escapeHtml(sc.model_name)}}`;
          chip.addEventListener("click", () => {{
            if (selectedRadarModels.has(sc.model_name)) {{
              if (selectedRadarModels.size > 1) selectedRadarModels.delete(sc.model_name);
            }} else {{
              selectedRadarModels.add(sc.model_name);
            }}
            renderRadarView();
          }});
          toggles.appendChild(chip);
        }});

        drawRadarChart();
        renderSubDisciplineBars();
      }}

      function drawRadarChart() {{
        const svg = document.getElementById("radarSvg");
        svg.innerHTML = "";

        const width = 400;
        const height = 360;
        const cx = width / 2;
        const cy = height / 2 + 10;
        const radius = 120;

        const tasks = [
          {{ key: "constants", label: "Constants & SI 2019" }},
          {{ key: "dimensions", label: "Dimensions" }},
          {{ key: "conversions", label: "Conversions" }},
          {{ key: "homogeneity", label: "Homogeneity" }},
          {{ key: "conventions", label: "SI Rules" }},
          {{ key: "uncertainty", label: "GUM Uncertainty" }}
        ];

        const numAxes = tasks.length;
        const angleStep = (Math.PI * 2) / numAxes;

        // Draw web rings
        [0.2, 0.4, 0.6, 0.8, 1.0].forEach(level => {{
          const r = radius * level;
          const points = [];
          for (let i = 0; i < numAxes; i++) {{
            const angle = i * angleStep - Math.PI / 2;
            points.push(`${{cx + r * Math.cos(angle)}},${{cy + r * Math.sin(angle)}}`);
          }}
          const polygon = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
          polygon.setAttribute("points", points.join(" "));
          polygon.setAttribute("fill", level === 1.0 ? "rgba(255,255,255,0.02)" : "none");
          polygon.setAttribute("stroke", "rgba(255,255,255,0.1)");
          polygon.setAttribute("stroke-width", "1");
          svg.appendChild(polygon);

          // Level text
          const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
          text.setAttribute("x", cx + 5);
          text.setAttribute("y", cy - r + 10);
          text.setAttribute("fill", "rgba(255,255,255,0.3)");
          text.setAttribute("font-size", "9");
          text.textContent = `${{Math.round(level * 100)}}%`;
          svg.appendChild(text);
        }});

        // Draw Axes & Labels
        tasks.forEach((t, i) => {{
          const angle = i * angleStep - Math.PI / 2;
          const x2 = cx + radius * Math.cos(angle);
          const y2 = cy + radius * Math.sin(angle);

          const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
          line.setAttribute("x1", cx);
          line.setAttribute("y1", cy);
          line.setAttribute("x2", x2);
          line.setAttribute("y2", y2);
          line.setAttribute("stroke", "rgba(255,255,255,0.15)");
          svg.appendChild(line);

          // Label
          const lx = cx + (radius + 22) * Math.cos(angle);
          const ly = cy + (radius + 22) * Math.sin(angle);
          const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
          label.setAttribute("x", lx);
          label.setAttribute("y", ly + 4);
          label.setAttribute("fill", "#94a3b8");
          label.setAttribute("font-size", "10");
          label.setAttribute("font-weight", "600");
          label.setAttribute("text-anchor", Math.cos(angle) > 0.3 ? "start" : (Math.cos(angle) < -0.3 ? "end" : "middle"));
          label.textContent = t.label;
          svg.appendChild(label);
        }});

        // Draw Polygons for selected models
        scorecards.forEach((sc, idx) => {{
          if (!selectedRadarModels.has(sc.model_name)) return;
          const stats = getModelStats(sc, activeFormat, activeDiff);
          if (!stats.has_data) return;

          const color = COLORS[idx % COLORS.length];

          const points = [];
          tasks.forEach((t, i) => {{
            const tb = stats.task_breakdown ? stats.task_breakdown[t.key] : null;
            const acc = tb ? (tb.accuracy_pct || 0) : 0;
            const r = radius * (Math.min(100, Math.max(0, acc)) / 100);
            const angle = i * angleStep - Math.PI / 2;
            points.push(`${{cx + r * Math.cos(angle)}},${{cy + r * Math.sin(angle)}}`);
          }});

          const poly = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
          poly.setAttribute("points", points.join(" "));
          poly.setAttribute("fill", color);
          poly.setAttribute("fill-opacity", "0.2");
          poly.setAttribute("stroke", color);
          poly.setAttribute("stroke-width", "2.5");
          svg.appendChild(poly);

          // Draw vertex points
          tasks.forEach((t, i) => {{
            const tb = stats.task_breakdown ? stats.task_breakdown[t.key] : null;
            const acc = tb ? (tb.accuracy_pct || 0) : 0;
            const r = radius * (Math.min(100, Math.max(0, acc)) / 100);
            const angle = i * angleStep - Math.PI / 2;
            const vx = cx + r * Math.cos(angle);
            const vy = cy + r * Math.sin(angle);

            const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
            dot.setAttribute("cx", vx);
            dot.setAttribute("cy", vy);
            dot.setAttribute("r", "3.5");
            dot.setAttribute("fill", color);
            svg.appendChild(dot);
          }});
        }});
      }}

      function renderSubDisciplineBars() {{
        const container = document.getElementById("taskBarCharts");
        container.innerHTML = "";

        const taskNames = [
          {{ key: "constants", label: "1. Physical Constants & SI 2019" }},
          {{ key: "dimensions", label: "2. Dimensional Decomposition" }},
          {{ key: "conversions", label: "3. Unit Conversions & Offsets" }},
          {{ key: "homogeneity", label: "4. Error Detection & Homogeneity" }},
          {{ key: "conventions", label: "5. SI Rules & Typography" }},
          {{ key: "uncertainty", label: "6. Metrological Uncertainty (GUM)" }}
        ];

        taskNames.forEach(t => {{
          const box = document.createElement("div");
          box.style.display = "flex";
          box.style.flexDirection = "column";
          box.style.gap = "0.25rem";

          let rowsHtml = "";
          scorecards.forEach((sc, i) => {{
            if (!selectedRadarModels.has(sc.model_name)) return;
            const stats = getModelStats(sc, activeFormat, activeDiff);
            if (!stats.has_data) return;

            const col = COLORS[i % COLORS.length];
            const tb = stats.task_breakdown ? stats.task_breakdown[t.key] : null;
            const acc = tb ? (tb.accuracy_pct || 0) : 0;

            rowsHtml += `
              <div style="display:flex; align-items:center; gap:0.5rem; font-size:0.75rem;">
                <span style="width:110px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:var(--text-secondary);" title="${{escapeHtml(sc.model_name)}}">${{escapeHtml(sc.model_name)}}</span>
                <div style="flex:1; height:6px; background:rgba(255,255,255,0.08); border-radius:3px; overflow:hidden;">
                  <div style="width:${{acc}}%; height:100%; background:${{col}}; border-radius:3px;"></div>
                </div>
                <span style="width:42px; text-align:right; font-weight:700; color:${{col}};">${{acc.toFixed(1)}}%</span>
              </div>
            `;
          }});

          box.innerHTML = `
            <div style="font-size:0.75rem; font-weight:700; color:var(--text-muted);">${{t.label}}</div>
            ${{rowsHtml}}
          `;
          container.appendChild(box);
        }});
      }}

      // 9. Model Comparison Arena
      function renderArenaView() {{
        const selA = document.getElementById("arenaModelA");
        const selB = document.getElementById("arenaModelB");
        const diffSel = document.getElementById("arenaDiffFilter");

        const curA = selA.value;
        const curB = selB.value;

        selA.innerHTML = "";
        selB.innerHTML = "";

        scorecards.forEach((sc) => {{
          const optA = document.createElement("option");
          optA.value = sc.model_name;
          optA.textContent = sc.model_name;
          selA.appendChild(optA);

          const optB = document.createElement("option");
          optB.value = sc.model_name;
          optB.textContent = sc.model_name;
          selB.appendChild(optB);
        }});

        if (scorecards.length >= 2) {{
          selA.value = curA || scorecards[0].model_name;
          selB.value = curB || scorecards[1].model_name;
        }} else if (scorecards.length === 1) {{
          selA.value = scorecards[0].model_name;
          selB.value = scorecards[0].model_name;
        }}

        selA.onchange = updateArenaComparison;
        selB.onchange = updateArenaComparison;
        diffSel.onchange = updateArenaComparison;

        updateArenaComparison();
      }}

      window.inspectModelInArena = function(modelName) {{
        document.querySelectorAll(".nav-tab").forEach(t => t.classList.remove("active"));
        document.querySelector(".nav-tab[data-tab='arena']").classList.add("active");
        document.querySelectorAll(".view-panel").forEach(p => p.classList.remove("active"));
        document.getElementById("viewArena").classList.add("active");

        const selB = document.getElementById("arenaModelB");
        selB.value = modelName;
        updateArenaComparison();
      }};

      function updateArenaComparison() {{
        const nameA = document.getElementById("arenaModelA").value;
        const nameB = document.getElementById("arenaModelB").value;
        const diffMode = document.getElementById("arenaDiffFilter").value;

        const scA = scorecards.find(s => s.model_name === nameA);
        const scB = scorecards.find(s => s.model_name === nameB);

        const kpiGrid = document.getElementById("arenaKpiGrid");
        if (!scA || !scB) {{
          kpiGrid.innerHTML = `<div style="color:var(--text-muted);">Please select two valid models to compare.</div>`;
          return;
        }}

        const statsA = getModelStats(scA, activeFormat, activeDiff);
        const statsB = getModelStats(scB, activeFormat, activeDiff);

        const accA = (statsA.has_data && statsA.overall_accuracy_pct !== null) ? statsA.overall_accuracy_pct : 0;
        const accB = (statsB.has_data && statsB.overall_accuracy_pct !== null) ? statsB.overall_accuracy_pct : 0;
        const delta = accB - accA;
        const deltaCol = delta > 0 ? '#34d399' : (delta < 0 ? '#f43f5e' : '#94a3b8');

        const fmtDiffLabel = (activeFormat !== 'all' || activeDiff !== 'all') ? ` (${{activeFormat.toUpperCase()}}${{activeDiff !== 'all' ? ' / ' + activeDiff : ''}})` : '';

        kpiGrid.innerHTML = `
          <div class="kpi-card">
            <div class="kpi-label">${{escapeHtml(scA.model_name)}} Score${{fmtDiffLabel}}</div>
            <div class="kpi-value">${{statsA.has_data ? accA.toFixed(1) + '%' : 'N/A'}}</div>
            <div class="kpi-subtext">${{statsA.passed}} / ${{statsA.total}} passed</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">${{escapeHtml(scB.model_name)}} Score${{fmtDiffLabel}}</div>
            <div class="kpi-value">${{statsB.has_data ? accB.toFixed(1) + '%' : 'N/A'}}</div>
            <div class="kpi-subtext">${{statsB.passed}} / ${{statsB.total}} passed</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Relative Delta (B - A)</div>
            <div class="kpi-value" style="color:${{deltaCol}};">${{delta >= 0 ? '+' : ''}}${{delta.toFixed(1)}}%</div>
            <div class="kpi-subtext">${{delta > 0 ? 'Model B leads' : (delta < 0 ? 'Model A leads' : 'Tie')}}</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Speed Comparison</div>
            <div class="kpi-value" style="font-size:1.25rem;">
              ${{(scA.metrics && scA.metrics.avg_tokens_per_second) ? scA.metrics.avg_tokens_per_second.toFixed(1) : '-'}} vs ${{ (scB.metrics && scB.metrics.avg_tokens_per_second) ? scB.metrics.avg_tokens_per_second.toFixed(1) : '-'}}
            </div>
            <div class="kpi-subtext">Tokens / second</div>
          </div>
        `;

        renderDifferentialQuestions(scA, scB, diffMode);
      }}

      function renderDifferentialQuestions(scA, scB, diffMode) {{
        const listEl = document.getElementById("arenaDiffList");
        listEl.innerHTML = "";

        const mapA = new Map((scA.detailed_results || []).map(r => [r.id, r]));
        const mapB = new Map((scB.detailed_results || []).map(r => [r.id, r]));

        const allIds = Array.from(new Set([...mapA.keys(), ...mapB.keys()]));

        let count = 0;
        allIds.forEach(id => {{
          const resA = mapA.get(id);
          const resB = mapB.get(id);

          // Filter by active format
          if (activeFormat === 'mcq') {{
            const fA = resA ? (resA.format || 'mcq') : '';
            const fB = resB ? (resB.format || 'mcq') : '';
            if ((resA && fA !== 'mcq') || (resB && fB !== 'mcq')) return;
          }} else if (activeFormat === 'free_form') {{
            const fA = resA ? (resA.format || '') : '';
            const fB = resB ? (resB.format || '') : '';
            if ((resA && fA === 'mcq') || (resB && fB === 'mcq')) return;
          }}

          // Filter by active difficulty
          if (activeDiff !== 'all') {{
            const dA = resA ? (resA.difficulty || 'intermediate').toLowerCase() : '';
            const dB = resB ? (resB.difficulty || 'intermediate').toLowerCase() : '';
            if ((resA && dA !== activeDiff.toLowerCase()) || (resB && dB !== activeDiff.toLowerCase())) return;
          }}

          const passA = resA && resA.grade && resA.grade.passed;
          const passB = resB && resB.grade && resB.grade.passed;

          if (diffMode === 'a_wins' && (!passA || passB)) return;
          if (diffMode === 'b_wins' && (!passB || passA)) return;
          if (diffMode === 'both_failed' && (passA || passB)) return;
          if (diffMode === 'both_passed' && (!passA || !passB)) return;
          if (diffMode === 'all' && (passA === passB)) return;

          count++;
          const sample = benchmarkSamples.find(s => s.id === id) || {{}};
          const questionText = sample.question || (resA && resA.grade && resA.grade.expected_key ? `Item ID: ${{id}}` : `Question ${{id}}`);

          const card = document.createElement("div");
          card.className = "diff-item";

          function renderModelOutcome(res, modelName) {{
            if (!res) return `<div class="resp-box"><em>Not evaluated</em></div>`;
            const grade = res.grade || {{}};
            const isPass = grade.passed;
            const predKey = grade.predicted_key || grade.predicted_option || "None";
            const exp = grade.predicted_explanation ? `<div style="margin-top:0.35rem; color:var(--text-secondary); font-size:0.75rem;"><strong>Rationale:</strong> ${{escapeHtml(grade.predicted_explanation)}}</div>` : '';
            const raw = grade.predicted_raw ? `<div style="margin-top:0.25rem; font-family:var(--font-mono); font-size:0.7rem; color:var(--text-muted); max-height:80px; overflow-y:auto;">${{escapeHtml(grade.predicted_raw)}}</div>` : '';

            return `
              <div class="resp-box ${{isPass ? 'pass' : 'fail'}}">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                  <strong>${{escapeHtml(modelName)}}</strong>
                  <span class="tag ${{isPass ? 'tag-local' : 'tag-baseline'}}">${{isPass ? '✓ PASSED' : '✗ MISSED'}}</span>
                </div>
                <div>Predicted: <strong>Option ${{escapeHtml(predKey)}}</strong></div>
                ${{exp}}
                ${{raw}}
              </div>
            `;
          }}

          card.innerHTML = `
            <div class="diff-item-header">
              <span><strong>ID:</strong> ${{escapeHtml(id)}} | <strong>Task:</strong> ${{escapeHtml(resA ? resA.task : (resB ? resB.task : 'general'))}} | <strong>Expected:</strong> Option ${{escapeHtml((resA && resA.grade && (resA.grade.expected_key || resA.grade.expected_option)) || (resB && resB.grade && (resB.grade.expected_key || resB.grade.expected_option)) || '-')}}</span>
            </div>
            <div style="font-size:0.875rem; color:var(--text-primary); font-weight:600;">
              ${{formatMathText(questionText)}}
            </div>
            <div class="diff-responses">
              ${{renderModelOutcome(resA, scA.model_name)}}
              ${{renderModelOutcome(resB, scB.model_name)}}
            </div>
          `;
          listEl.appendChild(card);
        }});

        if (count === 0) {{
          listEl.innerHTML = `<div style="text-align:center; padding:2rem; color:var(--text-muted);">No questions match the differential filter.</div>`;
        }}
      }}

      // 10. Hardware & Environment Table
      function renderHardwareTable() {{
        const tbody = document.getElementById("hardwareBody");
        tbody.innerHTML = "";

        const list = getFilteredScorecards();
        if (list.length === 0) {{
          tbody.innerHTML = `<tr><td colspan="8" style="text-align:center; padding:2rem; color:var(--text-muted);">No profiles match the filter.</td></tr>`;
          return;
        }}

        list.forEach(sc => {{
          const env = sc.environment || {{}};
          const isBaseline = (sc.model_name && sc.model_name.toLowerCase().includes("baseline")) || env.execution_type === 'baseline';
          const isLocal = env.execution_type === 'local' || env.is_local_inference;

          const targetBadge = isBaseline
            ? `<span class="tag tag-baseline">🎯 Baseline</span>`
            : (isLocal ? `<span class="tag tag-local">💻 Local Host</span>` : `<span class="tag tag-cloud">☁️ Cloud API</span>`);

          const tr = document.createElement("tr");
          tr.innerHTML = `
            <td>
              <strong style="color:var(--text-primary);">${{escapeHtml(sc.model_name || "Unknown")}}</strong>
              <div style="font-size:0.7rem; color:var(--text-muted); font-family:var(--font-mono);">${{escapeHtml(sc.benchmark_version ? sc.benchmark_version.substring(0, 20) + '...' : '-')}}</div>
            </td>
            <td>${{targetBadge}} <span style="font-size:0.75rem; color:var(--text-secondary); display:block;">${{escapeHtml(env.provider || "")}}</span></td>
            <td>${{escapeHtml(env.os || "Unknown")}}</td>
            <td>${{env.cpu_count ? `${{env.cpu_count}} cores` : '-'}}</td>
            <td>${{env.total_memory_gb ? `${{env.total_memory_gb}} GB` : '-'}}</td>
            <td>
              <strong>${{escapeHtml(env.gpu || "None / Host CPU")}}</strong>
              ${{env.gpu_memory_gb ? `<span style="font-size:0.7rem; color:var(--text-muted); display:block;">${{env.gpu_memory_gb}} GB VRAM</span>` : ''}}
            </td>
            <td>Python ${{escapeHtml(env.python_version || "-")}}</td>
            <td style="font-size:0.75rem; color:var(--text-secondary);">${{sc.timestamp ? new Date(sc.timestamp).toLocaleString() : '-'}}</td>
          `;
          tbody.appendChild(tr);
        }});
      }}

      // 11. Export Markdown Table
      function setupExport() {{
        document.getElementById("btnExportMarkdown").addEventListener("click", () => {{
          const list = getFilteredScorecards();
          if (list.length === 0) return;

          const fmtLabel = activeFormat === 'all' ? 'All Formats' : (activeFormat === 'mcq' ? 'MCQ Track A' : 'Free-Form Track B');
          const diffLabel = activeDiff === 'all' ? 'All Difficulties' : activeDiff;

          let md = `### DRUM Metrology Benchmark Leaderboard (${{fmtLabel}} | ${{diffLabel}})\n\n`;

          if (activeTable === 'subdiscipline') {{
            md += `| Rank | Model | Target | Overall Score | Constants | Dimensions | Conversions | Homogeneity | Conventions | Uncertainty | Speed | Latency |\n`;
            md += `|---|---|---|---|---|---|---|---|---|---|---|---|\n`;

            list.forEach((sc, i) => {{
              const env = sc.environment || {{}};
              const target = env.execution_type === 'cloud' ? 'Cloud API' : (sc.model_name.includes('baseline') ? 'Baseline' : 'Local');
              const stats = getModelStats(sc, activeFormat, activeDiff);
              const overall = stats.has_data && stats.overall_accuracy_pct !== null ? stats.overall_accuracy_pct.toFixed(1) + '%' : 'N/A';

              function getAcc(k) {{
                return (stats.task_breakdown && stats.task_breakdown[k] && stats.task_breakdown[k].total > 0)
                  ? stats.task_breakdown[k].accuracy_pct.toFixed(1) + '%'
                  : '-';
              }}

              const spd = sc.metrics && sc.metrics.avg_tokens_per_second ? `${{sc.metrics.avg_tokens_per_second.toFixed(1)}} tok/s` : '-';
              const lat = sc.metrics && sc.metrics.avg_latency_seconds ? `${{sc.metrics.avg_latency_seconds.toFixed(2)}}s` : '-';

              md += `| ${{i + 1}} | **${{sc.model_name}}** | ${{target}} | **${{overall}}** | ${{getAcc('constants')}} | ${{getAcc('dimensions')}} | ${{getAcc('conversions')}} | ${{getAcc('homogeneity')}} | ${{getAcc('conventions')}} | ${{getAcc('uncertainty')}} | ${{spd}} | ${{lat}} |\n`;
            }});
          }} else if (activeTable === 'difficulty') {{
            md += `| Rank | Model | Target | Overall Score | Introductory | Intermediate | Advanced | Speed | Latency |\n`;
            md += `|---|---|---|---|---|---|---|---|---|\n`;

            list.forEach((sc, i) => {{
              const env = sc.environment || {{}};
              const target = env.execution_type === 'cloud' ? 'Cloud API' : (sc.model_name.includes('baseline') ? 'Baseline' : 'Local');
              const stats = getModelStats(sc, activeFormat, activeDiff);
              const overall = stats.has_data && stats.overall_accuracy_pct !== null ? stats.overall_accuracy_pct.toFixed(1) + '%' : 'N/A';

              function getDiffAcc(k) {{
                return (stats.difficulty_breakdown && stats.difficulty_breakdown[k] && stats.difficulty_breakdown[k].total > 0)
                  ? stats.difficulty_breakdown[k].accuracy_pct.toFixed(1) + '%'
                  : '-';
              }}

              const spd = sc.metrics && sc.metrics.avg_tokens_per_second ? `${{sc.metrics.avg_tokens_per_second.toFixed(1)}} tok/s` : '-';
              const lat = sc.metrics && sc.metrics.avg_latency_seconds ? `${{sc.metrics.avg_latency_seconds.toFixed(2)}}s` : '-';

              md += `| ${{i + 1}} | **${{sc.model_name}}** | ${{target}} | **${{overall}}** | ${{getDiffAcc('introductory')}} | ${{getDiffAcc('intermediate')}} | ${{getDiffAcc('advanced')}} | ${{spd}} | ${{lat}} |\n`;
            }});
          }} else if (activeTable === 'tracks') {{
            md += `| Rank | Model | Target | Overall Score | Track A (MCQ) | Track B (Free-Form) | Exact Matches | Symbolic Matches | Speed | Latency |\n`;
            md += `|---|---|---|---|---|---|---|---|---|---|\n`;

            list.forEach((sc, i) => {{
              const env = sc.environment || {{}};
              const target = env.execution_type === 'cloud' ? 'Cloud API' : (sc.model_name.includes('baseline') ? 'Baseline' : 'Local');
              const stats = getModelStats(sc, activeFormat, activeDiff);
              const overall = stats.has_data && stats.overall_accuracy_pct !== null ? stats.overall_accuracy_pct.toFixed(1) + '%' : 'N/A';
              const trb = stats.track_breakdown || {{}};

              const mcqStr = trb.mcq && trb.mcq.total > 0 ? trb.mcq.accuracy_pct.toFixed(1) + '%' : '-';
              const freeStr = trb.free_form && trb.free_form.total > 0 ? trb.free_form.accuracy_pct.toFixed(1) + '%' : '-';
              const exactStr = trb.exact_matches !== undefined ? String(trb.exact_matches) : '-';
              const symbStr = trb.symbolic_matches !== undefined ? String(trb.symbolic_matches) : '-';

              const spd = sc.metrics && sc.metrics.avg_tokens_per_second ? `${{sc.metrics.avg_tokens_per_second.toFixed(1)}} tok/s` : '-';
              const lat = sc.metrics && sc.metrics.avg_latency_seconds ? `${{sc.metrics.avg_latency_seconds.toFixed(2)}}s` : '-';

              md += `| ${{i + 1}} | **${{sc.model_name}}** | ${{target}} | **${{overall}}** | ${{mcqStr}} | ${{freeStr}} | ${{exactStr}} | ${{symbStr}} | ${{spd}} | ${{lat}} |\n`;
            }});
          }}

          navigator.clipboard.writeText(md).then(() => {{
            showToast(`Leaderboard Markdown table copied to clipboard!`);
          }}).catch(() => {{
            showToast("Failed to copy table to clipboard");
          }});
        }});
      }}

      // Helper: Format Math & LaTeX
      function formatMathText(text) {{
        if (!text) return "";
        let escaped = escapeHtml(text);
        if (window.katex) {{
          escaped = escaped.replace(/\\$([^$]+)\\$/g, function(match, math) {{
            try {{
              return katex.renderToString(math, {{ throwOnError: false, displayMode: false }});
            }} catch (e) {{ return match; }}
          }});
        }}
        return escaped;
      }}

      function escapeHtml(str) {{
        if (!str) return "";
        return String(str)
          .replace(/&/g, "&amp;")
          .replace(/</g, "&lt;")
          .replace(/>/g, "&gt;")
          .replace(/"/g, "&quot;")
          .replace(/'/g, "&#039;");
      }}

      function showToast(msg) {{
        const t = document.getElementById("toast");
        t.textContent = msg;
        t.classList.add("show");
        setTimeout(() => t.classList.remove("show"), 3000);
      }}

    }})();
  </script>
</body>
</html>
"""
    return html


def save_leaderboard_dashboard(
    output_html_path: Path | str,
    benchmark_dir: Path | str = "./dataset/benchmark",
    benchmark_file: Path | str | None = None,
    open_browser: bool = False,
) -> Path:
    """Discovers all scorecards, compiles the interactive leaderboard HTML dashboard, and saves to file."""
    b_dir = Path(benchmark_dir)
    out_p = Path(output_html_path)

    scorecards = collect_scorecards_from_dir(b_dir)

    # Load benchmark items for differential comparison
    samples: list[dict[str, Any]] = []
    sample_path = Path(benchmark_file) if benchmark_file else (b_dir / "drum_benchmark_all.jsonl")
    if not sample_path.exists():
        sample_path = b_dir / "drum_benchmark_mcq.jsonl"

    if sample_path.exists():
        try:
            with open(sample_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))
        except Exception:
            pass

    html_content = generate_leaderboard_html(
        scorecards=scorecards,
        benchmark_samples=samples,
    )

    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        f.write(html_content)

    if open_browser:
        import webbrowser

        try:
            webbrowser.open(out_p.resolve().as_uri())
        except Exception:
            pass

    return out_p
