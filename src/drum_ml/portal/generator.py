"""DRUM-ML Web Portal and Scientific Repository Hub Generator.

Generates a standalone, feature-rich, high-performance web portal (index.html)
aligned with the CODATA DRUM website style (Space Grotesk, Inter, JetBrains Mono,
glassmorphic surfaces, dark/light mode toggle, and live dataset telemetry).
"""

from __future__ import annotations

import json
import webbrowser
from pathlib import Path
from typing import Any


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


PORTAL_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en" data-theme="dark" class="scroll-smooth">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>CODATA DRUM-ML · Digital Metrology Datasets & Benchmark Suite</title>
  <meta name="description" content="Digital Representation of Units of Measurement for Machine Learning (DRUM-ML): High-fidelity datasets, preference pairs, and standardized benchmarks aligning LLMs with BIPM SI, CODATA constants, and QUDT." />

  <!-- Open Graph & Social Cards -->
  <meta property="og:title" content="CODATA DRUM-ML · Digital Metrology Datasets & AI Benchmark" />
  <meta property="og:description" content="High-fidelity scientific instruction datasets, preference pairs, and evaluation benchmarks for metrologically rigorous Large Language Models." />
  <meta property="og:type" content="website" />
  <meta property="og:url" content="https://drum.codata.org/ml" />

  <!-- Google Fonts: Space Grotesk (Headings), Inter (Body), JetBrains Mono (Code/Hashes) -->
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet" />

  <style>
    :root {
      --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      --font-display: 'Space Grotesk', var(--font-sans);
      --font-mono: 'JetBrains Mono', monospace;

      /* Color Palette: Dark Theme (Default) */
      --bg-base: #0b0f19;
      --bg-surface: rgba(15, 23, 42, 0.75);
      --bg-surface-solid: #0f172a;
      --bg-card: rgba(30, 41, 59, 0.6);
      --bg-card-hover: rgba(30, 41, 59, 0.85);
      --border-subtle: rgba(255, 255, 255, 0.08);
      --border-hover: rgba(56, 189, 248, 0.35);

      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --text-dim: #64748b;

      --accent-cyan: #0ea5e9;
      --accent-cyan-light: #38bdf8;
      --accent-teal: #14b8a6;
      --accent-emerald: #10b981;
      --accent-amber: #f59e0b;
      --accent-purple: #a855f7;
      --accent-rose: #f43f5e;

      --glow-cyan: rgba(14, 165, 233, 0.18);
      --glow-teal: rgba(20, 184, 166, 0.15);
      --glow-amber: rgba(245, 158, 11, 0.15);

      --shadow-glass: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
      --radius-sm: 8px;
      --radius-md: 12px;
      --radius-lg: 16px;
      --radius-xl: 24px;
    }

    [data-theme="light"] {
      --bg-base: #f8fafc;
      --bg-surface: rgba(255, 255, 255, 0.8);
      --bg-surface-solid: #ffffff;
      --bg-card: rgba(241, 245, 249, 0.8);
      --bg-card-hover: rgba(255, 255, 255, 0.95);
      --border-subtle: rgba(0, 0, 0, 0.08);
      --border-hover: rgba(14, 165, 233, 0.4);

      --text-main: #0f172a;
      --text-muted: #475569;
      --text-dim: #64748b;

      --accent-cyan: #0284c7;
      --accent-cyan-light: #0369a1;
      --accent-teal: #0d9488;
      --accent-emerald: #059669;
      --accent-amber: #d97706;
      --accent-purple: #9333ea;
      --accent-rose: #e11d48;

      --glow-cyan: rgba(14, 165, 233, 0.1);
      --glow-teal: rgba(20, 184, 166, 0.08);
      --glow-amber: rgba(245, 158, 11, 0.08);

      --shadow-glass: 0 8px 32px 0 rgba(0, 0, 0, 0.06);
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      transition: background-color 0.25s ease, border-color 0.25s ease, color 0.2s ease;
    }

    body {
      font-family: var(--font-sans);
      background-color: var(--bg-base);
      color: var(--text-main);
      min-height: 100vh;
      overflow-x: hidden;
      line-height: 1.6;
    }

    /* Typography */
    h1, h2, h3, h4, .font-display {
      font-family: var(--font-display);
      letter-spacing: -0.025em;
      font-weight: 700;
    }

    code, pre, .font-mono {
      font-family: var(--font-mono);
    }

    /* Ambient Background Rings (DRUM Style) */
    .bg-rings {
      position: fixed;
      inset: 0;
      pointer-events: none;
      z-index: 0;
      overflow: hidden;
      opacity: 0.4;
    }
    .ring-circle {
      position: absolute;
      border-radius: 50%;
      border: 1px solid var(--border-subtle);
    }
    .ring-1 { width: 900px; height: 900px; right: -200px; top: -150px; border-color: rgba(56, 189, 248, 0.08); }
    .ring-2 { width: 650px; height: 650px; right: -75px; top: -25px; border-color: rgba(20, 184, 166, 0.06); }
    .ring-3 { width: 400px; height: 400px; right: 50px; top: 100px; border-color: rgba(245, 158, 11, 0.05); }
    .ring-glow {
      position: absolute;
      width: 500px;
      height: 500px;
      right: 0;
      top: 0;
      background: radial-gradient(circle, var(--glow-cyan) 0%, transparent 70%);
      filter: blur(60px);
    }

    /* Layout Containers */
    .container {
      width: 100%;
      max-width: 1280px;
      margin: 0 auto;
      padding: 0 24px;
      position: relative;
      z-index: 1;
    }

    /* Glassmorphic Navbar */
    .navbar {
      position: sticky;
      top: 0;
      z-index: 50;
      background: var(--bg-surface);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border-bottom: 1px solid var(--border-subtle);
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }
    .nav-inner {
      display: flex;
      align-items: center;
      justify-content: space-between;
      height: 72px;
    }
    .brand-group {
      display: flex;
      align-items: center;
      gap: 14px;
      text-decoration: none;
      color: inherit;
    }
    .brand-logo-badge {
      display: flex;
      align-items: center;
      justify-content: center;
      width: 40px;
      height: 40px;
      border-radius: 10px;
      background: linear-gradient(135deg, var(--accent-cyan), var(--accent-teal));
      color: #fff;
      font-weight: 800;
      font-family: var(--font-display);
      font-size: 1.1rem;
      box-shadow: 0 0 20px var(--glow-cyan);
    }
    .brand-title {
      font-size: 1.25rem;
      font-weight: 700;
      line-height: 1.2;
    }
    .brand-subtitle {
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--accent-cyan-light);
      font-weight: 600;
    }
    .nav-links {
      display: flex;
      align-items: center;
      gap: 20px;
    }
    .nav-link {
      color: var(--text-muted);
      text-decoration: none;
      font-size: 0.88rem;
      font-weight: 600;
      padding: 6px 12px;
      border-radius: var(--radius-sm);
      transition: all 0.2s ease;
    }
    .nav-link:hover {
      color: var(--accent-cyan-light);
      background: rgba(56, 189, 248, 0.08);
    }
    .nav-actions {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    /* Buttons */
    .btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      padding: 10px 20px;
      font-size: 0.9rem;
      font-weight: 600;
      border-radius: var(--radius-md);
      text-decoration: none;
      cursor: pointer;
      border: 1px solid transparent;
      transition: all 0.2s ease;
    }
    .btn-primary {
      background: linear-gradient(135deg, var(--accent-cyan), #0284c7);
      color: #ffffff;
      box-shadow: 0 4px 16px var(--glow-cyan);
    }
    .btn-primary:hover {
      transform: translateY(-2px);
      box-shadow: 0 6px 24px rgba(14, 165, 233, 0.4);
    }
    .btn-secondary {
      background: var(--bg-card);
      color: var(--text-main);
      border-color: var(--border-subtle);
    }
    .btn-secondary:hover {
      background: var(--bg-card-hover);
      border-color: var(--border-hover);
      transform: translateY(-2px);
    }
    .btn-outline-amber {
      background: rgba(245, 158, 11, 0.1);
      color: var(--accent-amber);
      border-color: rgba(245, 158, 11, 0.3);
    }
    .btn-outline-amber:hover {
      background: rgba(245, 158, 11, 0.2);
      border-color: var(--accent-amber);
    }
    .btn-icon {
      padding: 8px;
      border-radius: 50%;
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      color: var(--text-muted);
      cursor: pointer;
    }
    .btn-icon:hover {
      color: var(--accent-cyan-light);
      border-color: var(--border-hover);
    }

    /* Hero Section */
    .hero-section {
      padding: 72px 0 48px;
      position: relative;
    }
    .hero-badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 14px;
      border-radius: 9999px;
      background: rgba(14, 165, 233, 0.1);
      border: 1px solid rgba(14, 165, 233, 0.3);
      color: var(--accent-cyan-light);
      font-size: 0.8rem;
      font-weight: 700;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      margin-bottom: 24px;
    }
    .hero-title {
      font-size: 3.25rem;
      line-height: 1.15;
      font-weight: 800;
      margin-bottom: 20px;
      max-width: 950px;
      background: linear-gradient(135deg, var(--text-main) 30%, var(--accent-cyan-light) 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
      font-size: 1.2rem;
      color: var(--text-muted);
      max-width: 820px;
      margin-bottom: 36px;
      line-height: 1.6;
    }
    .hero-cta-group {
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      margin-bottom: 48px;
    }

    /* KPI Summary Stats Bar */
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
      margin-bottom: 64px;
    }
    .kpi-card {
      background: var(--bg-surface);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-lg);
      padding: 20px 24px;
      position: relative;
      overflow: hidden;
      transition: all 0.25s ease;
    }
    .kpi-card:hover {
      border-color: var(--border-hover);
      transform: translateY(-2px);
      box-shadow: var(--shadow-glass);
    }
    .kpi-card::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 3px;
      background: linear-gradient(90deg, var(--accent-cyan), var(--accent-teal));
      opacity: 0.8;
    }
    .kpi-val {
      font-size: 2.25rem;
      font-weight: 800;
      font-family: var(--font-display);
      color: var(--text-main);
      line-height: 1.1;
      margin-bottom: 4px;
    }
    .kpi-label {
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--text-muted);
      font-weight: 600;
    }
    .kpi-sub {
      font-size: 0.78rem;
      color: var(--accent-cyan-light);
      margin-top: 6px;
      font-family: var(--font-mono);
    }

    /* Section Headers */
    .section-header {
      margin-bottom: 32px;
    }
    .section-tag {
      font-size: 0.78rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--accent-cyan-light);
      margin-bottom: 8px;
    }
    .section-title {
      font-size: 2rem;
      font-weight: 700;
      color: var(--text-main);
    }
    .section-desc {
      font-size: 1rem;
      color: var(--text-muted);
      max-width: 700px;
      margin-top: 6px;
    }

    /* Interactive Apps Grid */
    .app-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 24px;
      margin-bottom: 64px;
    }
    .app-card {
      background: var(--bg-surface);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-xl);
      padding: 32px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      text-decoration: none;
      color: inherit;
      position: relative;
      overflow: hidden;
      transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .app-card:hover {
      border-color: var(--border-hover);
      transform: translateY(-4px);
      box-shadow: 0 12px 36px rgba(0, 0, 0, 0.3), 0 0 24px var(--glow-cyan);
    }
    .app-card-badge {
      align-self: flex-start;
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 16px;
    }
    .badge-explorer { background: rgba(56, 189, 248, 0.12); color: var(--accent-cyan-light); border: 1px solid rgba(56, 189, 248, 0.25); }
    .badge-leaderboard { background: rgba(245, 158, 11, 0.12); color: var(--accent-amber); border: 1px solid rgba(245, 158, 11, 0.25); }
    .badge-reviewer { background: rgba(16, 185, 129, 0.12); color: var(--accent-emerald); border: 1px solid rgba(16, 185, 129, 0.25); }

    .app-card-title {
      font-size: 1.5rem;
      font-weight: 700;
      margin-bottom: 12px;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .app-card-desc {
      color: var(--text-muted);
      font-size: 0.95rem;
      line-height: 1.6;
      margin-bottom: 24px;
      flex-grow: 1;
    }
    .app-card-footer {
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-top: 1px solid var(--border-subtle);
      padding-top: 16px;
      font-size: 0.85rem;
      color: var(--accent-cyan-light);
      font-weight: 600;
    }
    .app-card-footer span {
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }

    /* Knowledge Source Standards Banner */
    .standards-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 20px;
      margin-bottom: 64px;
    }
    .standard-card {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-lg);
      padding: 24px;
    }
    .standard-tier {
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--accent-amber);
      margin-bottom: 6px;
    }
    .standard-title {
      font-size: 1.15rem;
      font-weight: 700;
      margin-bottom: 8px;
    }
    .standard-desc {
      font-size: 0.88rem;
      color: var(--text-muted);
      line-height: 1.5;
    }
    .standard-link {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      margin-top: 12px;
      font-size: 0.82rem;
      color: var(--accent-cyan-light);
      text-decoration: none;
      font-weight: 600;
    }
    .standard-link:hover {
      text-decoration: underline;
    }

    /* Data Partitions & Downloads Table */
    .download-section {
      background: var(--bg-surface);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-xl);
      padding: 36px;
      margin-bottom: 64px;
    }
    .download-toolbar {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 24px;
    }
    .search-box {
      display: flex;
      align-items: center;
      gap: 10px;
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 8px 16px;
      width: 100%;
      max-width: 360px;
    }
    .search-box input {
      background: transparent;
      border: none;
      outline: none;
      color: var(--text-main);
      font-size: 0.9rem;
      width: 100%;
    }
    .filter-pills {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .pill-btn {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      color: var(--text-muted);
      padding: 6px 14px;
      border-radius: 9999px;
      font-size: 0.8rem;
      font-weight: 600;
      cursor: pointer;
    }
    .pill-btn.active, .pill-btn:hover {
      background: var(--accent-cyan);
      color: #fff;
      border-color: var(--accent-cyan);
    }

    .table-wrapper {
      overflow-x: auto;
      border-radius: var(--radius-md);
      border: 1px solid var(--border-subtle);
    }
    .data-table {
      width: 100%;
      border-collapse: collapse;
      text-align: left;
      font-size: 0.88rem;
    }
    .data-table th {
      background: rgba(30, 41, 59, 0.7);
      padding: 14px 18px;
      font-weight: 700;
      color: var(--text-muted);
      text-transform: uppercase;
      font-size: 0.75rem;
      letter-spacing: 0.05em;
      border-bottom: 1px solid var(--border-subtle);
    }
    .data-table td {
      padding: 14px 18px;
      border-bottom: 1px solid var(--border-subtle);
      color: var(--text-main);
    }
    .data-table tr:last-child td {
      border-bottom: none;
    }
    .data-table tr:hover td {
      background: rgba(56, 189, 248, 0.04);
    }
    .file-name-cell {
      font-family: var(--font-mono);
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--accent-cyan-light);
    }
    .hash-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 3px 8px;
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 4px;
      font-family: var(--font-mono);
      font-size: 0.75rem;
      color: var(--text-muted);
      cursor: pointer;
    }
    .hash-badge:hover {
      border-color: var(--accent-cyan);
      color: var(--accent-cyan-light);
    }

    /* Citation & Hugging Face Code Box */
    .code-box-card {
      background: var(--bg-surface-solid);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-lg);
      padding: 24px;
      margin-bottom: 64px;
      position: relative;
    }
    .code-box-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 12px;
      font-size: 0.85rem;
      font-weight: 700;
      color: var(--text-muted);
    }
    .code-content {
      background: #060911;
      padding: 16px 20px;
      border-radius: var(--radius-md);
      font-family: var(--font-mono);
      font-size: 0.85rem;
      color: #38bdf8;
      overflow-x: auto;
      white-space: pre-wrap;
    }

    /* Footer */
    .site-footer {
      border-top: 1px solid var(--border-subtle);
      background: var(--bg-surface);
      padding: 48px 0 32px;
      margin-top: 80px;
      font-size: 0.88rem;
      color: var(--text-muted);
    }
    .footer-grid {
      display: grid;
      grid-template-columns: 2fr 1fr 1fr 1fr;
      gap: 40px;
      margin-bottom: 36px;
    }
    .footer-brand h4 {
      font-size: 1.1rem;
      color: var(--text-main);
      margin-bottom: 10px;
    }
    .footer-col h5 {
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--text-main);
      margin-bottom: 14px;
      font-weight: 700;
    }
    .footer-links {
      list-style: none;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .footer-links a {
      color: var(--text-muted);
      text-decoration: none;
      transition: color 0.2s;
    }
    .footer-links a:hover {
      color: var(--accent-cyan-light);
    }
    .footer-bottom {
      border-top: 1px solid var(--border-subtle);
      padding-top: 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 0.8rem;
    }

    /* Toast Notification */
    .toast {
      position: fixed;
      bottom: 24px;
      right: 24px;
      background: var(--bg-surface-solid);
      border: 1px solid var(--accent-cyan);
      color: var(--text-main);
      padding: 12px 20px;
      border-radius: var(--radius-md);
      box-shadow: 0 8px 24px rgba(0,0,0,0.4);
      z-index: 100;
      opacity: 0;
      transform: translateY(10px);
      transition: all 0.3s ease;
      pointer-events: none;
    }
    .toast.show {
      opacity: 1;
      transform: translateY(0);
    }

    @media (max-width: 900px) {
      .hero-title { font-size: 2.5rem; }
      .footer-grid { grid-template-columns: 1fr; gap: 24px; }
      .nav-links { display: none; }
    }
  </style>
</head>
<body>

  <!-- Ambient Rings Background Texture -->
  <div class="bg-rings">
    <div class="ring-circle ring-1"></div>
    <div class="ring-circle ring-2"></div>
    <div class="ring-circle ring-3"></div>
    <div class="ring-glow"></div>
  </div>

  <!-- Navigation Bar -->
  <header class="navbar">
    <div class="container">
      <div class="nav-inner">
        <a href="/" class="brand-group">
          <div class="brand-logo-badge">SI</div>
          <div>
            <div class="brand-title">DRUM-ML</div>
            <div class="brand-subtitle">CODATA Working Group</div>
          </div>
        </a>

        <nav class="nav-links">
          <a href="#overview" class="nav-link">Overview</a>
          <a href="#apps" class="nav-link">Interactive Apps</a>
          <a href="#standards" class="nav-link">Standards</a>
          <a href="#downloads" class="nav-link">Downloads & Partitions</a>
          <a href="#citation" class="nav-link">Citation</a>
        </nav>

        <div class="nav-actions">
          <a href="https://drum.codata.org" target="_blank" rel="noopener" class="btn btn-secondary" style="font-size:0.82rem; padding:6px 14px;">
            drum.codata.org ↗
          </a>
          <button id="themeToggle" class="btn-icon" title="Toggle Light/Dark Theme">
            <svg id="themeIconSun" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>
            <svg id="themeIconMoon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="display:none;"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>
          </button>
        </div>
      </div>
    </div>
  </header>

  <!-- Hero Section -->
  <section class="hero-section" id="overview">
    <div class="container">
      <div class="hero-badge">
        <span>CODATA DRUM Initiative</span>
        <span>•</span>
        <span>Digital Representation of Units of Measurement</span>
      </div>

      <h1 class="hero-title">Metrological Knowledge & Benchmarks for AI</h1>
      <p class="hero-subtitle">
        High-fidelity scientific instruction datasets, preference pairs, and standardized benchmarks aligning Large Language Models with the <strong>BIPM SI Digital Framework</strong>, official <strong>CODATA Fundamental Constants</strong>, and <strong>QUDT</strong> ontologies.
      </p>

      <div class="hero-cta-group">
        <a href="dataset_viewer.html" class="btn btn-primary">
          <span>🔍 Launch Dataset Explorer</span>
        </a>
        <a href="benchmark/leaderboard.html" class="btn btn-secondary">
          <span>🏆 Model Leaderboard</span>
        </a>
        <a href="benchmark/benchmark_viewer.html" class="btn btn-secondary">
          <span>🧪 Benchmark Reviewer</span>
        </a>
        <a href="#downloads" class="btn btn-outline-amber">
          <span>📦 Download Partitions</span>
        </a>
      </div>

      <!-- Live KPI Grid -->
      <div class="kpi-grid">
        <div class="kpi-card">
          <div class="kpi-val" id="kpiTotalRecords">205,189</div>
          <div class="kpi-label">Total Dialogue Pairs</div>
          <div class="kpi-sub">SFT (Train/Val/Test) + DPO</div>
        </div>

        <div class="kpi-card">
          <div class="kpi-val" id="kpiTotalTokens">34.8M+</div>
          <div class="kpi-label">Synthesized Tokens</div>
          <div class="kpi-sub">169.6 avg tokens / sample</div>
        </div>

        <div class="kpi-card">
          <div class="kpi-val">43</div>
          <div class="kpi-label">Scientific Unions</div>
          <div class="kpi-sub">IUPAP, IUPAC, IAU, IUGG, IEEE...</div>
        </div>

        <div class="kpi-card">
          <div class="kpi-val">6</div>
          <div class="kpi-label">Pedagogical Archetypes</div>
          <div class="kpi-sub">Decomposition, Units, GUM...</div>
        </div>

        <div class="kpi-card">
          <div class="kpi-val">100%</div>
          <div class="kpi-label">Ground-Truth SI 2019</div>
          <div class="kpi-sub">BIPM & CODATA Verified</div>
        </div>
      </div>
    </div>
  </section>

  <!-- Interactive Application Suite -->
  <section class="container" id="apps" style="padding-top: 24px;">
    <div class="section-header">
      <div class="section-tag">Interactive Review & Benchmark Suite</div>
      <h2 class="section-title">Standalone Visual Explorers</h2>
      <p class="section-desc">Zero-dependency, browser-native applications for inspecting training pairs, auditing test cases, and tracking model leaderboards.</p>
    </div>

    <div class="app-grid">
      <!-- Card 1: Dataset Explorer -->
      <a href="dataset_viewer.html" class="app-card">
        <div>
          <span class="app-card-badge badge-explorer">🔍 Dataset Inspector</span>
          <h3 class="app-card-title">Metrology Instruction Explorer</h3>
          <p class="app-card-desc">
            Explore 200,000+ instruction fine-tuning samples and DPO preference pairs across 43 scientific personas. Features KaTeX math rendering, persona filters, multi-format JSON/ShareGPT switching, and full-text search.
          </p>
        </div>
        <div class="app-card-footer">
          <span>Open dataset_viewer.html</span>
          <span>→</span>
        </div>
      </a>

      <!-- Card 2: Leaderboard -->
      <a href="benchmark/leaderboard.html" class="app-card">
        <div>
          <span class="app-card-badge badge-leaderboard">🏆 Performance Arena</span>
          <h3 class="app-card-title">Multi-Model Leaderboard</h3>
          <p class="app-card-desc">
            Compare open and frontier LLMs across Track A (MCQ) and Track B (Free-form symbolic) evaluation testbeds. Includes throughput (tok/s), latency distributions, radar competency charts, and host hardware telemetry.
          </p>
        </div>
        <div class="app-card-footer">
          <span>Open benchmark/leaderboard.html</span>
          <span>→</span>
        </div>
      </a>

      <!-- Card 3: Benchmark Reviewer -->
      <a href="benchmark/benchmark_viewer.html" class="app-card">
        <div>
          <span class="app-card-badge badge-reviewer">🧪 Gold Testbed</span>
          <h3 class="app-card-title">Benchmark Question Reviewer</h3>
          <p class="app-card-desc">
            Audit held-out benchmark questions across 6 core metrological sub-disciplines. Inspect distractor rationales, SI typography violations, exact CODATA defining constant values, and symbolic Pint/SymPy assertions.
          </p>
        </div>
        <div class="app-card-footer">
          <span>Open benchmark/benchmark_viewer.html</span>
          <span>→</span>
        </div>
      </a>
    </div>
  </section>

  <!-- Standards & Knowledge Precedence -->
  <section class="container" id="standards">
    <div class="section-header">
      <div class="section-tag">Authoritative Foundation</div>
      <h2 class="section-title">Master Metrological Repositories</h2>
      <p class="section-desc">The DRUM-ML dataset enforces a strict three-tier precedence hierarchy to ensure zero hallucination against official international standards.</p>
    </div>

    <div class="standards-grid">
      <div class="standard-card">
        <div class="standard-tier">Tier 1 · Ground Truth</div>
        <h3 class="standard-title">BIPM SI Digital Framework</h3>
        <p class="standard-desc">
          The official digital representation of the International System of Units (SI) maintained by BIPM (9th Edition SI Brochure). Defining constants ($c, h, e, k, N_{\\text{A}}, \\Delta\\nu_{\\text{Cs}}, K_{\\text{cd}}$) and base unit rules.
        </p>
        <a href="https://si-digital-framework.org/SI" target="_blank" rel="noopener" class="standard-link">si-digital-framework.org ↗</a>
      </div>

      <div class="standard-card">
        <div class="standard-tier">Tier 2 · Physical Constants</div>
        <h3 class="standard-title">CODATA DRUM Constants</h3>
        <p class="standard-desc">
          Official physical constant re-evaluations published by NIST and recommended by CODATA (1969–2022). Standard uncertainties ($u$), relative uncertainties ($u_r$), and correlation matrices.
        </p>
        <a href="https://github.com/codata/drum-constants" target="_blank" rel="noopener" class="standard-link">github.com/codata/drum-constants ↗</a>
      </div>

      <div class="standard-card">
        <div class="standard-tier">Tier 3 · Semantic Ontologies</div>
        <h3 class="standard-title">QUDT 2.1 Ontologies</h3>
        <p class="standard-desc">
          Semantic web reference graph for quantities, units, dimensions, and types. Non-SI units, legacy units, UCUM mappings, and affine offset conversion formulas.
        </p>
        <a href="https://qudt.org" target="_blank" rel="noopener" class="standard-link">qudt.org ↗</a>
      </div>
    </div>
  </section>

  <!-- Data Downloads & Partitions Center -->
  <section class="container" id="downloads">
    <div class="download-section">
      <div class="section-header">
        <div class="section-tag">Data Repository</div>
        <h2 class="section-title">Dataset Partitions & Downloads</h2>
        <p class="section-desc">Download granular splits partitioned by scientific union persona, pedagogical archetype, and entity category.</p>
      </div>

      <div class="download-toolbar">
        <div class="search-box">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
          <input type="text" id="fileSearchInput" placeholder="Filter files by name or partition..." />
        </div>

        <div class="filter-pills" id="partitionFilterPills">
          <button class="pill-btn active" data-filter="all">All Files</button>
          <button class="pill-btn" data-filter="master">Master Splits</button>
          <button class="pill-btn" data-filter="persona">by_persona</button>
          <button class="pill-btn" data-filter="archetype">by_archetype</button>
          <button class="pill-btn" data-filter="benchmark">Benchmark</button>
        </div>
      </div>

      <div class="table-wrapper">
        <table class="data-table" id="filesTable">
          <thead>
            <tr>
              <th>File Name & Partition</th>
              <th>Category</th>
              <th>Records</th>
              <th>Size</th>
              <th>SHA-256 Checksum</th>
              <th style="text-align:right;">Action</th>
            </tr>
          </thead>
          <tbody id="filesTableBody">
            <!-- Populated dynamically from manifest.json -->
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <!-- Hugging Face & Quick Start -->
  <section class="container" id="citation">
    <div class="code-box-card">
      <div class="code-box-header">
        <span>🤗 Hugging Face Datasets Quickstart</span>
        <button class="btn btn-secondary" style="font-size:0.75rem; padding:4px 10px;" onclick="copyCode('hfSnippet')">Copy Code</button>
      </div>
      <div class="code-content" id="hfSnippet">from datasets import load_dataset

# Load the master supervised fine-tuning dataset
dataset = load_dataset("codata/drum-metrology-instruct")

# Load Direct Preference Optimization (DPO) pairs
dpo_pairs = load_dataset("codata/drum-metrology-instruct", name="dpo")</div>
    </div>

    <!-- BibTeX Citation Box -->
    <div class="code-box-card">
      <div class="code-box-header">
        <span>📖 BibTeX Attribution</span>
        <button class="btn btn-secondary" style="font-size:0.75rem; padding:4px 10px;" onclick="copyCode('bibtexSnippet')">Copy BibTeX</button>
      </div>
      <div class="code-content" id="bibtexSnippet">@misc{codata_drum_ml_2026,
  title={DRUM-ML: Digital Representation of Units of Measurement for Machine Learning},
  author={{CODATA DRUM Working Group}},
  year={2026},
  publisher={CODATA},
  howpublished={\\url{https://drum.codata.org/ml}}
}</div>
    </div>
  </section>

  <!-- Footer -->
  <footer class="site-footer">
    <div class="container">
      <div class="footer-grid">
        <div class="footer-brand">
          <h4>CODATA DRUM Working Group</h4>
          <p>Enabling AI-ready data, machine-actionable science, and dimensional integrity by standardizing digital units of measurement across global scientific communities.</p>
        </div>
        <div class="footer-col">
          <h5>Applications</h5>
          <ul class="footer-links">
            <li><a href="dataset_viewer.html">Dataset Explorer</a></li>
            <li><a href="benchmark/leaderboard.html">Model Leaderboard</a></li>
            <li><a href="benchmark/benchmark_viewer.html">Benchmark Reviewer</a></li>
          </ul>
        </div>
        <div class="footer-col">
          <h5>Knowledge Sources</h5>
          <ul class="footer-links">
            <li><a href="https://si-digital-framework.org/SI" target="_blank" rel="noopener">BIPM SI Digital Framework</a></li>
            <li><a href="https://github.com/codata/drum-constants" target="_blank" rel="noopener">CODATA Constants</a></li>
            <li><a href="https://qudt.org" target="_blank" rel="noopener">QUDT Ontologies</a></li>
          </ul>
        </div>
        <div class="footer-col">
          <h5>Initiative</h5>
          <ul class="footer-links">
            <li><a href="https://drum.codata.org" target="_blank" rel="noopener">DRUM Homepage</a></li>
            <li><a href="https://codata.org" target="_blank" rel="noopener">CODATA Official</a></li>
            <li><a href="#downloads">Dataset Manifest</a></li>
          </ul>
        </div>
      </div>

      <div class="footer-bottom">
        <div>© 2026 CODATA DRUM Working Group. Open Access Scientific Resource.</div>
        <div style="display:flex; gap:16px;">
          <a href="https://drum.codata.org" target="_blank" rel="noopener" style="color:inherit; text-decoration:none;">drum.codata.org</a>
          <span>•</span>
          <a href="https://codata.org" target="_blank" rel="noopener" style="color:inherit; text-decoration:none;">codata.org</a>
        </div>
      </div>
    </div>
  </footer>

  <!-- Toast Element -->
  <div class="toast" id="toast">Copied to clipboard!</div>

  <!-- Embedded Manifest & Stats Data -->
  <script id="embeddedManifest" type="application/json">
__MANIFEST_JSON__
  </script>
  <script id="embeddedStats" type="application/json">
__STATS_JSON__
  </script>

  <!-- Interactive Portal Script -->
  <script>
    (function() {
      "use strict";

      let manifest = {};
      let stats = {};
      let activeFilter = "all";

      // 1. Theme Management
      const themeToggleBtn = document.getElementById("themeToggle");
      const sunIcon = document.getElementById("themeIconSun");
      const moonIcon = document.getElementById("themeIconMoon");

      function initTheme() {
        const savedTheme = localStorage.getItem("drum_theme") || "dark";
        document.documentElement.setAttribute("data-theme", savedTheme);
        updateThemeIcons(savedTheme);
      }

      function updateThemeIcons(theme) {
        if (theme === "light") {
          sunIcon.style.display = "none";
          moonIcon.style.display = "block";
        } else {
          sunIcon.style.display = "block";
          moonIcon.style.display = "none";
        }
      }

      themeToggleBtn.addEventListener("click", () => {
        const current = document.documentElement.getAttribute("data-theme") || "dark";
        const next = current === "dark" ? "light" : "dark";
        document.documentElement.setAttribute("data-theme", next);
        localStorage.setItem("drum_theme", next);
        updateThemeIcons(next);
      });

      // 2. Ingest Embedded Manifest & Stats
      try {
        const rawMan = document.getElementById("embeddedManifest").textContent;
        manifest = JSON.parse(rawMan) || {};
      } catch (e) {
        manifest = {};
      }

      try {
        const rawStat = document.getElementById("embeddedStats").textContent;
        stats = JSON.parse(rawStat) || {};
      } catch (e) {
        stats = {};
      }

      // Update Hero KPIs
      if (stats.summary) {
        if (stats.summary.total_records) {
          document.getElementById("kpiTotalRecords").textContent = stats.summary.total_records.toLocaleString();
        }
      } else if (manifest.total_records) {
        document.getElementById("kpiTotalRecords").textContent = manifest.total_records.toLocaleString();
      }

      // 3. Render Files Table
      const filesList = [];

      // Collect files from manifest
      if (Array.isArray(manifest.files)) {
        manifest.files.forEach(f => filesList.push(f));
      } else {
        // Fallback default structure
        filesList.push(
          { name: "train.jsonl", relative_path: "train.jsonl", record_count: 174410, byte_size: 181285609, sha256: "master-split" },
          { name: "val.jsonl", relative_path: "val.jsonl", record_count: 20518, byte_size: 21356635, sha256: "master-split" },
          { name: "test.jsonl", relative_path: "test.jsonl", record_count: 10261, byte_size: 10691189, sha256: "master-split" },
          { name: "dpo_preferences.jsonl", relative_path: "dpo_preferences.jsonl", record_count: 70, byte_size: 60945, sha256: "master-split" },
          { name: "drum_benchmark_all.jsonl", relative_path: "benchmark/drum_benchmark_all.jsonl", record_count: 300, byte_size: 386366, sha256: "benchmark-suite" }
        );
      }

      function formatBytes(bytes) {
        if (!bytes || bytes === 0) return "0 B";
        const k = 1024;
        const sizes = ["B", "KB", "MB", "GB"];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
      }

      function getCategory(path) {
        if (path.startsWith("by_persona/")) return "Persona";
        if (path.startsWith("by_archetype/")) return "Archetype";
        if (path.startsWith("by_category/")) return "Category";
        if (path.startsWith("benchmark/")) return "Benchmark";
        return "Master Split";
      }

      function renderFiles() {
        const tbody = document.getElementById("filesTableBody");
        const query = document.getElementById("fileSearchInput").value.trim().toLowerCase();
        tbody.innerHTML = "";

        const filtered = filesList.filter(f => {
          const cat = getCategory(f.relative_path);
          if (activeFilter === "master" && cat !== "Master Split") return false;
          if (activeFilter === "persona" && cat !== "Persona") return false;
          if (activeFilter === "archetype" && cat !== "Archetype") return false;
          if (activeFilter === "benchmark" && cat !== "Benchmark") return false;

          if (query) {
            const matchName = f.name.toLowerCase().includes(query);
            const matchPath = f.relative_path.toLowerCase().includes(query);
            const matchCat = cat.toLowerCase().includes(query);
            return matchName || matchPath || matchCat;
          }
          return true;
        });

        if (filtered.length === 0) {
          tbody.innerHTML = `<tr><td colspan="6" style="text-align:center; padding:24px; color:var(--text-muted);">No matching files found.</td></tr>`;
          return;
        }

        filtered.forEach(f => {
          const cat = getCategory(f.relative_path);
          const tr = document.createElement("tr");
          const shortHash = f.sha256 && f.sha256.length > 12 ? f.sha256.substring(0, 10) + "..." : (f.sha256 || "—");

          tr.innerHTML = `
            <td class="file-name-cell">
              <a href="${f.relative_path}" download style="color:inherit; text-decoration:none;">📄 ${f.name}</a>
              <div style="font-size:0.75rem; color:var(--text-dim); font-family:var(--font-sans); margin-top:2px;">${f.relative_path}</div>
            </td>
            <td><span style="font-size:0.75rem; font-weight:700; text-transform:uppercase; padding:3px 8px; border-radius:4px; background:var(--bg-card); border:1px solid var(--border-subtle);">${cat}</span></td>
            <td><strong>${(f.record_count || 0).toLocaleString()}</strong></td>
            <td>${formatBytes(f.byte_size)}</td>
            <td>
              <span class="hash-badge" onclick="copyText('${f.sha256 || ''}')" title="Click to copy full SHA-256 hash">
                ${shortHash} 📋
              </span>
            </td>
            <td style="text-align:right;">
              <a href="${f.relative_path}" download class="btn btn-secondary" style="font-size:0.75rem; padding:4px 10px;">
                Download
              </a>
            </td>
          `;
          tbody.appendChild(tr);
        });
      }

      // 4. Setup Filters & Search
      document.getElementById("fileSearchInput").addEventListener("input", renderFiles);

      document.querySelectorAll("#partitionFilterPills .pill-btn").forEach(btn => {
        btn.addEventListener("click", () => {
          document.querySelectorAll("#partitionFilterPills .pill-btn").forEach(b => b.classList.remove("active"));
          btn.classList.add("active");
          activeFilter = btn.getAttribute("data-filter");
          renderFiles();
        });
      });

      // 5. Copy Helpers
      window.copyCode = function(id) {
        const el = document.getElementById(id);
        if (el) {
          navigator.clipboard.writeText(el.innerText || el.textContent);
          showToast("Copied to clipboard!");
        }
      };

      window.copyText = function(text) {
        if (!text) return;
        navigator.clipboard.writeText(text);
        showToast("SHA-256 copied: " + text.substring(0, 16) + "...");
      };

      function showToast(msg) {
        const toast = document.getElementById("toast");
        toast.textContent = msg;
        toast.classList.add("show");
        setTimeout(() => toast.classList.remove("show"), 2500);
      }

      // Initialize
      initTheme();
      renderFiles();
    })();
  </script>
</body>
</html>
"""


def generate_portal_html(
    manifest_data: dict[str, Any] | None = None,
    stats_data: dict[str, Any] | None = None,
) -> str:
    """Generates the self-contained scientific repository portal HTML."""
    manifest_json_str = json.dumps(manifest_data or {}, indent=2)
    stats_json_str = json.dumps(stats_data or {}, indent=2)

    html = PORTAL_HTML_TEMPLATE.replace("__MANIFEST_JSON__", manifest_json_str)
    html = html.replace("__STATS_JSON__", stats_json_str)
    return html


def save_portal_html(
    output_html_path: str | Path = "./dataset/index.html",
    manifest_path: str | Path = "./dataset/manifest.json",
    stats_path: str | Path = "./dataset/dataset_stats.json",
    open_browser: bool = False,
) -> Path:
    """Generates and saves the master portal index.html into the dataset root."""
    out_p = Path(output_html_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    manifest = load_json_file(manifest_path)
    stats = load_json_file(stats_path)

    html = generate_portal_html(manifest_data=manifest, stats_data=stats)

    with open(out_p, "w", encoding="utf-8") as f:
        f.write(html)

    if open_browser:
        webbrowser.open(out_p.resolve().as_uri())

    return out_p


if __name__ == "__main__":
    import sys

    out_file = sys.argv[1] if len(sys.argv) > 1 else "dataset/index.html"
    p = save_portal_html(out_file)
    print(f"Generated DRUM-ML Portal index.html at: {p.resolve()}")
