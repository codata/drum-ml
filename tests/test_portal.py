"""Unit tests for DRUM-ML Portal index.html generator."""

import tempfile
import unittest
from pathlib import Path

from drum_ml.portal.generator import generate_portal_html, save_portal_html


class TestPortalGenerator(unittest.TestCase):
    def test_generate_portal_html_basic(self):
        manifest = {
            "total_records": 1000,
            "files": [
                {
                    "name": "test_file.jsonl",
                    "relative_path": "by_archetype/test_file.jsonl",
                    "record_count": 500,
                    "byte_size": 123456,
                    "sha256": "abcdef1234567890",
                }
            ],
        }
        stats = {
            "summary": {
                "total_records": 1000,
                "unique_personas_count": 10,
                "unique_archetypes_count": 6,
            }
        }
        html = generate_portal_html(manifest_data=manifest, stats_data=stats)
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("DRUM-ML", html)
        self.assertIn("CODATA", html)
        self.assertIn("dataset_viewer.html", html)
        self.assertIn("benchmark/leaderboard.html", html)
        self.assertIn("benchmark/benchmark_viewer.html", html)
        self.assertIn("test_file.jsonl", html)

    def test_save_portal_html(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "index.html"
            p = save_portal_html(
                output_html_path=out_file,
                open_browser=False,
            )
            self.assertTrue(p.exists())
            content = p.read_text(encoding="utf-8")
            self.assertIn("CODATA DRUM Working Group", content)
            self.assertIn("Metrological Knowledge & Benchmarks for AI", content)


if __name__ == "__main__":
    unittest.main()
