"""Unit tests for DRUM-ML dataset website packaging utility."""

import tempfile
import unittest
import zipfile
from pathlib import Path

from drum_ml.portal.packager import (
    collect_website_files,
    package_website_zip,
    verify_website_zip,
)


class TestWebsitePackager(unittest.TestCase):
    def setUp(self):
        self.tmp_dir_obj = tempfile.TemporaryDirectory()
        self.test_root = Path(self.tmp_dir_obj.name)

        # Create mock dataset directory structure
        self.dataset_dir = self.test_root / "dataset"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        (self.dataset_dir / "benchmark").mkdir(parents=True, exist_ok=True)
        (self.dataset_dir / "by_persona").mkdir(parents=True, exist_ok=True)
        (self.dataset_dir / "by_archetype").mkdir(parents=True, exist_ok=True)

        # Website files
        (self.dataset_dir / "index.html").write_text(
            "<!DOCTYPE html><html>Portal</html>", encoding="utf-8"
        )
        (self.dataset_dir / "dataset_viewer.html").write_text(
            "<!DOCTYPE html><html>Viewer</html>", encoding="utf-8"
        )
        (self.dataset_dir / "manifest.json").write_text('{"total_records": 100}', encoding="utf-8")
        (self.dataset_dir / "dataset_stats.json").write_text('{"summary": {}}', encoding="utf-8")
        (self.dataset_dir / "dataset_card.md").write_text("# Dataset Card", encoding="utf-8")
        (self.dataset_dir / "archetype_report.json").write_text("{}", encoding="utf-8")
        (self.dataset_dir / "persona_report.json").write_text("{}", encoding="utf-8")
        (self.dataset_dir / "dpo_preferences.jsonl").write_text(
            '{"prompt": "p"}\n', encoding="utf-8"
        )

        # Benchmark files
        (self.dataset_dir / "benchmark" / "leaderboard.html").write_text(
            "<!DOCTYPE html><html>Leaderboard</html>", encoding="utf-8"
        )
        (self.dataset_dir / "benchmark" / "benchmark_viewer.html").write_text(
            "<!DOCTYPE html><html>Reviewer</html>", encoding="utf-8"
        )
        (self.dataset_dir / "benchmark" / "benchmark_report.json").write_text(
            "{}", encoding="utf-8"
        )
        (self.dataset_dir / "benchmark" / "drum_benchmark_all.jsonl").write_text(
            '{"task": "t"}\n', encoding="utf-8"
        )
        (self.dataset_dir / "benchmark" / "model_scorecard.json").write_text(
            '{"model": "m"}', encoding="utf-8"
        )

        # Files that SHOULD be excluded by default
        (self.dataset_dir / "train.jsonl").write_text('{"role": "user"}\n' * 500, encoding="utf-8")
        (self.dataset_dir / "val.jsonl").write_text('{"role": "user"}\n' * 100, encoding="utf-8")
        (self.dataset_dir / "test.jsonl").write_text('{"role": "user"}\n' * 50, encoding="utf-8")
        (self.dataset_dir / "by_persona" / "metrologist.jsonl").write_text(
            '{"persona": "p"}\n', encoding="utf-8"
        )
        (self.dataset_dir / "by_archetype" / "units.jsonl").write_text(
            '{"archetype": "a"}\n', encoding="utf-8"
        )
        (self.dataset_dir / ".DS_Store").write_bytes(b"\x00\x00\x00\x01")
        (self.dataset_dir / "benchmark" / ".checkpoint.json").write_text("{}", encoding="utf-8")

    def tearDown(self):
        self.tmp_dir_obj.cleanup()

    def test_collect_website_files_default(self):
        collected = collect_website_files(self.dataset_dir, include_all_data=False)
        rel_paths = [rel for _, rel in collected]

        # Key website files must be present
        self.assertIn("index.html", rel_paths)
        self.assertIn("dataset_viewer.html", rel_paths)
        self.assertIn("manifest.json", rel_paths)
        self.assertIn("dataset_stats.json", rel_paths)
        self.assertIn("dataset_card.md", rel_paths)
        self.assertIn("archetype_report.json", rel_paths)
        self.assertIn("persona_report.json", rel_paths)
        self.assertIn("dpo_preferences.jsonl", rel_paths)
        self.assertIn("benchmark/leaderboard.html", rel_paths)
        self.assertIn("benchmark/benchmark_viewer.html", rel_paths)
        self.assertIn("benchmark/benchmark_report.json", rel_paths)
        self.assertIn("benchmark/drum_benchmark_all.jsonl", rel_paths)
        self.assertIn("benchmark/model_scorecard.json", rel_paths)

        # Excluded data files & system artifacts must NOT be present
        self.assertNotIn("train.jsonl", rel_paths)
        self.assertNotIn("val.jsonl", rel_paths)
        self.assertNotIn("test.jsonl", rel_paths)
        self.assertNotIn("by_persona/metrologist.jsonl", rel_paths)
        self.assertNotIn("by_archetype/units.jsonl", rel_paths)
        self.assertNotIn(".DS_Store", rel_paths)
        self.assertNotIn("benchmark/.checkpoint.json", rel_paths)

    def test_package_website_zip_and_verify(self):
        out_zip = self.dataset_dir / "drum_ml_website.zip"
        res = package_website_zip(
            dataset_dir=self.dataset_dir,
            output_zip=out_zip,
            include_all_data=False,
            rebuild_portal=False,
            verify=True,
        )

        self.assertTrue(out_zip.exists())
        self.assertGreater(res.file_count, 10)
        self.assertGreater(res.uncompressed_bytes, 0)
        self.assertGreater(res.compressed_bytes, 0)
        self.assertGreater(len(res.sha256_hash), 30)
        self.assertTrue(res.verification_passed)

        # Ensure the output zip itself was not added inside the archive
        with zipfile.ZipFile(out_zip, "r") as zf:
            namelist = zf.namelist()
            self.assertNotIn("drum_ml_website.zip", namelist)
            self.assertIn("index.html", namelist)
            self.assertIn("benchmark/leaderboard.html", namelist)

    def test_package_website_include_all_data(self):
        out_zip = self.dataset_dir / "full_dataset.zip"
        res = package_website_zip(
            dataset_dir=self.dataset_dir,
            output_zip=out_zip,
            include_all_data=True,
            rebuild_portal=False,
            verify=True,
        )

        self.assertTrue(res.verification_passed)
        self.assertGreater(res.file_count, 15)

        with zipfile.ZipFile(out_zip, "r") as zf:
            namelist = zf.namelist()
            self.assertIn("train.jsonl", namelist)
            self.assertIn("val.jsonl", namelist)
            self.assertIn("by_persona/metrologist.jsonl", namelist)
            self.assertNotIn(".DS_Store", namelist)

    def test_verify_website_zip_missing_files(self):
        # Create a partial zip missing index.html
        partial_zip = self.test_root / "partial.zip"
        with zipfile.ZipFile(partial_zip, "w") as zf:
            zf.writestr("some_file.txt", "hello")

        v_res = verify_website_zip(partial_zip)
        self.assertTrue(v_res["exists"])
        self.assertTrue(v_res["integrity_ok"])
        self.assertIn("index.html", v_res["missing_key_files"])

    def test_nonexistent_dataset_dir(self):
        with self.assertRaises(FileNotFoundError):
            collect_website_files(self.test_root / "does_not_exist")


if __name__ == "__main__":
    unittest.main()
