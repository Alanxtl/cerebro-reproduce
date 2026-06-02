import json
import tempfile
import unittest
from pathlib import Path

from standard_pipeline import (
    WINDOWS_PATH_LIMIT,
    _unique_destination,
    build_arg_parser,
    resolve_dataset_archives,
)


class StandardPipelineTests(unittest.TestCase):
    def test_resolve_dataset_archives_reads_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "sample.tgz"
            archive.write_text("x", encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps({"archives": [str(archive)]}),
                encoding="utf-8",
            )

            self.assertEqual(resolve_dataset_archives(manifest_path=manifest, dataset_dir=None), [archive])

    def test_unique_destination_shortens_windows_long_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dest_dir = root / ("nested" * 12) / ("materialized" * 8)
            dest_dir.mkdir(parents=True)
            src = root / ("DataDog__" + ("very-long-package-name-" * 8) + "1.0.0.zip")
            src.write_text("x", encoding="utf-8")

            destination = _unique_destination(dest_dir, src)

            self.assertLessEqual(len(str(destination)), WINDOWS_PATH_LIMIT)
            self.assertEqual(destination.suffix, ".zip")
            self.assertNotEqual(destination.name, "")

    def test_arg_parser_defaults_seed_to_42(self) -> None:
        parser = build_arg_parser()

        args = parser.parse_args(
            [
                "--split-dir",
                "split",
                "--benign-train-dir",
                "benign-train",
                "--benign-test-dir",
                "benign-test",
                "--out-dir",
                "out",
            ]
        )

        self.assertEqual(args.seed, 42)

    def test_arg_parser_defaults_oversample_to_disabled(self) -> None:
        parser = build_arg_parser()

        args = parser.parse_args(
            [
                "--split-dir",
                "split",
                "--benign-train-dir",
                "benign-train",
                "--benign-test-dir",
                "benign-test",
                "--out-dir",
                "out",
            ]
        )

        self.assertFalse(args.oversample)

    def test_arg_parser_enables_oversample_when_requested(self) -> None:
        parser = build_arg_parser()

        args = parser.parse_args(
            [
                "--split-dir",
                "split",
                "--benign-train-dir",
                "benign-train",
                "--benign-test-dir",
                "benign-test",
                "--out-dir",
                "out",
                "--oversample",
            ]
        )

        self.assertTrue(args.oversample)


if __name__ == "__main__":
    unittest.main()
