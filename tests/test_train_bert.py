import importlib.util
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "05_train_bert.py"
SPEC = importlib.util.spec_from_file_location("train_bert_script", SCRIPT_PATH)
train_bert = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(train_bert)


class TrainBertTests(unittest.TestCase):
    def test_load_jsonl_rows_returns_stable_order(self) -> None:
        rows = [
            {"pkg": "b.tgz", "label": 1, "sequence_text": "zzz"},
            {"pkg": "a.tgz", "label": 0, "sequence_text": "aaa"},
            {"pkg": "a.tgz", "label": 1, "sequence_text": "bbb"},
        ]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.jsonl"
            path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
                encoding="utf-8",
            )

            loaded = train_bert.load_jsonl_rows(str(path))

        self.assertEqual(
            [(row["pkg"], row["label"], row["sequence_text"]) for row in loaded],
            [
                ("a.tgz", 0, "aaa"),
                ("a.tgz", 1, "bbb"),
                ("b.tgz", 1, "zzz"),
            ],
        )

    def test_parse_args_defaults_oversample_to_disabled(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["05_train_bert.py", "--train-jsonl", "train.jsonl", "--test-jsonl", "test.jsonl"],
        ):
            args = train_bert.parse_args()

        self.assertFalse(args.oversample)

    def test_parse_args_enables_oversample_when_requested(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "05_train_bert.py",
                "--train-jsonl",
                "train.jsonl",
                "--test-jsonl",
                "test.jsonl",
                "--oversample",
            ],
        ):
            args = train_bert.parse_args()

        self.assertTrue(args.oversample)


if __name__ == "__main__":
    unittest.main()
