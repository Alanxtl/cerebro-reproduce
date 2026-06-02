import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_all import WINDOWS_PATH_LIMIT, _bounded_archive_key


class RunAllTests(unittest.TestCase):
    def test_bounded_archive_key_shortens_windows_long_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / ("nested" * 12) / ("logs" * 8)
            parent.mkdir(parents=True)
            archive = root / ("DataDog__" + ("very-long-package-name-" * 8) + "1.0.0.tgz")
            archive.write_text("x", encoding="utf-8")

            key = _bounded_archive_key(archive, parent)

            self.assertLessEqual(len(str(parent / key)), WINDOWS_PATH_LIMIT)
            self.assertTrue(key)
            self.assertIn("_", key)


if __name__ == "__main__":
    unittest.main()
