import tempfile
import unittest
from pathlib import Path

import importlib.util
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_all import WINDOWS_PATH_LIMIT, _bounded_archive_key, find_pkg_dir

EXTRACT_ENTRIES_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "02_extract_entries.py"
)
SPEC = importlib.util.spec_from_file_location(
    "extract_entries_script", EXTRACT_ENTRIES_PATH
)
extract_entries = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(extract_entries)


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

    def test_find_pkg_dir_accepts_package_without_package_json_when_index_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "unpacked"
            package.mkdir()
            (package / "index.js").write_text(
                "const os = require('os'); console.log(os.hostname());",
                encoding="utf-8",
            )

            self.assertEqual(find_pkg_dir(package), package)

    def test_fallback_entries_lists_index_and_relative_imports_without_package_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp)
            index_js = package / "index.js"
            index_js.write_text("require('./lib/payload');", encoding="utf-8")
            (package / "lib").mkdir()
            payload_js = package / "lib" / "payload.js"
            payload_js.write_text("module.exports = () => process.cwd();", encoding="utf-8")

            entries = extract_entries.fallback_entries(package)

        self.assertEqual(entries, [str(index_js.resolve()), str(payload_js.resolve())])

    def test_fallback_entries_lists_source_files_without_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp)
            src = package / "lib"
            src.mkdir()
            payload_js = src / "payload.js"
            payload_js.write_text("module.exports = () => process.cwd();", encoding="utf-8")

            entries = extract_entries.fallback_entries(package)

        self.assertEqual(entries, [str(payload_js.resolve())])


if __name__ == "__main__":
    unittest.main()
