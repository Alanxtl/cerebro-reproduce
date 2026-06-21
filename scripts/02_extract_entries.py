#!/usr/bin/env python3
import json, re, pathlib, argparse

HOOKS = ["preinstall", "install", "postinstall"]
SOURCE_EXTS = {".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"}
JS_PATTERNS = [
    re.compile(r"(?:^|\s)(?:node\s+)?(?P<f>[\w./-]+\.js)\b"),
    re.compile(r"\bnode\s+-e\s+"),
]
IMPORT_PATTERNS = [
    re.compile(r"\brequire\s*\(\s*['\"](?P<path>\.{1,2}/[^'\"]+)['\"]\s*\)"),
    re.compile(r"\bimport\s+(?:[^'\"]+\s+from\s+)?['\"](?P<path>\.{1,2}/[^'\"]+)['\"]"),
    re.compile(r"\bimport\s*\(\s*['\"](?P<path>\.{1,2}/[^'\"]+)['\"]\s*\)"),
]


def iter_source_files(pkg: pathlib.Path):
    for path in sorted(pkg.rglob("*")):
        if not path.is_file():
            continue
        if "node_modules" in path.parts:
            continue
        if path.suffix.lower() in SOURCE_EXTS:
            yield path.resolve()


def resolve_relative_module(base_file: pathlib.Path, module_path: str) -> pathlib.Path | None:
    raw = (base_file.parent / module_path).resolve()
    candidates = []
    if raw.suffix:
        candidates.append(raw)
    else:
        candidates.extend(raw.with_suffix(ext) for ext in sorted(SOURCE_EXTS))
        candidates.extend(raw / f"index{ext}" for ext in sorted(SOURCE_EXTS))
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return None


def collect_relative_imports(entries: list[pathlib.Path]) -> list[pathlib.Path]:
    seen = set()
    ordered = []
    queue = list(entries)
    while queue:
        entry = queue.pop(0).resolve()
        if entry in seen or not entry.exists():
            continue
        seen.add(entry)
        ordered.append(entry)
        try:
            text = entry.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for pattern in IMPORT_PATTERNS:
            for match in pattern.finditer(text):
                resolved = resolve_relative_module(entry, match.group("path"))
                if resolved is not None and resolved not in seen:
                    queue.append(resolved)
    return ordered


def fallback_entries(pkg: pathlib.Path, main_file: str = "") -> list[str]:
    entries = []
    if main_file:
        p = (pkg / main_file).resolve()
        if p.exists() and p.suffix.lower() in SOURCE_EXTS:
            entries.append(p)
    index_js = (pkg / "index.js").resolve()
    if index_js.exists():
        entries.append(index_js)
    if not entries:
        entries.extend(iter_source_files(pkg))
    else:
        entries = collect_relative_imports(entries)
    return [str(path) for path in entries]


def unique_entries(entries: list[str]) -> list[str]:
    seen = set()
    out = []
    for entry in entries:
        if entry in seen:
            continue
        seen.add(entry)
        out.append(entry)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pkg = pathlib.Path(args.pkg_dir)
    pkg_json = pkg / "package.json"
    if not pkg_json.exists():
        pathlib.Path(args.out).write_text(
            "\n".join(fallback_entries(pkg)),
            encoding="utf-8",
        )
        return

    data = json.loads(pkg_json.read_text(encoding="utf-8", errors="ignore"))
    scripts = data.get("scripts") or {}
    entries = []
    for h in HOOKS:
        s = scripts.get(h)
        if not s:
            continue
        for pat in JS_PATTERNS:
            for m in pat.finditer(s):
                f = m.groupdict().get("f")
                if f:
                    p = (pkg / f).resolve()
                    if p.exists() and p.suffix == ".js":
                        entries.append(str(p))
    # fallback
    if not entries:
        entries = fallback_entries(pkg, (data.get("main") or "").strip())

    lines = unique_entries(entries)
    pathlib.Path(args.out).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
