#!/usr/bin/env python3
import json, re, pathlib, argparse

HOOKS = ["preinstall","install","postinstall"]
JS_PATTERNS = [
    re.compile(r"(?:^|\s)(?:node\s+)?(?P<f>[\w./-]+\.js)\b"),
    re.compile(r"\bnode\s+-e\s+"),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pkg = pathlib.Path(args.pkg_dir)
    pkg_json = pkg / "package.json"
    if not pkg_json.exists():
        open(args.out, "w", encoding="utf-8").close(); return

    data = json.loads(pkg_json.read_text(encoding="utf-8", errors="ignore"))
    scripts = data.get("scripts") or {}
    entries = []
    for h in HOOKS:
        s = scripts.get(h)
        if not s: continue
        for pat in JS_PATTERNS:
            for m in pat.finditer(s):
                f = m.groupdict().get("f")
                if f:
                    p = (pkg / f).resolve()
                    if p.exists() and p.suffix == ".js":
                        entries.append(str(p))
    # fallback
    if not entries:
        main_f = (data.get("main") or "").strip()
        if main_f:
            p = (pkg / main_f).resolve()
            if p.exists() and p.suffix == ".js":
                entries = [str(p)]
        else:
            index_js = (pkg / "index.js").resolve()
            if index_js.exists():
                entries = [str(index_js)]

    seen=set(); lines=[]
    for e in entries:
        if e not in seen: seen.add(e); lines.append(e)
    pathlib.Path(args.out).write_text("\n".join(lines), encoding="utf-8")

if __name__ == "__main__":
    main()
