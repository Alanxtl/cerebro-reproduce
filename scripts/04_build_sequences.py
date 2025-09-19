#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_build_sequences_cg_strict.py

- CG JSON:
  A) schema:
     files: [str,...]
     functions: {"0": "2:4:62:17:2", ...}  # fileIdx:startLine:startCol:endLine:endCol
     calls:     {"8": "0:1:1:1:25", ...}
     call2fun:  [[22,2], ...]
     fun2fun:   [[3,2], ...]

Usage:
python scripts/04_build_sequences_cg_strict.py \
  --pkg_name somepkg-1.0.0.tgz \
  --pkg_dir  /abs/path/to/unpacked/pkg \
  --entries  ./outputs/entries.txt \
  --cg       ./outputs/logs/<sample>/cg.json \
  --sequence_out ./outputs/sequence.jsonl \
  --long-threshold 50 \
  --scan-node-modules false
"""

import argparse
import json
import os
import pathlib
import re
import gc
from typing import Dict, Any, List, Tuple, Optional, Set

import tree_sitter_javascript as jsts
import tree_sitter_typescript as tsts
from tree_sitter import Language, Parser

# =========================
# Dimensions (Table 2)
# =========================
DIM = {
    "R1": "import operating system module",
    "R2": "use operating system module call",
    "R3": "import file system module",
    "R4": "use file system module call",
    "R5": "read sensitive information",
    "D1": "import network module",
    "D2": "use network module call",
    "D3": "use URL",
    "E1": "import encoding module",
    "E2": "use encoding module call",
    "E3": "use base64 string",
    "E4": "use long string",
    "P1": "import process module",
    "P2": "use process module call",
    "P3": "use bash script",
    "P4": "evaluate code at run-time",
}

# =========================
# Heuristics / Patterns
# =========================
OS_CORE   = {"os"}
FS_CORE   = {"fs", "path"}
NET_CORE  = {"http", "https", "net", "tls", "dgram", "dns", "url"}
ENC_CORE  = {"crypto", "zlib", "buffer", "querystring"}

NET_LIBS  = {
    "axios","node-fetch","cross-fetch","isomorphic-fetch","request","got",
    "superagent","needle","undici","ws","websocket","websocket-stream"
}
ENC_LIBS  = {"iconv","iconv-lite","base64-js","js-base64","pako"}
PROCESS_PKG = {"process"}

SHELL_TOKENS = [
    r"\bsh\b", r"\bbash\b", r"\bzsh\b", r"\bcmd\.exe\b", r"\bpowershell\b",
    r"\bsh\s+-c\b", r"\bbash\s+-c\b", r"[|&;><`]"
]
BASE64_RE_CHARS = re.compile(r"^[A-Za-z0-9+/=\s]+$")
URL_SCHEMES = re.compile(r"^(?:https?|data|ftp|ws|wss):", re.I)


def compile_shell_re() -> re.Pattern:
    return re.compile("|".join(SHELL_TOKENS), re.I)

# =========================
# Tree-sitter setup
# =========================
JS_LANGUAGE = Language(jsts.language()) 
TS_LANGUAGE = Language(tsts.language_typescript())

def set_lang_by_suffix(path: pathlib.Path):
    if path.suffix.lower() in (".ts", ".tsx"):
        return Parser(TS_LANGUAGE)
    else:
        return Parser(JS_LANGUAGE)

def code_text(src: bytes, node) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", "ignore")

def norm_string(src: bytes, node) -> str:
    return code_text(src, node).strip().strip("\"'`")

# =========================
# Path normalization
# =========================
def canon(p: str | pathlib.Path) -> str:
    try:
        s = str(pathlib.Path(p).resolve())
    except Exception:
        s = str(p)
    s = os.path.normpath(s)
    s = os.path.normcase(s)
    return s

def variants(p: str | pathlib.Path) -> List[str]:
    c = canon(p)
    back = c.replace("/", "\\")
    fwd  = c.replace("\\", "/")
    out, seen = [], set()
    for x in (c, back, fwd):
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def normalize_path_rel(base: pathlib.Path, p: str) -> str:
    try:
        pp = pathlib.Path(p)
        if not pp.is_absolute():
            pp = (base / pp).resolve()
        return canon(pp)
    except Exception:
        return canon(base / p)

# =========================
# CG schema helpers / loaders
# =========================
def decode_loc_str(s: str) -> Tuple[int, int, int, int, int]:
    """process 'fileIdx:startLine:startCol:endLine:endCol' """
    parts = s.split(":")
    if len(parts) != 5:
        return (0, 0, 0, 10**9, 0)
    fi, sl, sc, el, ec = parts
    return int(fi), int(sl), int(sc), int(el), int(ec)

def load_cg_jelly_indexed(cg: Dict[str, Any], pkg_dir: pathlib.Path) -> Optional[Dict[str, Any]]:
    if not isinstance(cg.get("files"), list): return None
    if not isinstance(cg.get("functions"), dict): return None
    if not isinstance(cg.get("calls"), dict): return None

    files = [normalize_path_rel(pkg_dir, s) for s in cg["files"]]

    fun_by_id: Dict[int, Dict[str, Any]] = {}
    funs_by_file: Dict[str, List[Dict[str, Any]]] = {}
    for k, v in cg["functions"].items():
        fid = int(k)
        fi, sl, sc, el, ec = decode_loc_str(v)
        file = files[fi] if 0 <= fi < len(files) else ""
        rec = {"id": fid, "file": file, "start": sl, "end": el}
        fun_by_id[fid] = rec
        if file:
            for key in variants(file):
                funs_by_file.setdefault(key, []).append(rec)

    call_by_id: Dict[int, Dict[str, Any]] = {}
    calls_by_file_line: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}
    calls_by_file_all: Dict[str, List[Dict[str, Any]]] = {}
    for k, v in cg["calls"].items():
        cid = int(k)
        fi, sl, sc, el, ec = decode_loc_str(v)
        file = files[fi] if 0 <= fi < len(files) else ""
        rec = {"id": cid, "file": file, "sline": sl, "scol": sc, "eline": el, "ecol": ec}
        call_by_id[cid] = rec
        if file:
            for key in variants(file):
                calls_by_file_line.setdefault(key, {}).setdefault(sl, []).append(rec)
                calls_by_file_all.setdefault(key, []).append(rec)

    c2f: Dict[int, List[int]] = {}
    for pair in cg.get("call2fun", []):
        try:
            c, f = int(pair[0]), int(pair[1])
            c2f.setdefault(c, []).append(f)
        except Exception:
            continue

    f2f: Dict[int, List[int]] = {}
    for pair in cg.get("fun2fun", []):
        try:
            fr, to = int(pair[0]), int(pair[1])
            f2f.setdefault(fr, []).append(to)
        except Exception:
            continue

    def find_fun_by_pos(file: str, line: int) -> Optional[int]:
        arr = funs_by_file.get(file, [])
        for f in arr:
            if f["start"] <= line <= f["end"]:
                return f["id"]
        return None

    return {
        "fun_by_id": fun_by_id,
        "funs_by_file": funs_by_file,
        "call_by_id": call_by_id,
        "calls_by_file_line": calls_by_file_line,
        "calls_by_file_all": calls_by_file_all,
        "call2fun": c2f,
        "fun2fun": f2f,
        "find_fun_by_pos": find_fun_by_pos,
    }

def load_cg_generic(cg_path: pathlib.Path, pkg_dir: pathlib.Path) -> Optional[Dict[str, Any]]:
    try:
        raw = json.loads(cg_path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None

    idx = load_cg_jelly_indexed(raw, pkg_dir)
    if idx:
        return idx

    def _get_list(o, k): return o.get(k) if isinstance(o.get(k), list) else None
    def _get_dict(o, k): return o.get(k) if isinstance(o.get(k), dict) else None

    functions = _get_list(raw, "functions") or _get_list(_get_dict(raw, "nodes") or {}, "functions") or []
    calls     = _get_list(raw, "calls")     or _get_list(_get_dict(raw, "nodes") or {}, "calls")     or []
    edges     = _get_dict(raw, "edges") or {}

    call2fun = edges.get("call2fun") or []
    fun2fun  = edges.get("fun2fun")  or []

    fun_by_id: Dict[Any, Dict[str, Any]] = {}
    funs_by_file: Dict[str, List[Dict[str, Any]]] = {}
    for f in functions:
        fid = f.get("id") or f.get("fid") or f.get("_id")
        file = f.get("file") or f.get("path") or (f.get("loc") or {}).get("file") or ""
        if file:
            file = normalize_path_rel(pkg_dir, file)
        start = f.get("startLine") or (f.get("loc") or {}).get("startLine") or f.get("line") or 0
        end   = f.get("endLine") or (f.get("loc") or {}).get("endLine") or 10**9
        if fid is None:
            continue
        rec = {"id": fid, "file": file, "start": int(start), "end": int(end)}
        fun_by_id[fid] = rec
        if file:
            for key in variants(file):
                funs_by_file.setdefault(key, []).append(rec)

    call_by_id: Dict[Any, Dict[str, Any]] = {}
    calls_by_file_line: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}
    calls_by_file_all: Dict[str, List[Dict[str, Any]]] = {}
    for c in calls:
        cid = c.get("id") or c.get("cid") or c.get("_id")
        file = c.get("file") or c.get("path") or (c.get("loc") or {}).get("file") or ""
        if file:
            file = normalize_path_rel(pkg_dir, file)
        sl = c.get("line") or (c.get("loc") or {}).get("line") or (c.get("position") or {}).get("line") or 0
        sc = c.get("column") or (c.get("loc") or {}).get("column") or (c.get("position") or {}).get("column") or 0
        el = (c.get("endLine") or (c.get("loc") or {}).get("endLine") or sl)
        ec = (c.get("endColumn") or (c.get("loc") or {}).get("endColumn") or sc)
        if cid is None:
            continue
        rec = {"id": cid, "file": file, "sline": int(sl), "scol": int(sc), "eline": int(el), "ecol": int(ec)}
        call_by_id[cid] = rec
        if file:
            for key in variants(file):
                calls_by_file_line.setdefault(key, {}).setdefault(int(sl), []).append(rec)
                calls_by_file_all.setdefault(key, []).append(rec)

    c2f: Dict[Any, List[Any]] = {}
    for e in call2fun:
        if isinstance(e, dict):
            cid = e.get("call") or e.get("from") or e.get("src")
            fid = e.get("fun")  or e.get("to")   or e.get("dst")
        else:
            try:
                cid, fid = e[0], e[1]
            except Exception:
                continue
        if cid is None or fid is None:
            continue
        c2f.setdefault(cid, []).append(fid)

    f2f: Dict[Any, List[Any]] = {}
    for e in fun2fun:
        if isinstance(e, dict):
            fr = e.get("from") or e.get("src") or e.get("caller")
            to = e.get("to")   or e.get("dst") or e.get("callee")
        else:
            try:
                fr, to = e[0], e[1]
            except Exception:
                continue
        if fr is None or to is None:
            continue
        f2f.setdefault(fr, []).append(to)

    def find_fun_by_pos(file: str, line: int) -> Optional[Any]:
        arr = funs_by_file.get(file, [])
        for f in arr:
            if f["start"] <= line <= f["end"]:
                return f["id"]
        return None

    return {
        "fun_by_id": fun_by_id,
        "funs_by_file": funs_by_file,
        "call_by_id": call_by_id,
        "calls_by_file_line": calls_by_file_line,
        "calls_by_file_all": calls_by_file_all,
        "call2fun": c2f,
        "fun2fun": f2f,
        "find_fun_by_pos": find_fun_by_pos,
    }

# =========================
# Range overlap helpers
# =========================
def _pt(line: int, col: int) -> tuple[int, int]:
    return (line, col)

def _le(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[0] or (a[0] == b[0] and a[1] <= b[1])

def _overlap(a_start: tuple[int, int], a_end: tuple[int, int],
             b_start: tuple[int, int], b_end: tuple[int, int]) -> bool:
    if _le(a_end, (b_start[0], b_start[1] - 1)):
        return False
    if _le(b_end, (a_start[0], a_start[1] - 1)):
        return False
    return True

def resolve_calls_for_node(file_s: str, node, src: bytes, cg_idx: Dict[str, Any]) -> List[Any]:
    """check (sline,scol,eline,ecol) overlap"""
    sline = node.start_point[0] + 1
    scol  = node.start_point[1]
    eline = node.end_point[0] + 1
    ecol  = node.end_point[1]
    a_start = _pt(sline, scol)
    a_end   = _pt(eline, ecol)

    out: List[Any] = []
    for key in variants(file_s):
        for rec in cg_idx.get("calls_by_file_all", {}).get(key, []):
            b_start = _pt(rec["sline"], rec["scol"])
            b_end   = _pt(rec["eline"], rec["ecol"])
            if _overlap(a_start, a_end, b_start, b_end):
                out.append(rec["id"])
        if out:
            break
    return list(dict.fromkeys(out))

# =========================
# Feature extraction (AST)
# =========================
def classify_import_to_dim(pkg: str) -> str:
    p = pkg.strip()
    # --- Process family ---
    if p in {"process", "child_process"}:
        return "P1"

    # --- OS / FS ---
    if p in OS_CORE: return "R1"
    if p in FS_CORE: return "R3"

    # --- Network / Encoding ---
    if p in NET_CORE or p in NET_LIBS: return "D1"
    if p in ENC_CORE or p in ENC_LIBS: return "E1"

    # --- Heuristic substrings ---
    if "http" in p or "https" in p or "fetch" in p or "socket" in p: return "D1"
    if "crypto" in p or "encode" in p or "decod" in p or "iconv" in p or "base64" in p or "zlib" in p: return "E1"
    if p == "url": return "D1"
    return ""


def is_url_like(s: str) -> bool:
    return ("http://" in s) or ("https://" in s) or bool(URL_SCHEMES.match(s))

def is_base64_like(s: str, min_len: int) -> bool:
    s2 = re.sub(r"\s+", "", s)
    if len(s2) < min_len:
        return False
    if not BASE64_RE_CHARS.match(s):
        return False
    if s2.endswith("=") and len(re.findall(r"=", s2)) <= 2:
        return True
    return True  # all no padding

def add_dim(seq: List[str], key: str):
    seq.append(DIM[key])

def scan_function_or_file(file_path: pathlib.Path, node, src: bytes, seq: List[str],
                          long_threshold: int, shell_re: re.Pattern):
    st = [node]
    while st:
        n = st.pop()
        for c in reversed(n.children or []):
            st.append(c)
        t = n.type
        code = code_text(src, n)

        # ESM import
        if t == "import_declaration":
            for ch in n.children:
                if ch.type == "string":
                    mod = norm_string(src, ch)
                    dim = classify_import_to_dim(mod)
                    if dim: add_dim(seq, dim)

        # require('x') / import('x') 
        if t == "call_expression":
            m = re.search(r"(?:require|import)\s*\(\s*(['\"])(.*?)\1\s*\)", code)
            if m:
                dim = classify_import_to_dim(m.group(2))
                if dim: add_dim(seq, dim)

            # OS/FS/NET/ENC/PROC/SENSITIVE/EVAL
            if re.search(r"\bos\.\w+\s*\(", code): add_dim(seq, "R2")
            if re.search(r"\bfs\.\w+\s*\(", code): add_dim(seq, "R4")
            if re.search(r"\bhttps?\.(request|get|Agent|createServer)\s*\(", code) or \
               re.search(r"\b(net|tls|dgram|dns)\.\w+\s*\(", code) or \
               re.search(r"\b(axios|fetch|got|request|superagent|needle|undici)\b", code) or \
               re.search(r"\bnew\s+(WebSocket|WS)\s*\(", code):
                add_dim(seq, "D2")
            if re.search(r"\b(Buffer\.from\(.+?,\s*(['\"])base64\2\)|\.toString\(\s*(['\"])base64\3\)|\batob\s*\(|\bbtoa\s*\(|\bnew\s+TextEncoder\s*\(|\bnew\s+TextDecoder\s*\(|\bcrypto\.(createCipher|createCipheriv|createDecipher|createDecipheriv|createHash|createHmac)\s*\(|\bzlib\.(gzip|gunzip|deflate|inflate|brotliCompress|brotliDecompress)\s*\()", code, re.I):
                add_dim(seq, "E2")
            if re.search(r"\bprocess\.(env|argv|versions|cwd|chdir|exit|kill|pid|ppid|umask|uptime|memoryUsage)\b", code) or \
               re.search(r"\bchild_process\.(exec|execFile|spawn|fork)\s*\(", code):
                add_dim(seq, "P2")
            if re.search(r"\bprocess\.env(\.|\[)", code) or \
               re.search(r"\bos\.(userInfo|homedir|tmpdir)\s*\(", code) or \
               re.search(r"(/etc/passwd|id_rsa|\.npmrc|\.ssh/|AppData\\Roaming|%APPDATA%)", code):
                add_dim(seq, "R5")
            if re.search(r"\beval\s*\(|\bnew\s+Function\s*\(|\bFunction\s*\(|\bvm\.(runInNewContext|runInContext|runInThisContext)\s*\(|\bnew\s+vm\.Script\s*\(|\bset(?:Timeout|Interval)\s*\(\s*(['\"])", code):
                add_dim(seq, "P4")

            # bash-like via child_process
            if re.search(r"\bchild_process\.(exec|execFile|spawn|fork)\s*\(", code):
                if re.search(r"shell\s*:\s*true", code) or shell_re.search(code):
                    add_dim(seq, "P3")

            # new URL(...)
            if re.search(r"\bnew\s+URL\s*\(", code):
                add_dim(seq, "D3")

            # ---- NEW: chained require('x').method(...) ----
            # e.g., require("child_process").fork(...),
            #       require("https").get(...), require("os").hostname()
            m2 = re.search(r"require\s*\(\s*(['\"])(?P<mod>[^'\"]+)\1\s*\)\s*\.\s*(?P<meth>[A-Za-z_]\w*)\s*\(", code)
            if m2:
                mod  = m2.group("mod")
                meth = m2.group("meth")

                if mod in {"os"}:
                    add_dim(seq, "R2")
                if mod in {"fs", "path"}:
                    add_dim(seq, "R4")
                if mod in {"http", "https", "net", "tls", "dgram", "dns", "url"}:
                    add_dim(seq, "D2")
                if mod in {"crypto", "zlib", "buffer", "querystring"}:
                    add_dim(seq, "E2")
                if mod in {"process", "child_process"}:
                    add_dim(seq, "P2")
                    if re.search(r"\bexec\s*\(", code):
                        if re.search(r"shell\s*:\s*true", code) or shell_re.search(code):
                            add_dim(seq, "P3")


        # string
        if t in ("string", "template_string", "string_fragment"):
            s = norm_string(src, n)
            if is_url_like(s): add_dim(seq, "D3")
            if is_base64_like(s, max(40, long_threshold)): add_dim(seq, "E3")
            if len(s) >= long_threshold: add_dim(seq, "E4")

        # member expr：process.env / os.userInfo()
        if t == "member_expression":
            if re.search(r"\bprocess\.env(\.|\[)", code) or \
               re.search(r"\bos\.(userInfo|homedir|tmpdir)\s*\(", code):
                add_dim(seq, "R5")
            if re.search(r"\bprocess\.(env|argv|versions|cwd|chdir|exit|kill|pid|ppid|umask|uptime|memoryUsage)\b", code):
                add_dim(seq, "P2")

# =========================
# Traversal (CG-driven DFS)
# =========================
def build_sequence_with_cg(pkg_dir: pathlib.Path,
                           entries: List[pathlib.Path],
                           cg_idx: Dict[str, Any],
                           long_threshold: int,
                           scan_node_modules: bool) -> List[str]:
    shell_re = compile_shell_re()
    seq: List[str] = []
    visited_fun: Set[Any] = set()

    def skip(p: pathlib.Path) -> bool:
        return (not scan_node_modules) and ("node_modules" in p.parts)

    def visit_function(fid: Any):
        if fid in visited_fun:
            return
        visited_fun.add(fid)
        f = cg_idx["fun_by_id"].get(fid)
        if not f or not f["file"]:
            return
        fpath = pathlib.Path(f["file"])
        if not fpath.exists() or skip(fpath):
            return

        try:
            src = fpath.read_bytes()
        except Exception:
            return
        
        tree = set_lang_by_suffix(fpath).parse(src)
        root = tree.root_node

        def in_fun(n) -> bool:
            ln = n.start_point[0] + 1
            return f["start"] <= ln <= f["end"]

        st = [root]
        while st:
            n = st.pop()
            for c in reversed(n.children or []):
                st.append(c)
            if not in_fun(n):
                continue

            scan_function_or_file(fpath, n, src, seq, long_threshold, shell_re)

            # first call2fun，second fun2fun
            if n.type == "call_expression":
                cids = resolve_calls_for_node(str(fpath.resolve()), n, src, cg_idx)
                jumped = False
                for cid in cids:
                    for callee in cg_idx["call2fun"].get(cid, []):
                        visit_function(callee)
                        jumped = True
                if not jumped:
                    for to in cg_idx["fun2fun"].get(fid, []):
                        visit_function(to)

        for to in cg_idx["fun2fun"].get(fid, []):
            visit_function(to)

    # start from entries
    for ent in entries:
        if not ent.exists() or skip(ent):
            continue
        try:
            src = ent.read_bytes()
        except Exception:
            continue
        
        tree = set_lang_by_suffix(ent).parse(src)
        root = tree.root_node

        scan_function_or_file(ent, root, src, seq, long_threshold, shell_re)

        # call2fun → fun2fun 
        st = [root]
        while st:
            n = st.pop()
            for c in reversed(n.children or []):
                st.append(c)
            if n.type != "call_expression":
                continue

            cids = resolve_calls_for_node(str(ent.resolve()), n, src, cg_idx)
            jumped = False
            for cid in cids:
                for callee in cg_idx["call2fun"].get(cid, []):
                    visit_function(callee)
                    jumped = True
            if not jumped:
                fid = cg_idx["find_fun_by_pos"](canon(str(ent.resolve())), n.start_point[0] + 1) \
                      if "find_fun_by_pos" in cg_idx else None
                if fid is not None:
                    for to in cg_idx["fun2fun"].get(fid, []):
                        visit_function(to)

    del src, tree, root, st
    return seq

def build_sequence_entries_only(entries: List[pathlib.Path],
                                long_threshold: int,
                                scan_node_modules: bool) -> List[str]:
    shell_re = compile_shell_re()
    seq: List[str] = []
    for ent in entries:
        if not ent.exists():
            continue
        if (not scan_node_modules) and ("node_modules" in ent.parts):
            continue
        try:
            src = ent.read_bytes()
        except Exception:
            continue
        
        tree = set_lang_by_suffix(ent).parse(src)
        root = tree.root_node
        scan_function_or_file(ent, root, src, seq, long_threshold, shell_re)
        del src, tree, root
    return seq

# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg_name", required=True)
    ap.add_argument("--pkg_dir", required=True, help="Unpacked package root (resolve CG file indices here)")
    ap.add_argument("--entries", required=True)
    ap.add_argument("--cg", required=True)
    ap.add_argument("--sequence_out", required=True)
    ap.add_argument("--long-threshold", type=int, default=50)
    ap.add_argument("--scan-node-modules", type=str, default="false")
    ap.add_argument("--label", type=int, required=True)
    args = ap.parse_args()

    pkg_dir = pathlib.Path(args.pkg_dir).resolve()
    entries = [pathlib.Path(l.strip()) for l in pathlib.Path(args.entries).read_text(encoding="utf-8").splitlines() if l.strip()]
    scan_nm = args.scan_node_modules.strip().lower() == "true"

    seq: List[str] = []
    cg_path = pathlib.Path(args.cg)

    if cg_path.exists():
        cg_idx = load_cg_generic(cg_path, pkg_dir)
        if cg_idx and cg_idx.get("call2fun") and cg_idx.get("calls_by_file_all"):
            seq = build_sequence_with_cg(pkg_dir, entries, cg_idx, args.long_threshold, scan_nm)

    if not seq:
        # fallback
        seq = build_sequence_entries_only(entries, args.long_threshold, scan_nm)

    if seq:
        with open(args.sequence_out, "a", encoding="utf-8") as fw:
            fw.write(json.dumps({"pkg": args.pkg_name, 
                                 "sequence_text": ", ".join(seq),
                                 "label": args.label
                                 }, ensure_ascii=False) + "\n")
    del seq
    gc.collect()

if __name__ == "__main__":
    main()
