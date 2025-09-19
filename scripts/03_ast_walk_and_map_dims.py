#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stronger AST feature extractor for Cerebro-style dimensions.

- Supports CJS require, ESM import, dynamic import()
- Covers Node core + common 3rd-party network libs
- Richer runtime-eval, encoding, process/os/fs/http/https/etc.
- Detects bash-like shell execution via child_process and command strings
"""

import argparse
import json
import pathlib
import re
from typing import Dict, Any, List

import tree_sitter_javascript as jsts
import tree_sitter_typescript as tsts
from tree_sitter import Language, Parser

# ----------------------------
# Configuration (can be tuned)
# ----------------------------

# Node core modules by category
OS_CORE = {"os"}
FS_CORE = {"fs", "path"} 
NET_CORE = {"http", "https", "net", "tls", "dgram", "dns", "url"}
ENC_CORE = {"crypto", "zlib", "buffer", "querystring"}

# Popular 3rd-party network libs (D1/D2)
NET_THIRD_PARTY = {
    "axios", "node-fetch", "cross-fetch", "isomorphic-fetch",
    "request", "got", "superagent", "needle", "undici",
    "ws", "websocket", "websocket-stream",
}

# Encoding-related userland libs (E1/E2) 
ENC_THIRD_PARTY = {
    "iconv", "iconv-lite", "base64-js", "js-base64", "pako",
}

# Process module import name
PROCESS_PKG = {"process", "child_process"}  # require('process')

# Shell-ish indicators in command strings (P3)
SHELL_TOKENS = [
    r"\bsh\b", r"\bbash\b", r"\bzsh\b", r"\bcmd\.exe\b", r"\bpowershell\b",
    r"\bsh\s+-c\b", r"\bbash\s+-c\b",
    r"[|&;><`]"  # pipe/and/or/redirect/backtick
]

BASE64_MIN_LEN_DEFAULT = 40  # base64 min length

# ----------------------------
# Dimension dictionary
# ----------------------------
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

# ----------------------------
# Helpers
# ----------------------------

def compile_shell_re():
    return re.compile("|".join(SHELL_TOKENS), re.I)

def is_base64_like(s: str, min_len: int) -> bool:
    # long + Base64 charset + padding
    if len(s) < max(min_len, 10):
        return False
    if not re.fullmatch(r"[A-Za-z0-9+/=\s]+", s):
        return False
    dense = re.sub(r"\s+", "", s)
    if len(dense) < min_len:
        return False
    if dense.endswith("=") and not re.search(r"={3,}$", dense):
        return True
    return True

def is_url_like(s: str) -> bool:
    if re.search(r"https?://", s, re.I):
        return True
    # data: or ftp: -> URL 
    if re.match(r"(?:data|ftp|ws|wss):", s, re.I):
        return True
    return False

def norm_string(src: bytes, node) -> str:
    text = src[node.start_byte:node.end_byte].decode("utf-8", "ignore")
    return text.strip().strip("\"'`")

def code_text(src: bytes, node) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", "ignore")

def add_feat(feats: List[Dict[str, Any]], file_path: pathlib.Path, node, dim_key: str, src: bytes):
    feats.append({
        "file": str(file_path),
        "line": node.start_point[0] + 1,
        "dim": dim_key,
        "dim_text": DIM[dim_key],
        "snippet": code_text(src, node)[:200],
    })

# ----------------------------
# Import / require name extraction
# ----------------------------

def get_import_source_from_import_decl(src: bytes, node) -> str:
    # import ... from 'xxx'
    # import 'xxx'
    for ch in node.children:
        if ch.type in ("string",):
            return norm_string(src, ch)
    return ""

def get_require_arg_from_call(src: bytes, node) -> str:
    # require('xxx') / import('xxx')
    text = code_text(src, node)
    m = re.search(r"(?:require|import)\s*\(\s*(['\"])(.*?)\1\s*\)", text)
    return m.group(2) if m else ""

# ----------------------------
# Category tests
# ----------------------------

def classify_import_to_dim(pkg: str):
    if pkg in OS_CORE:
        return "R1"
    if pkg in FS_CORE:
        return "R3"
    if pkg in NET_CORE or pkg in NET_THIRD_PARTY:
        return "D1"
    if pkg in ENC_CORE or pkg in ENC_THIRD_PARTY:
        return "E1"
    if pkg in PROCESS_PKG:
        return "P1"
    # @aws-sdk/node-http-handler 
    if "http" in pkg or "https" in pkg or "fetch" in pkg or "socket" in pkg:
        return "D1"
    if "crypto" in pkg or "encode" in pkg or "decod" in pkg or "iconv" in pkg or "base64" in pkg or "zlib" in pkg:
        return "E1"
    if pkg == "url":
        return "D1"  # url 
    return ""

def is_os_call(code: str) -> bool:
    return bool(re.search(r"\bos\.\w+\s*\(", code))

def is_fs_call(code: str) -> bool:
    return bool(re.search(r"\bfs\.\w+\s*\(", code))

def is_net_call(code: str) -> bool:
    if re.search(r"\bhttps?\.(request|get|Agent|createServer)\s*\(", code):
        return True
    if re.search(r"\b(net|tls|dgram|dns)\.\w+\s*\(", code):
        return True
    # axios/got/request/fetch/superagent/needle/undici
    if re.search(r"\b(axios|fetch|got|request|superagent|needle|undici)\b", code):
        return True
    # ws/websocket constructors
    if re.search(r"\bnew\s+(WebSocket|WS)\s*\(", code):
        return True
    return False

def is_enc_call(code: str) -> bool:
    # Buffer / atob / btoa / TextEncoder / crypto / zlib
    if re.search(r"\bBuffer\.from\s*\(\s*.+?,\s*(['\"])base64\1\s*\)", code):
        return True
    if re.search(r"\.toString\s*\(\s*(['\"])base64\1\s*\)", code):
        return True
    if re.search(r"\b(atob|btoa)\s*\(", code):
        return True
    if re.search(r"\bnew\s+TextEncoder\s*\(|\bnew\s+TextDecoder\s*\(", code):
        return True
    if re.search(r"\bcrypto\.(createCipher|createCipheriv|createDecipher|createDecipheriv|createHash|createHmac)\s*\(", code, re.I):
        return True
    if re.search(r"\bzlib\.(gzip|gunzip|deflate|inflate|brotliCompress|brotliDecompress)\s*\(", code):
        return True
    return False

def is_process_call(code: str) -> bool:
    # process API 
    if re.search(r"\bprocess.*\.(env|argv|versions|cwd|chdir|exit|kill|pid|ppid|umask|uptime|memoryUsage)\b", code):
        return True
    if re.search(r"child_process.*\.(exec|execFile|spawn|fork)\s*\(", code):
        return True
    return False

def is_sensitive_info(code: str) -> bool:
    # read of env_var, userinfo, homedir, tempdir
    if re.search(r"\bprocess\.env(\.|\[)", code):
        return True
    if re.search(r"\bos\.(userInfo|homedir|tmpdir)\s*\(", code):
        return True
    if re.search(r"(/etc/passwd|id_rsa|\.npmrc|\.ssh/|AppData\\Roaming|%APPDATA%)", code):
        return True
    return False

def is_runtime_eval(code: str) -> bool:
    if re.search(r"\beval\s*\(", code):
        return True
    if re.search(r"\bnew\s+Function\s*\(", code):
        return True
    if re.search(r"\bFunction\s*\(", code):
        return True
    # vm.runInNewContext / vm.Script
    if re.search(r"\bvm\.(runInNewContext|runInContext|runInThisContext)\s*\(", code):
        return True
    if re.search(r"\bnew\s+vm\.Script\s*\(", code):
        return True
    # setTimeout/Interval
    if re.search(r"\bset(?:Timeout|Interval)\s*\(\s*(['\"])", code):
        return True
    return False

def is_bash_like_command(code: str, shell_re: re.Pattern) -> bool:
    # child_process 
    return bool(shell_re.search(code))

# ----------------------------
# Core scanner
# ----------------------------

JS_LANGUAGE = Language(jsts.language()) 
TS_LANGUAGE = Language(tsts.language_typescript())


def scan_file(file_path: pathlib.Path, long_threshold: int, shell_re: re.Pattern) -> List[Dict[str, Any]]:
    feats: List[Dict[str, Any]] = []
    try:
        src = file_path.read_bytes()
    except Exception:
        return feats

    if file_path.suffix.lower() in (".ts", ".tsx"):
        PARSER = Parser(JS_LANGUAGE)
    else:
        PARSER = Parser(TS_LANGUAGE)

    tree = PARSER.parse(src)
    root = tree.root_node

    # DFS
    stack = [root]
    while stack:
        n = stack.pop()
        for c in reversed(n.children or []):
            stack.append(c)

        t = n.type
        code = code_text(src, n)

        # ---------- Imports ----------
        # ESM: import ... from 'X' | import 'X'
        if t == "import_declaration":
            mod = get_import_source_from_import_decl(src, n)
            if mod:
                dim = classify_import_to_dim(mod)
                if dim:
                    add_feat(feats, file_path, n, dim, src)

        # Dynamic import('X') / require('X')
        if t == "call_expression":
            # import/require source
            mod = get_require_arg_from_call(src, n)
            if mod:
                dim = classify_import_to_dim(mod)
                if dim:
                    add_feat(feats, file_path, n, dim, src)

            # ---------- Calls / usage ----------
            # OS / FS / NET / ENC / PROC usage
            if is_os_call(code):
                add_feat(feats, file_path, n, "R2", src)
            if is_fs_call(code):
                add_feat(feats, file_path, n, "R4", src)
            if is_net_call(code):
                add_feat(feats, file_path, n, "D2", src)
            if is_enc_call(code):
                add_feat(feats, file_path, n, "E2", src)
            if is_process_call(code):
                add_feat(feats, file_path, n, "P2", src)
            if is_sensitive_info(code):
                add_feat(feats, file_path, n, "R5", src)
            if is_runtime_eval(code):
                add_feat(feats, file_path, n, "P4", src)

            # child_process + bash-like
            if re.search(r"\bchild_process.*\.(exec|execFile|spawn|fork)\s*\(", code):
                # shell:true 
                if re.search(r"shell\s*:\s*true", code):
                    add_feat(feats, file_path, n, "P3", src)
                # exec
                if is_bash_like_command(code, shell_re):
                    add_feat(feats, file_path, n, "P3", src)

        # new URL('...') 
        if t in ("new_expression", "call_expression"):
            if re.search(r"\bnew\s+URL\s*\(", code):
                add_feat(feats, file_path, n, "D3", src)

        # ---------- Strings ----------
        if t in ("string", "template_string", "string_fragment"):
            s = norm_string(src, n)

            # URL
            if is_url_like(s):
                add_feat(feats, file_path, n, "D3", src)

            # Base64-like
            if is_base64_like(s, long_threshold if long_threshold < 80 else 80):
                add_feat(feats, file_path, n, "E3", src)

            # Long string
            if len(s) >= long_threshold:
                add_feat(feats, file_path, n, "E4", src)

        # ---------- Member expr for sensitive/process ----------
        if t == "member_expression":
            if is_sensitive_info(code):
                add_feat(feats, file_path, n, "R5", src)
            if is_process_call(code):
                add_feat(feats, file_path, n, "P2", src)

    return feats

# ----------------------------
# Driver
# ----------------------------

def should_skip(path: pathlib.Path, scan_node_modules: bool) -> bool:
    if scan_node_modules:
        return False
    return "node_modules" in path.parts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entries", required=True, help="Path to a file containing one JS/TS entry path per line")
    ap.add_argument("--features_out", required=True, help="Append JSONL features here")
    ap.add_argument("--long-threshold", type=int, default=50, help="E4: long string length threshold (default: 50)")
    ap.add_argument("--scan-node-modules", type=str, default="false", help="Scan files under node_modules (true/false)")
    args = ap.parse_args()

    scan_nm = args.scan_node_modules.strip().lower() == "true"
    shell_re = compile_shell_re()

    feats_out = pathlib.Path(args.features_out)
    with open(args.entries, "r", encoding="utf-8") as fr, open(feats_out, "a", encoding="utf-8") as fw:
        for line in fr:
            p = pathlib.Path(line.strip())
            if not p or not p.exists():
                continue
            if should_skip(p, scan_nm):
                continue

            feats = scan_file(p, long_threshold=args.long_threshold, shell_re=shell_re)
            for f in feats:
                fw.write(json.dumps(f, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
