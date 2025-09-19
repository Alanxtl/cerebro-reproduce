#!/usr/bin/env bash
set -euo pipefail
PKG_DIR="${1:-./data/npm_pkgs/<your_pkg>}"  
OUT_JSON="${2:-./outputs/cg.json}"
OUT_HTML="${3:-./outputs/cg.html}"

jelly -j "$OUT_JSON" -m "$OUT_HTML" "$PKG_DIR"

echo "CG JSON => $OUT_JSON"
echo "CG HTML => $OUT_HTML 
