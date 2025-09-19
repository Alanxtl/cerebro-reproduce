#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gc
import os
import sys
import json
import tarfile
import zipfile
import shutil
import pathlib
import subprocess
import argparse
import hashlib
from tqdm import tqdm

ZIP_PWD = b"infected"
EXTS = (".tgz", ".tar.gz", ".tar", ".zip")

def jhash(p: pathlib.Path) -> str:
    return hashlib.sha1(str(p).encode()).hexdigest()[:10]

def ensure_dir(p: pathlib.Path):
    p.mkdir(parents=True, exist_ok=True)

def list_archives(root: pathlib.Path):
    out = []
    for d, _, fs in os.walk(root):
        for fn in fs:
            if fn.endswith(EXTS):
                out.append(pathlib.Path(d) / fn)
    return out

def safe_extract(arch: pathlib.Path, dest_root: pathlib.Path) -> pathlib.Path:
    out = dest_root / (arch.stem.replace(".tar", "") + "_" + jhash(arch))
    ensure_dir(out)
    ap = str(arch)
    if ap.endswith(".zip"):
        with zipfile.ZipFile(ap) as zf:
            try:
                zf.extractall(out, pwd=ZIP_PWD)
            except RuntimeError:
                zf.extractall(out)
    else:
        mode = "r:gz" if (ap.endswith(".tgz") or ap.endswith(".tar.gz")) else "r:"
        with tarfile.open(ap, mode) as tf:
            tf.extractall(out)
    return out

def find_pkg_dir(root: pathlib.Path):
    if not root.exists():
        return None
    pj = root / "package.json"
    if pj.exists():
        return root
    for d, sub, fs in os.walk(root):
        if "package.json" in fs:
            return pathlib.Path(d)
        # avoid deep traversal
        rel = pathlib.Path(d).relative_to(root)
        if len(rel.parts) > 3:
            sub[:] = []
    return None

def run(cmd, timeout=None, cwd=None):
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        out, err = p.communicate(timeout=timeout)
        code = p.returncode
    except subprocess.TimeoutExpired:
        p.kill()
        return -9, "", "TIMEOUT"
    return code, out, err

def process_one(arch: pathlib.Path, tools_dir: pathlib.Path, tmp_root: pathlib.Path,
                out_root: pathlib.Path, jelly_timeout: int, label: int,
                batch_seq_path: pathlib.Path):
    """single package process"""
    sid = arch.stem + "_" + jhash(arch)
    work = tmp_root / sid
    ensure_dir(work)
    logd = out_root / "logs" / sid
    ensure_dir(logd)

    try:
        src = safe_extract(arch, work)
        pkg_dir = find_pkg_dir(src)
        if not pkg_dir or not os.path.exists(pkg_dir):
            (logd / "error.txt").write_text("No package.json\n", encoding="utf-8")
            return None

        # 2) Jelly gen CG
        cg_json = logd / "cg.json"
        cmd = f"jelly -j \"{cg_json}\" ."
        p = subprocess.Popen(
            cmd,
            cwd=str(pkg_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True,
        )
        try:
            out, err = p.communicate(timeout=jelly_timeout)
            code = p.returncode
        except subprocess.TimeoutExpired:
            p.kill()
            out, err = "", "TIMEOUT"
            code = -9
        (logd / "jelly.stdout.log").write_text(out or "", encoding="utf-8", errors="ignore")
        (logd / "jelly.stderr.log").write_text(err or "", encoding="utf-8", errors="ignore")

        # 3) entries
        entries_txt = logd / "entries.txt"
        code, out, err = run([
            sys.executable, str(tools_dir / "02_extract_entries.py"),
            "--pkg_dir", str(pkg_dir),
            "--out", str(entries_txt)
        ])
        (logd / "step02.stdout.log").write_text(out or "", encoding="utf-8", errors="ignore")
        (logd / "step02.stderr.log").write_text(err or "", encoding="utf-8", errors="ignore")
        if not entries_txt.exists() or entries_txt.read_text(encoding="utf-8").strip() == "":
            (logd / "warn.txt").write_text("No entries\n", encoding="utf-8")
            return None

        # 4) AST 
        feat_jsonl = logd / "features.jsonl"
        code, out, err = run([
            sys.executable, str(tools_dir / "03_ast_walk_and_map_dims.py"),
            "--entries", str(entries_txt),
            "--features_out", str(feat_jsonl)
        ])
        (logd / "step03.stdout.log").write_text(out or "", encoding="utf-8", errors="ignore")
        (logd / "step03.stderr.log").write_text(err or "", encoding="utf-8", errors="ignore")

        # 5) Build sequences
        seq_jsonl = logd / "sequence.jsonl"
        code, out, err = run([
            sys.executable, str(tools_dir / "04_build_sequences.py"),
            "--pkg_name", arch.name,
            "--pkg_dir", str(pkg_dir.resolve()),
            "--entries", str(entries_txt),
            "--cg", str(cg_json),
            "--sequence_out", str(seq_jsonl),
            "--long-threshold", "50",
            "--scan-node-modules", "false",
            "--label", str(label)
        ], timeout=None)
        (logd / "seq.stdout.log").write_text(out or "", encoding="utf-8", errors="ignore")
        (logd / "seq.stderr.log").write_text(err or "", encoding="utf-8", errors="ignore")

        # 6) write batch-level sequences
        if seq_jsonl.exists():
            with open(batch_seq_path, "a", encoding="utf-8") as fw, open(seq_jsonl, "r", encoding="utf-8") as fr:
                for line in fr:
                    o = json.loads(line)
                    o["pkg"] = arch.name
                    o["label"] = label
                    fw.write(json.dumps(o, ensure_ascii=False) + "\n")

        return True
    except Exception as e:
        (logd / "exception.txt").write_text(repr(e), encoding="utf-8")
        return None
    finally:
        try:
            shutil.rmtree(work, ignore_errors=True)
            # shutil.rmtree(logd, ignore_errors=True)
        except:
            pass
        gc.collect()

# ----------------------------
# workers
# ----------------------------
def run_worker(batch_file: pathlib.Path, batch_id: int, tools: pathlib.Path,
               tmp: pathlib.Path, out: pathlib.Path, jelly_timeout: int, label: int):
    data = json.loads(pathlib.Path(batch_file).read_text(encoding="utf-8"))
    archives = [pathlib.Path(p) for p in data["archives"]]

    batch_seq_path = out / f"sequences.batch_{batch_id:05d}.jsonl"
    if batch_seq_path.exists():
        batch_seq_path.unlink()

    ok, skipped, errc = 0, 0, 0
    for arch in archives:
        r = process_one(arch, tools, tmp, out, jelly_timeout, label, batch_seq_path)
        if r is True:
            ok += 1
        elif r is None:
            skipped += 1
        else:
            errc += 1
    gc.collect()
    print(json.dumps({"ok": ok, "skipped": skipped, "err": errc}))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--label", type=int, required=True)
    ap.add_argument("--jelly_timeout", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=50, help="how many archives per batch")
    ap.add_argument("--parallel_batches", type=int, default=1, help="how many batches to run in parallel")
    # worker args
    ap.add_argument("--worker_batch_file", type=str, default="")
    ap.add_argument("--worker_batch_id", type=int, default=-1)
    args = ap.parse_args()

    dataset = pathlib.Path(args.dataset_dir).resolve()
    out = pathlib.Path(args.out_dir).resolve()
    ensure_dir(out)
    ensure_dir(out / "logs")
    tmp = out / "tmp"
    ensure_dir(tmp)
    tools = pathlib.Path(__file__).resolve().parent

    if args.worker_batch_file:
        run_worker(pathlib.Path(args.worker_batch_file), int(args.worker_batch_id),
                   tools, tmp, out, args.jelly_timeout, args.label)
        return

    archives = list_archives(dataset)
    if not archives:
        print("No archives found.", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(archives)} archives")

    global_seq = out / "sequences.jsonl"
    if global_seq.exists():
        global_seq.unlink()

    # split into batches
    B = max(1, int(args.batch_size))
    batches = [archives[i:i+B] for i in range(0, len(archives), B)]
    print(f"Planned {len(batches)} batches (batch_size={B}, parallel={args.parallel_batches})")

    running = {}
    pbar = tqdm(total=len(batches), desc="Batches")

    def launch_batch(batch_idx: int, batch_archives):
        bfile = tmp / f"._batch_{batch_idx:05d}.json"
        with open(bfile, "w", encoding="utf-8") as f:
            json.dump({"archives": [str(a) for a in batch_archives]}, f)
        proc = subprocess.Popen([
            sys.executable, str(pathlib.Path(__file__).resolve()),
            "--dataset_dir", str(dataset),
            "--out_dir", str(out),
            "--label", str(args.label),
            "--jelly_timeout", str(args.jelly_timeout),
            "--worker_batch_file", str(bfile),
            "--worker_batch_id", str(batch_idx),
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return proc, bfile

    idx = 0
    max_parallel = max(1, int(args.parallel_batches))
    while idx < len(batches) or running:
        while idx < len(batches) and len(running) < max_parallel:
            proc, bfile = launch_batch(idx, batches[idx])
            running[idx] = (proc, bfile)
            idx += 1

        done_idxs = []
        for bi, (proc, bfile) in list(running.items()):
            ret = proc.poll()
            if ret is None:
                continue
            _out, _err = proc.communicate()
            try:
                bfile.unlink(missing_ok=True)
            except Exception:
                pass

            batch_seq_path = out / f"sequences.batch_{bi:05d}.jsonl"
            if batch_seq_path.exists():
                with open(global_seq, "a", encoding="utf-8") as fw, open(batch_seq_path, "r", encoding="utf-8") as fr:
                    shutil.copyfileobj(fr, fw)
                try:
                    batch_seq_path.unlink()
                except Exception:
                    pass

            pbar.update(1)
            done_idxs.append(bi)

        for bi in done_idxs:
            running.pop(bi, None)

    pbar.close()
    print("Done. sequences.jsonl =>", global_seq)

if __name__ == "__main__":
    main()
