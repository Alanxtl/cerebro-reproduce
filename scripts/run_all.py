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
import tempfile
import time
from tqdm import tqdm
from typing import List

ZIP_PWD = b"infected"
EXTS = (".tgz", ".tar.gz", ".tar", ".zip")
WINDOWS_PATH_LIMIT = 259


def canon_path(p) -> str:
    try:
        s = os.path.normpath(str(p))
    except Exception:
        s = str(p)
    return os.path.normcase(s)


def path_variants(p) -> List[str]:
    s = canon_path(p)
    return list({s, s.replace("\\", "/"), s.replace("/", "\\")})


def load_groundtruth_index(gt_path: pathlib.Path):
    by_path = {}
    by_name = {}
    dup_names = set()
    for line in gt_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        mt = o.get("annotation").get("malicious_types")
        if not isinstance(mt, list):
            mt = []
        mt = [str(x) for x in mt if str(x).strip()]
        tb = o.get("bin_label")
        entry = {
            "malicious_types": mt,
            "type_bucket": tb,
        }
        for key in (o.get("archive_name")):
            if key:
                for v in path_variants(key):
                    by_path[v] = entry
                name = pathlib.Path(str(key).replace("\\", "/")).name
                if name:
                    if name in by_name and by_name.get(name) is not entry:
                        dup_names.add(name)
                    else:
                        by_name[name] = entry
    for name in dup_names:
        by_name.pop(name, None)
    return {"by_path": by_path, "by_name": by_name}


def lookup_groundtruth(arch: pathlib.Path, gt_index):
    if not gt_index:
        return None
    for v in path_variants(arch):
        if v in gt_index["by_path"]:
            return gt_index["by_path"][v]
    return gt_index["by_name"].get(arch.name)


def jhash(p: pathlib.Path) -> str:
    return hashlib.sha1(str(p).encode()).hexdigest()[:10]


def _archive_stem(arch: pathlib.Path) -> str:
    return arch.stem.replace(".tar", "")


def _bounded_archive_key(arch: pathlib.Path, parent: pathlib.Path) -> str:
    stem = _archive_stem(arch)
    suffix = f"_{jhash(arch)}"
    candidate = f"{stem}{suffix}"
    if len(str(parent / candidate)) <= WINDOWS_PATH_LIMIT:
        return candidate

    max_name_length = max(1, WINDOWS_PATH_LIMIT - len(str(parent)) - 1)
    max_stem_length = max(1, max_name_length - len(suffix))
    return f"{stem[:max_stem_length]}{suffix}"


def _shared_tmp_root(out_dir: pathlib.Path) -> pathlib.Path:
    # Keep extraction paths short on Windows so archive members do not exceed
    # MAX_PATH once unpacked under the per-package work directory.
    digest = hashlib.sha1(str(out_dir).encode("utf-8")).hexdigest()[:12]
    return pathlib.Path(tempfile.gettempdir()) / "cb" / digest


def ensure_dir(p: pathlib.Path):
    p.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        print(f"Failed to create output directory: {p}", file=sys.stderr)
        sys.exit(1)


def list_archives(root: pathlib.Path):
    out = []
    for d, _, fs in os.walk(root):
        for fn in fs:
            if fn.endswith(EXTS):
                out.append(pathlib.Path(d) / fn)
    return out


def safe_extract(arch: pathlib.Path, dest_root: pathlib.Path) -> pathlib.Path:
    out = dest_root / _bounded_archive_key(arch, dest_root)
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
    p = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        out, err = p.communicate(timeout=timeout)
        code = p.returncode
    except subprocess.TimeoutExpired:
        try:
            p.terminate()
            out, err = p.communicate(timeout=3)
        except Exception:
            p.kill()
            out, err = "", "TIMEOUT"
        return -9, out or "", err or "TIMEOUT"
    return code, out, err


def process_one(
    arch: pathlib.Path,
    tools_dir: pathlib.Path,
    tmp_root: pathlib.Path,
    out_root: pathlib.Path,
    name: str,
    jelly_timeout: int,
    label: int,
    batch_seq_path: pathlib.Path,
    gt_index,
    process_timeout: int = None,
):
    """single package process"""
    sid = _bounded_archive_key(arch, out_root / "logs")
    work = tmp_root / sid
    ensure_dir(work)
    logd = out_root / "logs" / sid
    ensure_dir(logd)

    deadline = None
    if process_timeout and process_timeout > 0:
        deadline = time.time() + process_timeout

    def remaining_seconds():
        if deadline is None:
            return None
        left = int(deadline - time.time())
        return max(left, 0)

    try:
        # 0) no time last
        if deadline is not None and remaining_seconds() == 0:
            (logd / "error.txt").write_text(
                "TIMEOUT (before start)\n", encoding="utf-8"
            )
            return None

        src = safe_extract(arch, work)
        pkg_dir = find_pkg_dir(src)
        if not pkg_dir or not os.path.exists(pkg_dir):
            (logd / "error.txt").write_text("No package.json\n", encoding="utf-8")
            return None

        # 2) Jelly gen CG
        cg_json = logd / "cg.json"
        cmd = f'jelly -j "{cg_json}" .'

        step_left = remaining_seconds()
        per_step_timeout = (
            jelly_timeout if step_left is None else min(jelly_timeout, step_left)
        )

        p = subprocess.Popen(
            cmd,
            cwd=str(pkg_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True,
        )
        try:
            out, err = p.communicate(timeout=per_step_timeout)
            code = p.returncode
        except subprocess.TimeoutExpired:
            try:
                p.terminate()
                out, err = p.communicate(timeout=3)
            except Exception:
                p.kill()
                out, err = "", "TIMEOUT"
            code = -9

        (logd / "01_jelly.stdout.log").write_text(
            out or "pass", encoding="utf-8", errors="ignore"
        )
        (logd / "01_jelly.stderr.log").write_text(
            err or "pass", encoding="utf-8", errors="ignore"
        )

        if deadline is not None and remaining_seconds() == 0:
            (logd / "error.txt").write_text("TIMEOUT after jelly\n", encoding="utf-8")
            return None

        # 3) entries
        entries_txt = logd / "entries.txt"
        code, out, err = run(
            [
                sys.executable,
                str(tools_dir / "02_extract_entries.py"),
                "--pkg_dir",
                str(pkg_dir),
                "--out",
                str(entries_txt),
            ],
            timeout=remaining_seconds(),
        )
        (logd / "02_extract_entries.stdout.log").write_text(
            out or "pass", encoding="utf-8", errors="ignore"
        )
        (logd / "02_extract_entries.stderr.log").write_text(
            err or "pass", encoding="utf-8", errors="ignore"
        )
        if (
            not entries_txt.exists()
            or entries_txt.read_text(encoding="utf-8").strip() == ""
        ):
            (logd / "warn.txt").write_text("No entries\n", encoding="utf-8")
            return None

        if deadline is not None and remaining_seconds() == 0:
            (logd / "error.txt").write_text("TIMEOUT after entries\n", encoding="utf-8")
            return None

        # 4) AST
        feat_jsonl = logd / "features.jsonl"
        code, out, err = run(
            [
                sys.executable,
                str(tools_dir / "03_ast_walk_and_map_dims.py"),
                "--entries",
                str(entries_txt),
                "--features_out",
                str(feat_jsonl),
            ],
            timeout=remaining_seconds(),
        )
        (logd / "03_ast_walk_and_map_dims.stdout.log").write_text(
            out or "pass", encoding="utf-8", errors="ignore"
        )
        (logd / "03_ast_walk_and_map_dims.stderr.log").write_text(
            err or "pass", encoding="utf-8", errors="ignore"
        )

        if deadline is not None and remaining_seconds() == 0:
            (logd / "error.txt").write_text("TIMEOUT after ast\n", encoding="utf-8")
            return None

        # 5) Build sequences
        seq_jsonl = logd / f"{name}.jsonl"
        code, out, err = run(
            [
                sys.executable,
                str(tools_dir / "04_build_sequences.py"),
                "--pkg_name",
                arch.name,
                "--pkg_dir",
                str(pkg_dir.resolve()),
                "--entries",
                str(entries_txt),
                "--cg",
                str(cg_json),
                "--sequence_out",
                str(seq_jsonl),
                "--long-threshold",
                "50",
                "--scan-node-modules",
                "false",
                "--label",
                str(label),
            ],
            timeout=remaining_seconds(),
        )
        (logd / "04_build_sequences.stdout.log").write_text(
            out or "pass", encoding="utf-8", errors="ignore"
        )
        (logd / "04_build_sequences.stderr.log").write_text(
            err or "pass", encoding="utf-8", errors="ignore"
        )

        # 6) write batch-level sequences
        if seq_jsonl.exists():
            with (
                open(batch_seq_path, "a", encoding="utf-8") as fw,
                open(seq_jsonl, "r", encoding="utf-8") as fr,
            ):
                for line in fr:
                    o = json.loads(line)
                    o["pkg"] = arch.name
                    o["label"] = label
                    gt = lookup_groundtruth(arch, gt_index)
                    if gt:
                        if gt.get("malicious_types") is not None:
                            o["malicious_types"] = gt["malicious_types"]
                        if gt.get("type_bucket") is not None:
                            o["type_bucket"] = gt["type_bucket"]
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
def run_worker(
    batch_file: pathlib.Path,
    batch_id: int,
    tools: pathlib.Path,
    tmp: pathlib.Path,
    out: pathlib.Path,
    name: str,
    jelly_timeout: int,
    label: int,
    groundtruth_jsonl: str = "",
    process_timeout: int = None,
):
    data = json.loads(pathlib.Path(batch_file).read_text(encoding="utf-8"))
    archives = [pathlib.Path(p) for p in data["archives"]]

    batch_seq_path = out / f"{name}.batch_{batch_id:05d}.jsonl"
    if batch_seq_path.exists():
        batch_seq_path.unlink()

    gt_index = None
    if groundtruth_jsonl:
        gt_index = load_groundtruth_index(pathlib.Path(groundtruth_jsonl))

    ok, skipped, errc = 0, 0, 0
    for arch in archives:
        r = process_one(
            arch=arch,
            tools_dir=tools,
            tmp_root=tmp,
            out_root=out,
            name=name,
            jelly_timeout=jelly_timeout,
            label=label,
            batch_seq_path=batch_seq_path,
            gt_index=gt_index,
            process_timeout=process_timeout,  # NEW
        )
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
    ap.add_argument("--name", required=True)
    ap.add_argument("--label", type=int, required=True)
    ap.add_argument("--groundtruth_jsonl", type=str, default="")
    ap.add_argument("--jelly_timeout", type=int, default=120)
    ap.add_argument(
        "--batch_size", type=int, default=50, help="how many archives per batch"
    )
    ap.add_argument(
        "--parallel_batches",
        type=int,
        default=1,
        help="how many batches to run in parallel",
    )
    ap.add_argument(
        "--process_timeout",
        type=int,
        default=0,
        help="per-package hard timeout (seconds); 0 or negative disables",
    )
    # worker args
    ap.add_argument("--worker_batch_file", type=str, default="")
    ap.add_argument("--worker_batch_id", type=int, default=-1)
    args = ap.parse_args()

    dataset = pathlib.Path(args.dataset_dir).resolve()
    out = pathlib.Path(args.out_dir).resolve()
    ensure_dir(out)
    ensure_dir(out / "logs")
    tmp = _shared_tmp_root(out)
    if not args.worker_batch_file and tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    ensure_dir(tmp)
    tools = pathlib.Path(__file__).resolve().parent

    # process_timeout from cli
    process_timeout = int(args.process_timeout)
    process_timeout = process_timeout if process_timeout > 0 else None

    if args.worker_batch_file:
        run_worker(
            batch_file=pathlib.Path(args.worker_batch_file),
            batch_id=int(args.worker_batch_id),
            tools=tools,
            tmp=tmp,
            out=out,
            name=args.name,
            jelly_timeout=args.jelly_timeout,
            label=args.label,
            groundtruth_jsonl=args.groundtruth_jsonl,
            process_timeout=process_timeout,  # NEW
        )
        return

    archives = list_archives(dataset)
    if not archives:
        print("No archives found.", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(archives)} archives")

    global_seq = out / f"{args.name}.jsonl"
    if global_seq.exists():
        global_seq.unlink()

    # split into batches
    B = max(1, int(args.batch_size))
    batches = [archives[i : i + B] for i in range(0, len(archives), B)]
    print(
        f"Planned {len(batches)} batches (batch_size={B}, parallel={args.parallel_batches})"
    )

    running = {}
    pbar = tqdm(total=len(batches), desc="Batches")

    def launch_batch(batch_idx: int, batch_archives):
        bfile = tmp / f"._batch_{batch_idx:05d}.json"
        with open(bfile, "w", encoding="utf-8") as f:
            json.dump({"archives": [str(a) for a in batch_archives]}, f)
        proc = subprocess.Popen(
            [
                sys.executable,
                str(pathlib.Path(__file__).resolve()),
                "--dataset_dir",
                str(dataset),
                "--out_dir",
                str(out),
                "--name",
                str(args.name),
                "--label",
                str(args.label),
                "--jelly_timeout",
                str(args.jelly_timeout),
                "--process_timeout",
                str(args.process_timeout),
                "--groundtruth_jsonl",
                str(args.groundtruth_jsonl),
                "--worker_batch_file",
                str(bfile),
                "--worker_batch_id",
                str(batch_idx),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
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

            batch_seq_path = out / f"{args.name}.batch_{bi:05d}.jsonl"
            if batch_seq_path.exists():
                with (
                    open(global_seq, "a", encoding="utf-8") as fw,
                    open(batch_seq_path, "r", encoding="utf-8") as fr,
                ):
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
    print(f"Done. {args.name}.jsonl =>", global_seq)


if __name__ == "__main__":
    main()
