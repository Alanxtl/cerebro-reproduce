from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ARCHIVE_EXTS = (".tgz", ".tar.gz", ".tar", ".zip")
WINDOWS_PATH_LIMIT = 259


def load_manifest_archives(manifest_path: Path) -> list[Path]:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return [Path(path) for path in payload.get("archives") or []]


def iter_archives(dataset_dir: Path):
    for path in Path(dataset_dir).rglob("*"):
        if path.is_file() and any(path.name.lower().endswith(ext) for ext in ARCHIVE_EXTS):
            yield path


def resolve_dataset_archives(
    *, manifest_path: Path | None, dataset_dir: Path | None
) -> list[Path]:
    if manifest_path is not None:
        return load_manifest_archives(Path(manifest_path))
    if dataset_dir is not None:
        return sorted(iter_archives(Path(dataset_dir)))
    raise ValueError("Either manifest_path or dataset_dir must be provided.")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _reset_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_optional_path(value: str | None) -> Path | None:
    if not value:
        return None
    return Path(value).resolve()


def _unique_destination(dest_dir: Path, src: Path) -> Path:
    counter = 0
    while True:
        candidate_name = _bounded_destination_name(dest_dir, src.name, counter=counter)
        candidate = dest_dir / candidate_name
        if not candidate.exists():
            return candidate
        counter += 1


def _archive_name_parts(filename: str) -> tuple[str, str]:
    lower_name = filename.lower()
    for ext in sorted(ARCHIVE_EXTS, key=len, reverse=True):
        if lower_name.endswith(ext):
            return filename[: -len(ext)], filename[-len(ext) :]
    path = Path(filename)
    return path.stem, path.suffix


def _bounded_destination_name(dest_dir: Path, filename: str, *, counter: int) -> str:
    stem, suffix = _archive_name_parts(filename)
    counter_suffix = "" if counter == 0 else f"-{counter}"
    candidate_name = f"{stem}{counter_suffix}{suffix}"
    candidate_path = dest_dir / candidate_name
    if len(str(candidate_path)) <= WINDOWS_PATH_LIMIT:
        return candidate_name

    digest = hashlib.sha1(filename.encode("utf-8")).hexdigest()[:12]
    hash_suffix = f"-{digest}{counter_suffix}"
    max_name_length = max(1, WINDOWS_PATH_LIMIT - len(str(dest_dir)) - 1)
    max_stem_length = max(1, max_name_length - len(hash_suffix) - len(suffix))
    return f"{stem[:max_stem_length]}{hash_suffix}{suffix}"


def _materialize_file(src: Path, dest: Path, mode: str) -> None:
    try:
        if mode == "hardlink":
            os.link(src, dest)
        elif mode == "symlink":
            dest.symlink_to(src)
        elif mode == "copy":
            shutil.copy2(src, dest)
        else:
            raise ValueError(f"Unsupported materialize mode: {mode}")
    except OSError:
        shutil.copy2(src, dest)


def materialize_archives(archives: list[Path], dest_dir: Path, mode: str) -> list[Path]:
    _ensure_dir(dest_dir)
    materialized = []
    for archive in archives:
        destination = _unique_destination(dest_dir, archive)
        _materialize_file(archive, destination, mode)
        materialized.append(destination)
    return materialized


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _concat_jsonl(parts: list[Path], destination: Path) -> None:
    with destination.open("w", encoding="utf-8") as out_handle:
        for part in parts:
            if not part.exists():
                continue
            out_handle.write(part.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_sequence_builder(
    *,
    repo_root: Path,
    dataset_dir: Path,
    out_dir: Path,
    name: str,
    label: int,
    jelly_timeout: int,
    parallel_batches: int,
    process_timeout: int,
    groundtruth_jsonl: str | None,
) -> Path:
    cmd = [
        sys.executable,
        "scripts/run_all.py",
        "--dataset_dir",
        str(dataset_dir),
        "--out_dir",
        str(out_dir),
        "--name",
        name,
        "--label",
        str(label),
        "--jelly_timeout",
        str(jelly_timeout),
        "--parallel_batches",
        str(parallel_batches),
        "--process_timeout",
        str(process_timeout),
    ]
    if groundtruth_jsonl:
        cmd.extend(["--groundtruth_jsonl", groundtruth_jsonl])
    _run(cmd, repo_root)
    return out_dir / f"{name}.jsonl"


def run_standard_eval(args: argparse.Namespace) -> dict:
    repo_root = Path(__file__).resolve().parent
    split_dir = Path(args.split_dir).resolve()
    out_dir = _ensure_dir(Path(args.out_dir).resolve())
    work_dir = _reset_dir(out_dir / "work")
    datasets_dir = _reset_dir(work_dir / "datasets")
    sequences_dir = _reset_dir(work_dir / "sequences")
    models_dir = _reset_dir(out_dir / "models")

    train_malicious = resolve_dataset_archives(
        manifest_path=split_dir / "train_manifest.json",
        dataset_dir=None,
    )
    test_malicious = resolve_dataset_archives(
        manifest_path=split_dir / "test_manifest.json",
        dataset_dir=None,
    )
    train_benign = resolve_dataset_archives(
        manifest_path=_resolve_optional_path(args.benign_train_manifest),
        dataset_dir=_resolve_optional_path(args.benign_train_dir),
    )
    test_benign = resolve_dataset_archives(
        manifest_path=_resolve_optional_path(args.benign_test_manifest),
        dataset_dir=_resolve_optional_path(args.benign_test_dir),
    )

    train_mal_dir = datasets_dir / "train_malicious"
    test_mal_dir = datasets_dir / "test_malicious"
    train_ben_dir = datasets_dir / "train_benign"
    test_ben_dir = datasets_dir / "test_benign"
    materialize_archives(train_malicious, train_mal_dir, args.materialize)
    materialize_archives(test_malicious, test_mal_dir, args.materialize)
    materialize_archives(train_benign, train_ben_dir, args.materialize)
    materialize_archives(test_benign, test_ben_dir, args.materialize)

    train_mal_jsonl = _run_sequence_builder(
        repo_root=repo_root,
        dataset_dir=train_mal_dir,
        out_dir=sequences_dir / "train_malicious",
        name="train_malicious",
        label=1,
        jelly_timeout=args.jelly_timeout,
        parallel_batches=args.parallel_batches,
        process_timeout=args.process_timeout,
        groundtruth_jsonl=args.groundtruth_jsonl,
    )
    train_ben_jsonl = _run_sequence_builder(
        repo_root=repo_root,
        dataset_dir=train_ben_dir,
        out_dir=sequences_dir / "train_benign",
        name="train_benign",
        label=0,
        jelly_timeout=args.jelly_timeout,
        parallel_batches=args.parallel_batches,
        process_timeout=args.process_timeout,
        groundtruth_jsonl=None,
    )
    test_mal_jsonl = _run_sequence_builder(
        repo_root=repo_root,
        dataset_dir=test_mal_dir,
        out_dir=sequences_dir / "test_malicious",
        name="test_malicious",
        label=1,
        jelly_timeout=args.jelly_timeout,
        parallel_batches=args.parallel_batches,
        process_timeout=args.process_timeout,
        groundtruth_jsonl=args.groundtruth_jsonl,
    )
    test_ben_jsonl = _run_sequence_builder(
        repo_root=repo_root,
        dataset_dir=test_ben_dir,
        out_dir=sequences_dir / "test_benign",
        name="test_benign",
        label=0,
        jelly_timeout=args.jelly_timeout,
        parallel_batches=args.parallel_batches,
        process_timeout=args.process_timeout,
        groundtruth_jsonl=None,
    )

    train_jsonl = out_dir / "train.jsonl"
    test_jsonl = out_dir / "test.jsonl"
    trainer_metrics_json = out_dir / "trainer_metrics.json"
    _concat_jsonl([train_mal_jsonl, train_ben_jsonl], train_jsonl)
    _concat_jsonl([test_mal_jsonl, test_ben_jsonl], test_jsonl)

    trainer_cmd = [
        sys.executable,
        "scripts/05_train_bert.py",
        "--train-jsonl",
        str(train_jsonl),
        "--test-jsonl",
        str(test_jsonl),
        "--output-dir",
        str(models_dir / "bert"),
        "--final-model-dir",
        str(models_dir / "bert-final"),
        "--metrics-out",
        str(trainer_metrics_json),
        "--num-epochs",
        str(args.num_epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--max-length",
        str(args.max_length),
        "--eval-fraction",
        str(args.eval_fraction),
        "--seed",
        str(args.seed),
    ]
    if args.oversample:
        trainer_cmd.append("--oversample")
    if args.groundtruth_jsonl:
        trainer_cmd.extend(["--ground-truth-jsonl", args.groundtruth_jsonl])
    if args.report_by_type:
        trainer_cmd.append("--report-by-type")
    if args.report_by_difficulty:
        trainer_cmd.append("--report-by-difficulty")
    if args.split_stratify_by_type:
        trainer_cmd.append("--split-stratify-by-type")
    if args.split_stratify_by_difficulty:
        trainer_cmd.append("--split-stratify-by-difficulty")
    _run(trainer_cmd, repo_root)

    trainer_metrics = json.loads(trainer_metrics_json.read_text(encoding="utf-8"))
    payload = {
        "baseline": "cerebro",
        "split_dir": str(split_dir),
        "materialize_mode": args.materialize,
        "seed": args.seed,
        "oversample": args.oversample,
        "counts": {
            "train_malicious": len(train_malicious),
            "test_malicious": len(test_malicious),
            "train_benign": len(train_benign),
            "test_benign": len(test_benign),
        },
        "artifacts": {
            "train_jsonl": str(train_jsonl),
            "test_jsonl": str(test_jsonl),
            "trainer_metrics_json": str(trainer_metrics_json),
            "model_dir": str(models_dir / "bert-final"),
        },
        "metrics": trainer_metrics,
    }
    _write_json(out_dir / "metrics.json", payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Cerebro on explicit train/test manifests from research group splits."
    )
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--benign-train-dir")
    parser.add_argument("--benign-train-manifest")
    parser.add_argument("--benign-test-dir")
    parser.add_argument("--benign-test-manifest")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--groundtruth-jsonl")
    parser.add_argument(
        "--materialize",
        default="hardlink",
        choices=["copy", "hardlink", "symlink"],
    )
    parser.add_argument("--jelly-timeout", type=int, default=1000)
    parser.add_argument("--parallel-batches", type=int, default=2)
    parser.add_argument("--process-timeout", type=int, default=300)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    oversample_group = parser.add_mutually_exclusive_group()
    oversample_group.add_argument("--oversample", dest="oversample", action="store_true")
    oversample_group.add_argument(
        "--no-oversample",
        dest="oversample",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(oversample=False)
    parser.add_argument("--report-by-type", action="store_true")
    parser.add_argument("--report-by-difficulty", action="store_true")
    parser.add_argument("--split-stratify-by-type", action="store_true")
    parser.add_argument("--split-stratify-by-difficulty", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if not (args.benign_train_dir or args.benign_train_manifest):
        raise SystemExit("Provide either --benign-train-dir or --benign-train-manifest.")
    if not (args.benign_test_dir or args.benign_test_manifest):
        raise SystemExit("Provide either --benign-test-dir or --benign-test-manifest.")
    payload = run_standard_eval(args)
    print(json.dumps(payload["metrics"], indent=2))


if __name__ == "__main__":
    main()
