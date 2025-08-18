#!/usr/bin/env python

# 2025-05-09: Initial version by TG Gowda

"""
Evaluation pipeline. The tests and metrics are for demo purposes only.
The actual testsets and metrics will be based on WMT25 General MT task.
"""
import argparse
import logging as LOG
import os
import subprocess as sp
from pathlib import Path

from modelzip.config import DEF_BATCH_SIZE, DEF_LANG_PAIRS, TASK_CONF, WORK_DIR
import shutil

DEF_SHOW_PROGRESS = True


def get_score(src_file: Path, out_file: Path, ref_file: Path, metric: str):
    if metric == "chrf":
        cmd = f"sacrebleu {ref_file} -i {out_file} -m {metric} -b -lc"
    else:
        cmd = f"pymarian-eval -m {metric} -r {ref_file} -t {out_file} -s {src_file} -a only"
    LOG.info(f"Scoring: {cmd}")
    return sp.check_output(cmd, shell=True, text=True).strip()


def get_run_cmd(model_dir: Path) -> str:
    """finds a run script inside the model_dir and returns the command to run it
    Looks for run.sh
    """
    run_script = model_dir / "run.sh"
    assert run_script.exists(), f"run.sh not found in {model_dir}"
    run_cmd = f"bash {run_script}"
    return run_cmd


def line_count(file: Path) -> int:
    """Returns the number of lines in a file."""
    with open(file, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def evaluate(
    tests_dir: Path,
    model_dir: Path,
    langs=DEF_LANG_PAIRS,
    metrics=TASK_CONF["metrics"],
    batch_size: int = DEF_BATCH_SIZE,
    show_progress: bool = DEF_SHOW_PROGRESS,
    backup_dir: Path = None
):

    run_cmd = get_run_cmd(model_dir)
    model_name = model_dir.name
    for pair in langs:
        src, tgt = pair.split("-")
        lang_dir = tests_dir / pair
        for src_file in lang_dir.glob(f"*.{src}-{tgt}.{src}"):
            test = src_file.name.replace(f".{src}-{tgt}.{src}", "")
            ref = lang_dir / f"{test}.{src}-{tgt}.{tgt}"
            out = lang_dir / f"{test}.{src}-{tgt}.{tgt}.{model_name}.out"
            if not out.exists() or out.stat().st_size == 0:
                tmp_file = out.with_suffix(out.suffix + ".tmp")
                tmp_file.unlink(missing_ok=True)

                n_lines = line_count(src_file)
                pbar_cmd = ""
                if show_progress:
                    pbar_cmd = (
                        f" | tqdm --total {n_lines} --desc {model_dir.name}-{src_file.name} --unit line --dynamic-ncols"
                    )
                run_cmd_full = f"{run_cmd} {pair} {batch_size} < {src_file} {pbar_cmd} > {tmp_file}"
                LOG.info(f"Running command: {run_cmd_full}")
                try:
                    sp.check_call(run_cmd_full, shell=True)
                    if tmp_file.exists() and tmp_file.stat().st_size > 0:
                        tmp_file.rename(out)
                        LOG.info(f"Wrote translations to {out}")
                except sp.CalledProcessError as e:
                    LOG.error(f"Error running command: {e}")
                    continue
            for m in metrics:
                score_file = out.with_suffix(out.suffix + f".{m}.score")
                if not score_file.exists() or score_file.stat().st_size == 0:
                    score = get_score(src_file, out, ref, m)
                    score_file.write_text(score)
                    LOG.info(f"{score_file.name} : {score}")
                else:
                    LOG.info(f"Skipping existing score file {score_file}")
            if backup_dir:
                backup_results(lang_dir, backup_dir / lang_dir.name)
                

def backup_results(from_dir:Path, to_dir:Path):
    """backup from from_dir to to_dir. Update new files. ignore existing and old files"""
    if not from_dir.exists():
        LOG.warning(f"Source directory {from_dir} does not exist")
        return
    LOG.info(f"Backing up results from {from_dir} --> {to_dir}")
    to_dir.mkdir(parents=True, exist_ok=True)
    copied, updated, skipped = 0, 0, 0
    for src in from_dir.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(from_dir)
        dst = to_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            if not dst.exists() or dst.stat().st_size == 0:
                shutil.copy2(src, dst)
                copied += 1
                LOG.info(f"Copied new file {rel}")
            else:
                if src.stat().st_mtime > dst.stat().st_mtime:
                    shutil.copy2(src, dst)
                    updated += 1
                    LOG.info(f"Updated file {rel}")
                else:
                    skipped += 1
        except OSError as e:
            LOG.error(f"Failed to copy {rel}: {e}")

    LOG.info(f"Backup summary: copied={copied} updated={updated} skipped={skipped}")
    

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate models on WMT25", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-w", "--work", type=Path, default=WORK_DIR)
    parser.add_argument("-l", "--langs", nargs="+", help="Lang pairs to evaluate", default=DEF_LANG_PAIRS)
    parser.add_argument("-b", "--batch", dest="batch_size", type=int, default=DEF_BATCH_SIZE)
    parser.add_argument(
        "-m", "--model", type=Path, required=True, help="Path to model directory. Must have a run.py or run.sh script"
    )
    parser.add_argument(
        "-M", "--metrics", nargs="+", default=TASK_CONF["metrics"], help="Metrics to use for evaluation"
    )
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--pbar", dest="show_progress", action="store_true", default=DEF_SHOW_PROGRESS,
        help="Enable progress bar during evaluation"
    )
    group.add_argument(
        "--no-pbar", dest="show_progress", action="store_false",
        help="Disable progress bar during evaluation"
    )

    job_name = os.environ.get("AMLT_JOB_NAME", "$SUB_ID")
    def_backup_name = f"/mnt/tg/data/projects/wmt25-model-compression/evals/backup-v1/{job_name}"
    parser.add_argument(
        "-B", "--backup", type=Path, default=def_backup_name,
        help=f"Backup directory to save or update results. Use shared drive like blob container mount for archival purposes.")

    args = parser.parse_args()
    tests_dir = args.work / "tests"
    evaluate(tests_dir, args.model, langs=args.langs, batch_size=args.batch_size,
             metrics=args.metrics, show_progress=args.show_progress, backup_dir=args.backup)


if __name__ == "__main__":
    main()
