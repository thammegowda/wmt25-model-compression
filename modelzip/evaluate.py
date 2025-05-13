#!/usr/bin/env python

# 2025-05-09: Initial version by TG Gowda

"""
Evaluation pipeline. The tests and metrics are for demo purposes only.
The actual testsets and metrics will be based on WMT25 General MT task.
"""
import argparse
import logging as LOG
import subprocess as sp
import sys
from pathlib import Path

from modelzip.config import DEF_BATCH_SIZE, DEF_LANG_PAIRS, TASK_CONF, WORK_DIR


def get_score(src_file: Path, out_file: Path, ref_file: Path, metric: str):
    if metric == "chrf":
        cmd = f"sacrebleu {ref_file} -i {out_file} -m {metric} -b -lc"
    else:
        cmd = f"pymarian-eval -m {metric} -r {ref_file} -t {out_file} -s {src_file} -a only"
    LOG.info(f"Scoring: {cmd}")
    return sp.check_output(cmd, shell=True, text=True).strip()


def get_run_cmd(model_dir: Path) -> str:
    """finds a run script inside the model_dir and returns the command to run it

    Looks for run.py, run.sh, or run (executable)
    """

    run_scripts = [model_dir / "run.py", model_dir / "run.sh", model_dir / "run"]

    if not any(script.exists() for script in run_scripts):
        raise ValueError(
            f"Model directory {model_dir} does not contain a run.py or run.sh script"
        )
    run_script = next(script for script in run_scripts if script.exists())
    if run_script.suffix == ".sh":
        run_cmd = f"bash {run_script}"
    elif run_script.suffix == ".py":
        run_cmd = f"{sys.executable} {run_script}"
    else:  # hope it got its own shebang and is executable
        run_cmd = f"{run_script}"
    return run_cmd


def evaluate(
    tests_dir: Path,
    model_dir: Path,
    langs=DEF_LANG_PAIRS,
    metrics=TASK_CONF["metrics"],
    batch_size: int = DEF_BATCH_SIZE,
):

    run_cmd = get_run_cmd(model_dir)
    model_name = model_dir.name
    for pair in langs:
        src, tgt = pair.split("-")
        lang_dir = tests_dir / pair
        for src_file in lang_dir.glob(f"*.{src}-{tgt}.{src}"):
            test = src_file.stem.replace(f".{src}-{tgt}.{src}", "")
            ref = lang_dir / f"{test}.{src}-{tgt}.{tgt}"
            out = lang_dir / f"{test}.{src}-{tgt}.{tgt}.{model_name}.out"
            if not out.exists() or out.stat().st_size == 0:
                tmp_file = out.with_suffix(out.suffix + ".tmp")
                tmp_file.unlink(missing_ok=True)
                run_cmd_full = (
                    f"{run_cmd} {pair} {batch_size} < {src_file} > {tmp_file}"
                )
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


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate models on WMT25",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-w", "--work", type=Path, default=WORK_DIR)
    parser.add_argument(
        "-l",
        "--langs",
        nargs="+",
        help="Lang pairs to evaluate",
        default=DEF_LANG_PAIRS,
    )
    parser.add_argument(
        "-b", "--batch", dest="batch_size", type=int, default=DEF_BATCH_SIZE
    )
    parser.add_argument(
        "-m",
        "--model",
        type=Path,
        required=True,
        help="Path to model directory. Must have a run.py or run.sh script",
    )
    parser.add_argument(
        "-M",
        "--metrics",
        nargs="+",
        default=TASK_CONF["metrics"],
        help="Metrics to use for evaluation",
    )
    args = parser.parse_args()
    tests_dir = args.work / "tests"
    evaluate(
        tests_dir,
        args.model,
        langs=args.langs,
        batch_size=args.batch_size,
        metrics=args.metrics,
    )


if __name__ == "__main__":
    main()
