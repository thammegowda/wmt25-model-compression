#!/usr/bin/env python
#
# 2025-05-09: Initial version by TG Gowda
#
"""
Setup script for WMT25 models

This script downloads models and development sets for WMT25 Model Compression Shared Task.
For constrained task, the setup should work out of the box.
For the unconstrained task, participants may tweak this to download their own models.
"""

import argparse
import logging as LOG
import subprocess as sp
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

from modelzip.config import DEF_LANG_PAIRS, HF_CACHE, TASK_CONF, WORK_DIR

LOG.basicConfig(level=LOG.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def setup_eval(work_dir: Path, langs=None):
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    tests_dir = work_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    langs = langs or DEF_LANG_PAIRS
    for lang_pair in langs:
        assert lang_pair in TASK_CONF["langs"], f"Language pair {lang_pair} not in config"
        src, tgt = lang_pair.split("-")
        lang_dir = tests_dir / lang_pair
        lang_dir.mkdir(parents=True, exist_ok=True)
        for test_name, get_cmd in TASK_CONF["langs"][lang_pair].items():
            src_file = lang_dir / f"{test_name}.{src}-{tgt}.{src}"
            ref_file = lang_dir / f"{test_name}.{src}-{tgt}.{tgt}"
            if (
                src_file.exists()
                and ref_file.exists()
                and src_file.stat().st_size > 0
                and ref_file.stat().st_size > 0
            ):
                LOG.info(f"Test files exist for {lang_pair}:{test_name}")
                continue
            LOG.info(f"Fetching {test_name} via: {get_cmd}")
            lines = sp.check_output(get_cmd, shell=True, text=True).strip().replace("\r", "").split("\n")
            lines = [x.strip().split("\t") for x in lines]
            if "mtdata" in get_cmd:
                lines = [x[:2] for x in lines]
            n_errs = sum(1 for i, x in enumerate(lines, 1) if len(x) != 2)
            if n_errs:
                raise ValueError(f"Invalid output from {get_cmd}: {n_errs} errors")
            srcs = [x[0] for x in lines]
            refs = [x[1] for x in lines]
            src_file.write_text("\n".join(srcs))
            ref_file.write_text("\n".join(refs))
            LOG.info(f"Created test files {src_file}, {ref_file}")


def setup_model(work_dir: Path, cache_dir: Path, model_ids=TASK_CONF["models"]):
    # downloads
    work_dir = Path(work_dir)
    models_dir = work_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for model_id in model_ids:
        simple_name = model_id.split("/")[-1]
        model_dir = models_dir / (simple_name + "-base")
        model_dir.mkdir(parents=True, exist_ok=True)
        flag_file = model_dir / "._OK"
        if flag_file.exists():
            LOG.info(f"Model {model_id} already exists; rm {flag_file} to force download")
            continue

        loader_args = dict(cache_dir=cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_id, **loader_args)
        loader_args["device_map"] = "auto"
        loader_args["torch_dtype"] = "auto"
        LOG.info(f"Loading model from {model_id}; args: {loader_args}")
        model = AutoModelForCausalLM.from_pretrained(model_id, **loader_args)
        LOG.info(f"{model_id} loaded successfully. Storing at {model_dir}")
        model_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(model_dir)
        model.save_pretrained(model_dir)

        # copy baseline.py as run.py inside the model_dir
        run_script = model_dir / "run.sh"
        baseline_script = Path(__file__).parent / "run.sh"
        assert baseline_script.exists(), f"Baseline script {baseline_script} does not exist"
        run_script.write_text(baseline_script.read_text())

        flag_file.touch()
        LOG.info(f"Model {model_id} saved successfully at {model_dir}")


def main():
    parser = argparse.ArgumentParser(description="Setup WMT25 shared task")
    parser.add_argument("-w", "--work", type=Path, default=WORK_DIR, help="Work directory")
    parser.add_argument("-l", "--langs", nargs="+", help="Language pairs to setup")
    parser.add_argument(
        "-t",
        "--task",
        choices=["eval", "model", "all"],
        default="all",
        help="Task to perform",
    )
    parser.add_argument("-c", "--cache", type=Path, default=HF_CACHE, help="Cache directory for models")
    args = parser.parse_args()

    # dispatch based on task
    if args.task in ("model", "all"):
        setup_model(work_dir=args.work, cache_dir=args.cache)
    if args.task in ("eval", "all"):
        setup_eval(work_dir=args.work, langs=args.langs)


if __name__ == "__main__":
    main()
