#!/usr/bin/env python
#
# 2025-05-09: Initial version by TG Gowda
#
"""
This is a sample script to demo a model compression pipeline.
Participants are expected to implement their own compression pipeline,
    which should produce an output model directory with run script.
Organizers would be using the output directory of this script to run the evaluation
"""

import argparse
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from modelzip.config import LOG


def compress_model(model_dir: Path, output_dir: Path, approach="bnb-8bit"):

    flag_file = output_dir / "._OK"
    if flag_file.exists():
        LOG.info(f"Model {output_dir} already exists; skipping")
        return
    loader_args = dict(local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, **loader_args)
    loader_args["device_map"] = "auto"
    qargs = {}
    if approach == "bnb-8bit":
        qargs["load_in_8bit"] = True
    elif approach == "bnb-4bit":
        qargs["load_in_4bit"] = True
    else:
        raise ValueError(f"Unknown approach {approach}")

    # Set the quantization config
    loader_args["quantization_config"] = BitsAndBytesConfig(**qargs)
    LOG.info(f"Loading model {model_dir}; approach: {approach}")
    model = AutoModelForCausalLM.from_pretrained(model_dir, **loader_args)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    # copy run.py
    copy_files = ["run.py", "run.sh"]
    for file in copy_files:
        src_file = model_dir / file
        dst_file = output_dir / file
        if src_file.exists():
            dst_file.write_text(src_file.read_text())
        else:
            LOG.warning(f"{file} not found in {model_dir}; skipping")

    flag_file.touch()
    LOG.info(f"Saved {model_dir.name} with {approach} at {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download and compress models for WMT25")
    parser.add_argument(
        "-m",
        "--model",
        type=Path,
        help="Path to base model directory",
        default="./workdir/models/aya-expanse-8b-base",
    )
    args = parser.parse_args()
    model_dir = args.model
    for approach in ["bnb-8bit", "bnb-4bit"]:
        name = model_dir.name.replace("-base", "") + f"-{approach}"
        output_dir = model_dir.with_name(name)
        compress_model(model_dir=model_dir, output_dir=output_dir, approach=approach)


if __name__ == "__main__":
    main()
