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


def compress_model(model_dir: Path, output_dir: Path, quant_config):

    flag_file = output_dir / "._OK"
    if flag_file.exists():
        LOG.info(f"Model {output_dir} already exists; skipping")
        return
    loader_args = dict(local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, **loader_args)
    loader_args["device_map"] = "auto"
    # Set the quantization config
    loader_args["quantization_config"] = quant_config
    LOG.info(f"Loading model {model_dir}; quantization config: {quant_config}")
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
    LOG.info(f"Saved {output_dir}; quantization config: {quant_config}")


def demo_methods(model_dir: Path):
    """
    Demo methosds included in organizers' code.
    """
    model_name = model_dir.name.replace("-base", "")
    for approach in ["bnb-8bit", "bnb-4bit"]:
        output_dir = model_dir.with_name(model_name + f"-{approach}")
        qargs = {}
        if approach == "bnb-8bit":
            qargs["load_in_8bit"] = True
        elif approach == "bnb-4bit":
            qargs["load_in_4bit"] = True
        else:
            raise ValueError(f"Unknown approach {approach}")
        #########
        quant_config = BitsAndBytesConfig(**qargs)
        compress_model(model_dir=model_dir, output_dir=output_dir, quant_config=quant_config)


def explore_bnb_variants(model_dir: Path):
    """
    My exploration methods for model compression using bitsandbytes.
    """
    model_name = model_dir.name.replace("-base", "")
    # 4bit nf4 vs fp4
    output_dir = model_dir.with_name(model_name + "-bnb-4bit-nf4")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
     )
    compress_model(model_dir=model_dir, output_dir=output_dir, quant_config=quant_config)
    
    # 4bit fp4
    output_dir = model_dir.with_name(model_name + "-bnb-4bit-fp4")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="fp4",
        )
    compress_model(model_dir=model_dir, output_dir=output_dir, quant_config=quant_config)
   
   # double quantization
    output_dir = model_dir.with_name(model_name + "-bnb-4bit-nf4-2q")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    compress_model(model_dir=model_dir, output_dir=output_dir, quant_config=quant_config)
    
    # quantize lm_head too
    output_dir = model_dir.with_name(model_name + "-bnb-8bit-qlmhead")
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_skip_modules=[],  # 
    )
    

def my_exploration_methods(model_dir: Path):
    """
    My exploration methods for model compression.
    Participants are expected to implement their own methods.
    """
    LOG.info("Exploring bitsandbytes variants")
    explore_bnb_variants(model_dir)
    
    # Add more methods as needed
    # e.g., explore other quantization methods, pruning, etc.
    # compress_model(model_dir, output_dir, quant_config)
    

def main():
    parser = argparse.ArgumentParser(
        description="Download and compress models for WMT25 Model Compression Task")
    parser.add_argument(
        "-m",
        "--model",
        type=Path,
        help="Path to base model directory",
        default="./workdir/models/aya-expanse-8b-base",
    )
    
    args = parser.parse_args()
    model_dir = args.model
    
    # demo methods included in organizers' code
    run_demos = False
    if run_demos:
        demo_methods(model_dir)
    else:
        my_exploration_methods(model_dir)
    

if __name__ == "__main__":
    main()
