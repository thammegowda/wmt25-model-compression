#!/usr/bin/env python

#
# WMT25 Model Compression Shared Task
#
# Revisions
#  2025-04-28: Initial version by Thamme Gowda
#
#
import argparse
import logging as LOG
import os
from pathlib import Path
from typing import List
import subprocess as sp

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig


LOG.basicConfig(
    level=LOG.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HF_CACHE = Path(os.getenv("HF_HOME", default = Path.home() / ".cache" / "huggingface")) / "hub"

TASK_CONF = {
    "langs": {
        "ces-deu": {"wmt19": "sacrebleu -t wmt19 -l cs-de --echo src ref"},
        "jpn-zho": {
            "wmt24": "sacrebleu -t wmt24 -l ja-zh --echo src ref:refA",
        },
        # "eng-ara": {  # adds \r and messes up alignment
        #    "wmt24pp": "mtdata echo Google-wmt24pp-1-eng-ara_SA ", #| sed $'s/\\r//g'| cut -f1,2
        # }
    },
    "models": ["CohereLabs/aya-expanse-8b"],
    "metrics": [
        "chrf",  # sacrebleu
        #"wmt23-cometkiwi-da-xl",  # from pymarian
    ],
}
DEF_LANG_PAIRS = list(TASK_CONF["langs"].keys())
DEF_MODEL_ID = TASK_CONF["models"][0]
WORK_DIR = "wmt25-compression"
DEF_BATCH_SIZE = 16

LANGS_MAP = dict(
    ces="Czech",
    deu="German",
    jpn="Japanese",
    zho="Chinese",
    eng="English",
    ara="Arabic",
)
TRANSLATE_PROMPT = "Translate the following text from {src} to {tgt}.\n{text}\n"

"""Work dir structure:
wmt25-compression/
    | -- models /
    |    | -- model-name/
    |        | -- base/
    |        | -- bnb-8bit/
    |        | -- bnb-4bit/
    |
    | -- tests/
    |    | -- lang1-lang2/
    |        | -- test1.lang1-lang2.lang1
    |        | -- test1.lang1-lang2.lang2
    |        | -- test1.lang1-lang2.lang2.{model-name}.{approach}.out
"""


def setup_task(
    work_dir: Path = WORK_DIR,
    langs: List[str] = DEF_LANG_PAIRS,
    models: List[str] = TASK_CONF["models"],
    cache_dir: Path = HF_CACHE,
):
    """
    Setup the task by downloading the models and creating the directory structure
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    models_dir = work_dir / "models"
    tests_dir = work_dir / "tests"
    for lang_pair in langs:
        assert lang_pair in TASK_CONF["langs"], f"Language pair {lang_pair} not found in constrained languages"

    models_dir.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)
    for lang_pair in langs:
        src, tgt = lang_pair.split("-")
        lang_dir = tests_dir / lang_pair
        lang_dir.mkdir(parents=True, exist_ok=True)

        # Create test files for each language pair
        for test_name, get_cmd in TASK_CONF["langs"][lang_pair].items():
            src_file = lang_dir / f"{test_name}.{src}-{tgt}.{src}"
            ref_file = lang_dir / f"{test_name}.{src}-{tgt}.{tgt}"
            if src_file.exists() and ref_file.exists() and src_file.stat().st_size > 0 and ref_file.stat().st_size > 0:
                LOG.info(f"Test files already exist for {lang_pair}: {test_name}")
                continue

            LOG.info(f"Run: {get_cmd}")
            lines = sp.check_output(get_cmd, shell=True, text=True)
            lines = lines.strip().replace("\r", "").split("\n")
            lines = [x.strip().split("\t") for x in lines]
            LOG.info(f"Got {len(lines)} lines from {get_cmd}")
            if "mtdata" in get_cmd:
                # mtdata outputs meta info; i could add |cut -f1,2 but it wont be portable
                lines = [x[:2] for x in lines]
            n_errs = 0
            for i, x in enumerate(lines, start=1):
                if len(x) != 2:
                    LOG.error(f"Invalid line {i}: {x}")
                    n_errs += 1
            if n_errs > 0:
                raise ValueError(f"Invalid output from {get_cmd}: {n_errs} errors")

            srcs = [x[0] for x in lines]
            refs = [x[1] for x in lines]
            src_file.write_text("\n".join(srcs))
            ref_file.write_text("\n".join(refs))
            LOG.info(f"Created test files for {lang_pair}: {test_name}")

    for model_id in models:
        simple_name = model_id.split("/")[-1]
        model_dir = models_dir / simple_name
        model_dir.mkdir(parents=True, exist_ok=True)
        for approach in ["base", "bnb-8bit", "bnb-4bit"]:
            try:
                approach_dir = model_dir / approach
                flag_file = approach_dir / "._OK"
                if flag_file.exists():
                    LOG.info(f"Model {model_id} already exists; rm {flag_file} to force download")
                    continue

                loader_args = dict(cache_dir=cache_dir)
                tokenizer = AutoTokenizer.from_pretrained(model_id, **loader_args)
                loader_args["device_map"] = "auto"
                if approach == "base":
                    loader_args["torch_dtype"] = torch.bfloat16
                elif approach in ("bnb-8bit", "bnb-4bit"):
                    qargs = {}
                    if approach == "bnb-8bit":
                        qargs["load_in_8bit"] = True
                    elif approach == "bnb-4bit":
                        qargs["load_in_4bit"] = True
                    else:
                        raise ValueError(f"Unsupported approach: {approach}")
                    # loader_args["device_map"] = None  # dont try to map the quantized model
                    loader_args["quantization_config"] = BitsAndBytesConfig(**qargs)
                else:
                    raise ValueError(f"Unsupported approach: {approach}")

                LOG.info(f"Loading model from {model_id}; args: {loader_args}")
                model = AutoModelForCausalLM.from_pretrained(model_id, **loader_args)
                LOG.info(f"{model_id}x{approach} loaded successfully. Storing at {approach_dir}")
                # save
                approach_dir.mkdir(parents=True, exist_ok=True)
                tokenizer.save_pretrained(approach_dir)
                model.save_pretrained(approach_dir)
                LOG.info(f"Model {model_id} x {approach} saved successfully at {approach_dir}")
                flag_file.touch()

            except Exception as e:
                LOG.error(f"Error loading model {model_id} x {approach}: {e}", exc_info=True)


class LLMWrapper:

    def __init__(self, model_dir: Path, device: str = DEVICE,
                use_chat_template=True, prompt_template=TRANSLATE_PROMPT,
                progress_bar=False, model_name=None, approach=None):
        assert model_dir.exists(), f"Model directory {model_dir} does not exist"
        assert model_dir.is_dir(), f"Model directory {model_dir} is not a directory"
        self.model_dir = model_dir
        self.device = device
        self.use_chat_template = use_chat_template
        self.prompt_template = prompt_template
        self.progress_bar = progress_bar
        self.model_name = model_name or model_dir.parent.name
        self.approach = approach or model_dir.name
        self._tokenizer = None
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = AutoModelForCausalLM.from_pretrained(self.model_dir, device_map="auto")
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir, use_fast=True)
        return self._tokenizer


    def translate_file(self, pair:str, src_file:Path, out_file:Path, batch_size:int=DEF_BATCH_SIZE):
        """
        Translate a file using the model and tokenizer
        """
        if out_file.exists() and out_file.stat().st_size > 0:
            LOG.info(f"Output file {out_file} already exists; skipping")
            return
        src, tgt = pair.split("-")
        src = LANGS_MAP[src]
        tgt = LANGS_MAP[tgt]
        LOG.info(f"Translating {src_file} to {out_file}")
        src_lines = src_file.read_text().splitlines()
        out_lines = []
        pbar_args = dict(
            desc=f"Decoding {src_file.name}", disable=not self.progress_bar, leave=False, dynamic_ncols=True
        )
        max_length = self.model.config.max_position_embeddings
        gen_args = dict(max_new_tokens=max_length, do_sample=False, num_beams=1)
        device = self.model.device
        self.model.eval()
        torch.set_grad_enabled(False)
        LOG.info(f"Batch size {batch_size}; max_length {max_length}")

        def make_batches(lines, batch_size):
            for i in range(0, len(lines), batch_size):
                yield lines[i : i + batch_size]

        # add line numbers to src_lines for sorting and restoring order
        src_lines = list(enumerate(src_lines, start=0))
        # sort by length; we tokenize the text to get the length
        def find_length(x):
            return len(self.tokenizer(x, add_special_tokens=False))
        src_lines = list(sorted(src_lines, key=lambda x: find_length(x[1]), reverse=True))

        batches = list(make_batches(src_lines, batch_size))
        pbar = tqdm(total=len(src_lines), **pbar_args)

        for batch in batches:
            line_nums = [x[0] for x in batch]
            lines = [x[1] for x in batch]
            lines = [self.prompt_template.format(src=src, tgt=tgt, text=line) for line in lines]
            if self.use_chat_template:
                chats = [[{"role": "user", "content": line}] for line in lines]
                lines = [self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True) for chat in chats]

            inputs = self.tokenizer(lines, return_tensors="pt",  padding=True, add_special_tokens=not self.use_chat_template)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = self.model.generate(**inputs, **gen_args)
            input_len = inputs["input_ids"].shape[1]
            # inputs are padded with right alignment; so we just crop entire input
            outputs = outputs[:, input_len:]
            hyps = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            hyps = [ol.replace("\n", " ").replace("\r", "") for ol in hyps]

            out_lines.extend(zip(line_nums, hyps))
            pbar.update(len(batch))
            #LOG.info(f"\n  SRC: {line}\n  OUT: {out_line}")

        pbar.close()
        # restore input order
        out_lines = list(sorted(out_lines, key=lambda x: x[0]))
        assert len(out_lines) == len(src_lines), f"Expected {len(src_lines)} lines but got {len(out_lines)}"
        for i in range(len(src_lines)):
            assert i == out_lines[i][0], f"Expected line number {i} but got {out_lines[i][0]}"
        out_lines = [x[1] for x in out_lines]
        LOG.info(f"Writing {len(out_lines)} lines to {out_file}")
        out_file.write_text("\n".join(out_lines))

    def get_score(self, src_file: Path, out_file: Path, ref_file: Path, metric: str):
        LOG.warning(f"Compute {metric} score for {src_file} -> {out_file} using {ref_file}")
        if metric == "chrf":
            cmd = f"sacrebleu {ref_file} -i {out_file} -m {metric} -b -lc"
        elif "comet" in metric:
            cmd = f"pymarian-eval -m {metric} -r {ref_file} -t {out_file} -s {src_file} -a only"
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        LOG.info(f"Running command: {cmd}")
        output = sp.check_output(cmd, shell=True, text=True)
        return output.strip()

    def evaluate(
        self,
        work_dir: Path,
        langs: List[str] = DEF_LANG_PAIRS,
        metrics: List[str] = TASK_CONF["metrics"],
        batch_size: int = DEF_BATCH_SIZE,
    ):
        # model-name/approach-name
        LOG.info(f"Evaluating model {self.model_name} x {self.approach} from {self.model_dir}")

        for pair in langs:
            src, tgt = pair.split("-")
            lang_dir = work_dir / "tests" / pair
            assert lang_dir.exists(), f"Language directory {lang_dir} does not exist"
            assert lang_dir.is_dir(), f"Language directory {lang_dir} is not a directory"

            for src_file in lang_dir.glob(f"*.{src}-{tgt}.{src}"):
                test_name = src_file.name.replace(f".{src}-{tgt}.{src}", "")
                ref_file = lang_dir / f"{test_name}.{src}-{tgt}.{tgt}"
                out_file = lang_dir / f"{test_name}.{src}-{tgt}.{tgt}.{self.model_name}.{self.approach}.out"
                if out_file.exists() and out_file.stat().st_size > 0:
                    LOG.info(f"Output file {out_file} already exists; skipping")
                else:
                    self.translate_file(pair, src_file=src_file, out_file=out_file, batch_size=batch_size)
                for metric in metrics:
                    score_file = out_file.with_name(out_file.name + f".{metric}.score")
                    if score_file.exists() and score_file.stat().st_size > 0:
                        LOG.info(f"Score file {score_file} already exists; skipping")
                    else:
                        score = self.get_score(src_file=src_file, out_file=out_file, ref_file=ref_file, metric=metric)
                        score_file.write_text(str(score))
                        LOG.info(f"Score: {score_file.name} : {score}")
        LOG.info(f"Evaluation completed for {self.model_name} x {self.approach}")

def report(work_dir: Path, format: str = "tsv"):
    score_files = work_dir.glob("tests/*/*.score")
    #[metric][lang][testset][model][approach]
    rows = []
    for sf in score_files:
        if sf.stat().st_size == 0:
            LOG.warning(f"Empty score file {sf}")
            continue
        tag = sf.name.replace(".score", "")
        score = sf.read_text().strip()
        rows.append((tag, score))
    #TODO: break tag into parts and make a nice table
    rows = list(sorted(rows))
    delim = "\t" if format == "tsv" else ","
    for row in rows:
        print(delim.join(row))

def main():
    LOG.debug(f"torch version: {torch.__version__}; transformers version: {torch.__version__}")
    LOG.debug(f"torch cuda available: {torch.cuda.is_available()}; device count: {torch.cuda.device_count()}")
    args = parse_args()

    if args.command == "setup":
        setup_task(work_dir=args.work, cache_dir=args.cache)
    elif args.command == "eval":
        llm = LLMWrapper(
            model_dir=args.model,
            use_chat_template=True,
            prompt_template=args.prompt,
            progress_bar=args.progress,
        )
        llm.evaluate(
            work_dir=args.work,
            langs=args.langs,
            metrics=TASK_CONF["metrics"],
            batch_size=args.batch_size,
        )
    elif args.command == "report":
        report(work_dir = args.work, format=args.format)
    else:
        raise ValueError(f"Unknown command: {args.command}")

def parse_args():
    parser = argparse.ArgumentParser(description="wmt25 model compression")

    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub-commands")

    # Common arguments
    def add_common_args(p):
        p.add_argument(
            "-w", "--work", type=Path, default=WORK_DIR, help="Working directory to store models, test sets and results"
        )

    # setup subparser
    setup = subparsers.add_parser(
        "setup",
        help="Download model and make 8bit and 4bit baselines",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(setup)
    setup.add_argument("-c", "--cache", type=Path, default=HF_CACHE, help="Hugging Face cache directory")

    # Evaluate subparser
    eval_parser = subparsers.add_parser(
        "eval", help="Evaluate models", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_common_args(eval_parser)
    eval_parser.add_argument(
        "-m", "--model", type=Path, help=f"Model dir path. example {WORK_DIR}/models/{DEF_MODEL_ID.split('/')[-1]}/base"
    )
    eval_parser.add_argument(
        "-l", "--langs", type=str, nargs="+", default=DEF_LANG_PAIRS, help="Language pairs to evaluate."
    )
    eval_parser.add_argument(
        "-p", "--prompt", type=str, default=TRANSLATE_PROMPT, help="Prompt template to use for translation"
    )
    eval_parser.add_argument(
        "-b", "--batch", dest="batch_size", type=int, default=DEF_BATCH_SIZE, help="Batch size for translation"
    )
    eval_parser.add_argument("-pb", "--progress", action="store_true", help="Show progress bar")

    # report
    report_parser = subparsers.add_parser(
        "report", help="Report scores", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_common_args(report_parser)
    report_parser.add_argument(
        "-f", "--format",
        choices=["csv", "tsv"],
        default="tsv",
        help="Output format for the report (csv or tsv)",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
