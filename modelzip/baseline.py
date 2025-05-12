#!/usr/bin/env python
import argparse
import sys
from typing import List
from pathlib import Path

from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from modelzip.config import LOG, DEF_BATCH_SIZE, LANGS_MAP, TRANSLATE_PROMPT, USE_CHAT_TEMPLATE

LOG.basicConfig(level=LOG.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class LLMWrapper:
    def __init__(self, model_dir: Path, use_chat_template=True,
                prompt_template=TRANSLATE_PROMPT, progress_bar=False):
        self.model_dir = Path(model_dir)
        self.use_chat_template = use_chat_template
        self.prompt_template = prompt_template
        self.progress_bar = progress_bar
        self._tokenizer = None
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = AutoModelForCausalLM.from_pretrained(self.model_dir, device_map="auto")
            self._model.eval()
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir, use_fast=True)
        return self._tokenizer

    def translate_lines(self, pair: str, lines: List[str],
                       batch_size: int = DEF_BATCH_SIZE):
        src, tgt = pair.split("-")
        src_lang = LANGS_MAP[src]
        tgt_lang = LANGS_MAP[tgt]
        indexed = list(enumerate(lines))
        # sort by token length
        def length(x):
            return len(self.tokenizer(x, add_special_tokens=False)["input_ids"])
        indexed.sort(key=lambda x: length(x[1]), reverse=True)

        batches = [indexed[i:i+batch_size] for i in range(0, len(indexed), batch_size)]
        out_indexed = []
        pbar = tqdm(total=len(indexed), disable=not self.progress_bar)
        max_length = self.model.config.max_position_embeddings
        gen_args = dict(max_new_tokens=max_length, do_sample=False, num_beams=1)
        for batch in batches:
            ids, texts = zip(*batch)
            prompts = [self.prompt_template.format(src=src_lang, tgt=tgt_lang, text=t) for t in texts]
            if self.use_chat_template:
                chats = [[{"role": "user", "content": p}] for p in prompts]
                prompts = [self.tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in chats]
            inputs = self.tokenizer(list(prompts), return_tensors="pt", padding=True, add_special_tokens=not self.use_chat_template)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            outputs = self.model.generate(**inputs, **gen_args)
            seq_len = inputs["input_ids"].shape[1]
            hyps = [h.replace("\n", " ") for h in self.tokenizer.batch_decode(outputs[:, seq_len:], skip_special_tokens=True)]
            out_indexed.extend(zip(ids, hyps))
            pbar.update(len(batch))
        pbar.close()
        out_indexed.sort(key=lambda x: x[0])
        results = [t for (_, t) in out_indexed]
        return results


def main():
    args = parse_args()
    llm = LLMWrapper(args.model, use_chat_template=USE_CHAT_TEMPLATE, prompt_template=args.prompt, progress_bar=args.progress)
    if args.inp is sys.stdin:
        LOG.info("Reading from stdin") # just in case if we forget to pass input via STDIN
    # buffering all inputs into one big maxibatch for sorting based on length, assuming test sets are not too big
    lines = args.input.read().splitlines()
    lines = [x.strip() for x in lines] # remove empty lines
    assert not any(x == "" for x in lines), "Input file contains empty lines. Please fix them and try again."
    assert len(lines) > 0, "Input file is empty. Please provide some input."
    outputs = llm.translate_lines(args.langs, lines, batch_size=args.batch_size)
    assert len(outputs) == len(lines), f"Output length {len(outputs)} does not match input length {len(lines)}"
    args.output.write("\n".join(outputs))

def parse_args():
    parser = argparse.ArgumentParser(description="Run translation using LLM",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # eval script only sets these two required args. They are positional args for simplicity
    parser.add_argument("langs",  help="Lang pairs to evaluate, eg, 'ces-deu")
    parser.add_argument("batch", dest="batch_size", type=int, default=DEF_BATCH_SIZE)

    # this script will/should be placed inside model directory for each model and called run.py, so assume this file's parent dir as model dir
    my_name = Path(__file__).name
    my_dir = Path(__file__).parent
    if my_name == "run.py":
        parser.add_argument("-m", "--model", type=Path, default=my_dir, help="Path to model directory that is compatible for HuggingFace Transformers")
    else:
        parser.add_argument("-m", "--model", type=Path, required=True, help="Path to model directory that is compatible for HuggingFace Transformers")

    # these are optional args. They are not positional args for simplicity
    parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, default=DEF_BATCH_SIZE, help="Batch size for translation")
    # optional args. Will not be set during evaluation, so make sure the defaults are correct
    parser.add_argument("-i", "--input", type=argparse.FileType('r', encoding='utf-8', errors='replace'), default=sys.stdin,
                        help="Input file")
    parser.add_argument("-o", "--output", type=argparse.FileType('w', encoding='utf-8', errors='replace'), default=sys.stdout,
                        help="Output file")
    parser.add_argument("-pb", "--progress", action="store_true", help="Show progress bar")
    parser.add_argument("-pt", "--prompt", type=str, default=TRANSLATE_PROMPT, help="Prompt template for translation")
    return parser.parse_args()

if __name__ == '__main__':
    main()
