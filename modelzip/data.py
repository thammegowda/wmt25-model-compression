import subprocess as sp
import json
from pathlib import Path
from dataclasses import dataclass
import logging
from typing import ClassVar, List

# Initialize logging
LOG = logging.getLogger(__name__)

@dataclass
class CmdGetter:
    cmd: str
    
    def __call__(self) -> list[list[str]]:
        lines = sp.check_output(self.cmd, shell=True, text=True).strip().replace("\r", "").split("\n")
        lines = [x.strip().split("\t") for x in lines]
        if "mtdata" in self.cmd:
            lines = [x[:2] for x in lines]
        return lines

@dataclass
class Wmt25Data:
    pair: str
    line_break: str = " " # replace new lines with this text
        
    # Define HF_CACHE (you may need to adjust this path)
    CACHE_DIR: ClassVar[Path] = Path.home() / ".cache" / "wmt25modelzip"


    @property
    def cache_file(self) -> Path:
        cache_file = self.CACHE_DIR / f"wmt25.jsonl"
        if not (cache_file.exists() and cache_file.stat().st_size > 0):
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            url = "https://data.statmt.org/wmt25/general-mt/wmt25.jsonl"
            cmd = f"wget {url} -O {cache_file}.tmp && mv {cache_file}.tmp {cache_file}"
            sp.run(cmd, shell=True, check=True)
        return cache_file

    def filter(self):
        seen_langs = set()
        count = 0
        with open(self.cache_file, 'r', encoding='utf-8') as lines:
            for line in lines:
                rec = json.loads(line.strip())
                pair = rec["src_lang"] + "-" + rec["tgt_lang"]
                seen_langs.add(pair)
                if pair == self.pair:
                    count += 1
                    yield rec
        if count == 0:
            LOG.warning(f"No records found for {self.pair}. Seen pairs: {seen_langs}")

    def __call__(self) -> list[str]:
        lines = [] 
        for rec in self.filter():
            src = rec["src_text"].replace("\n", self.line_break)
            # tgt unavailable in testset
            lines.append(src)
        return lines
