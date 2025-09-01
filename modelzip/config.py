import logging as LOG
import os
from pathlib import Path
from .data import CmdGetter, Wmt25Data


LOG.basicConfig(level=LOG.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Hugging Face cache directory
HF_CACHE = Path(os.getenv("HF_HOME", default=Path.home() / ".cache" / "huggingface")) / "hub"
WORK_DIR = "./workdir"

# reusing hf cache for data; not perfect but for simpler config sake
Wmt25Data.CACHE_DIR = HF_CACHE / "wmt25-modelzip"


# Task configuration
TASK_CONF = {
    "langs": {
        "ces-deu": {
            "warmup": CmdGetter("echo 'ahoj světe\tHallo Welt'"),  # a single sentence dataset
            "wmt19": CmdGetter("sacrebleu -t wmt19 -l cs-de --echo src ref"),
            "wmt25": Wmt25Data("cs-de_DE"),
        },
        "jpn-zho": {
            "wmt24": CmdGetter("sacrebleu -t wmt24 -l ja-zh --echo src ref:refA"),
            "wmt25": Wmt25Data("ja-zh_CN"),
        },
        "eng-ara": {
            "wmt24pp": CmdGetter("mtdata echo Google-wmt24pp-1-eng-ara_SA | sed 's/\\r//g'"),
            "wmt25": Wmt25Data("en-ar_EG"),
        },
    },
    "models": ["CohereLabs/aya-expanse-8b"],
    "metrics": ["chrf", "wmt22-comet-da"],  # "wmt22-cometkiwi-da" is a gated model
}

# Default language pairs
DEF_LANG_PAIRS = list(TASK_CONF["langs"].keys())
# Default batch size for translation
DEF_BATCH_SIZE = 1

# Mapping of language codes to full names
LANGS_MAP = dict(
    ces="Czech",
    deu="German",
    jpn="Japanese",
    zho="Chinese",
    eng="English",
    ara="Arabic",
)

# two-letter language codes just in case we need them
_aliases = """
ces cs
deu de
jpn ja
zho zh
eng en
ara ar
""".strip()
for line in _aliases.splitlines():
    code3, code2 = line.split()
    LANGS_MAP[code2] = LANGS_MAP[code3]


# Default translation prompt template
TRANSLATE_PROMPT = "Translate the following text from {src} to {tgt}.\n{text}\n"
USE_CHAT_TEMPLATE = True
