# wmt25-model-compression
This repo contains setup and baselines for WMT25 Model Compression task.
For more details about the shared task, visit https://www2.statmt.org/wmt25/model-compression.html 



## Changelog / Announcements
* 2025-04-30: setup and baseline tools


TODOS:
1. support newer metric like comet (Currently chrf is used for demo)
1. eng-ara dataset

> NOTE: watch this repository for updates on the above TODOs


## Setup

```bash
pip install -r requirements.txt

# optional: login to huggingface hub to access gated models
# https://huggingface.co/CohereLabs/aya-expanse-8b
huggingface-cli login

# download models and test sets
python run.py setup
```

The script assumes `./wmt25-compression` as the default work directory. The work directory can be changed with `-w|--work` CLI argument


The above step downloads models and tests under $WORK_DIR
```
tree -h $WORK_DIR
[4.0K]  wmt25-compression/
├── [4.0K]  models
│   └── [4.0K]  aya-expanse-8b
│       ├── [4.0K]  base
│       │   ├── [ 707]  config.json
│       │   ├── [ 137]  generation_config.json
│       │   ├── [4.6G]  model-00001-of-00004.safetensors
│       │   ├── [4.6G]  model-00002-of-00004.safetensors
│       │   ├── [4.7G]  model-00003-of-00004.safetensors
│       │   ├── [1.1G]  model-00004-of-00004.safetensors
│       │   ├── [ 20K]  model.safetensors.index.json
│       │   ├── [ 439]  special_tokens_map.json
│       │   ├── [ 19M]  tokenizer.json
│       │   └── [8.4K]  tokenizer_config.json
│       ├── [4.0K]  bnb-4bit
│       │   ├── [1.2K]  config.json
│       │   ├── [ 137]  generation_config.json
│       │   ├── [4.6G]  model-00001-of-00002.safetensors
│       │   ├── [999M]  model-00002-of-00002.safetensors
│       │   ├── [ 84K]  model.safetensors.index.json
│       │   ├── [ 439]  special_tokens_map.json
│       │   ├── [ 19M]  tokenizer.json
│       │   └── [8.4K]  tokenizer_config.json
│       └── [4.0K]  bnb-8bit
│           ├── [1.2K]  config.json
│           ├── [ 137]  generation_config.json
│           ├── [4.6G]  model-00001-of-00002.safetensors
│           ├── [3.8G]  model-00002-of-00002.safetensors
│           ├── [ 57K]  model.safetensors.index.json
│           ├── [ 439]  special_tokens_map.json
│           ├── [ 19M]  tokenizer.json
│           └── [8.4K]  tokenizer_config.json
└── [4.0K]  tests
    ├── [4.0K]  ces-deu
    │   ├── [267K]  wmt19.ces-deu.ces
    │   └── [289K]  wmt19.ces-deu.deu
    └── [4.0K]  jpn-zho
        ├── [190K]  wmt24.jpn-zho.jpn
        └── [145K]  wmt24.jpn-zho.zho
```

Note: the `tests/` downloaded here are for development purpose. The actual testsets used for shared task evaluation will be different. See shared task webpage for details.


## Run Baselines

```bash
for m in $WORK_DIR/models/aya-expanse-8b/*; do
    python run.py eval -m $m -pb;
done

# report
$ python run.py report
wmt19.ces-deu.deu.aya-expanse-8b.base.out.chrf  54.5
wmt19.ces-deu.deu.aya-expanse-8b.bnb-4bit.out.chrf      54.2
wmt19.ces-deu.deu.aya-expanse-8b.bnb-8bit.out.chrf      54.5
wmt24.jpn-zho.zho.aya-expanse-8b.base.out.chrf  24.4
wmt24.jpn-zho.zho.aya-expanse-8b.bnb-4bit.out.chrf      23.4
wmt24.jpn-zho.zho.aya-expanse-8b.bnb-8bit.out.chrf      24.8
```

## CLI Options

### subcommands
```bash
python run.py -h
usage: run.py [-h] {setup,eval,report} ...

wmt25 model compression

positional arguments:
  {setup,eval,report}  Sub-commands
    setup              Download model and make 8bit and 4bit baselines
    eval               Evaluate models
    report             Report scores

options:
  -h, --help           show this help message and exit
```

### setup
```bash
python run.py setup -h
usage: run.py setup [-h] [-w WORK] [-c CACHE]

options:
  -h, --help            show this help message and exit
  -w WORK, --work WORK  Working directory to store models, test sets and results (default: wmt25-compression)
  -c CACHE, --cache CACHE
                        Hugging Face cache directory (default: /mnt/home/tg/.cache/huggingface/hub)
```

### eval
```bash
python run.py eval -h 
usage: run.py eval [-h] [-w WORK] [-m MODEL] [-l LANGS [LANGS ...]] [-p PROMPT] [-b BATCH_SIZE] [-pb]

options:
  -h, --help            show this help message and exit
  -w WORK, --work WORK  Working directory to store models, test sets and results (default: wmt25-compression)
  -m MODEL, --model MODEL
                        Model dir path. example wmt25-compression/models/aya-expanse-8b/base (default: None)
  -l LANGS [LANGS ...], --langs LANGS [LANGS ...]
                        Language pairs to evaluate. (default: ['ces-deu', 'jpn-zho'])
  -p PROMPT, --prompt PROMPT
                        Prompt template to use for translation (default: Translate the following text from {src} to {tgt}. {text} )
  -b BATCH_SIZE, --batch BATCH_SIZE
                        Batch size for translation (default: 16)
  -pb, --progress       Show progress bar (default: False)
```

### report
```bash
python run.py report -h 
usage: run.py report [-h] [-w WORK] [-f {csv,tsv}]

options:
  -h, --help            show this help message and exit
  -w WORK, --work WORK  Working directory to store models, test sets and results (default: wmt25-compression)
  -f {csv,tsv}, --format {csv,tsv}
                        Output format for the report (csv or tsv) (default: tsv)
```