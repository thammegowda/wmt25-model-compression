# wmt25-model-compression
WMT25 Model Compression task: setup and baselines






## Setup

```bash
pip install -r requirements.txt

# optional: login to huggingface hub to access gated models
# https://huggingface.co/CohereLabs/aya-expanse-8b
huggingface-cli login

# download models and test sets
WORK_DIR=~/wmt25-compression
python run.py setup -w $WORK_DIR
```

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

## Run Baselines

```bash
for m in $WORK_DIR/models/aya-expanse-8b/*; do
    python run.py eval -w $WORK_DIR -m $m -pb;
done
```
