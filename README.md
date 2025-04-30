# WMT25 Model Compression

This repository provides setup scripts and baseline tools for the WMT25 Model Compression shared task.  
For task details, visit the [official page](https://www2.statmt.org/wmt25/model-compression.html).

---

## Announcements

- **2025-04-30:** Initial release with setup and baseline tools.

**Planned:**
- Add support for newer metrics (e.g., COMET; currently using chrF for demo)
- Add English-Arabic dataset

> **Tip:** Watch this repository for updates.

---

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. (Optional) Hugging Face Login

```bash
huggingface-cli login
```
*Required for gated models (e.g., [aya-expanse-8b](https://huggingface.co/CohereLabs/aya-expanse-8b)).*

### 3. Download Models & Test Sets

```bash
python run.py setup
```
- Default work directory: `./wmt25-compression`
- Change with `-w` or `--work` argument

**Directory Structure:**
```
$WORK_DIR/
├── models/
│   └── aya-expanse-8b/
│       ├── base/
│       ├── bnb-4bit/
│       └── bnb-8bit/
└── tests/
  ├── ces-deu/
  └── jpn-zho/
```
*Note: Test sets here are for development only. Official evaluation uses different data.*

---

## Running Baselines

```bash
for m in wmt25-compression/models/aya-expanse-8b/*; do
  python run.py eval -m $m -pb
done

python run.py report
```

**Sample Output:**
```
wmt19.ces-deu.deu.aya-expanse-8b.base.out.chrf      54.5
wmt19.ces-deu.deu.aya-expanse-8b.bnb-4bit.out.chrf  54.2
wmt19.ces-deu.deu.aya-expanse-8b.bnb-8bit.out.chrf  54.5
wmt24.jpn-zho.zho.aya-expanse-8b.base.out.chrf      24.4
...
```

---

## Command-Line Interface

### Subcommands

- `setup`   — Download models and prepare baselines
- `eval`    — Evaluate models
- `report`  — Summarize results

### Common Options

- `-w, --work`   Working directory (default: `wmt25-compression`)
- `-h, --help`   Show help

### Examples

**Setup:**
```bash
python run.py setup -w mydir
```

**Evaluate:**
```bash
python run.py eval -m <MODEL_DIR> -pb -b 64
```

**Report:**
```bash
python run.py report -f tsv
```

---

For more details, run `python run.py <subcommand> -h`.
