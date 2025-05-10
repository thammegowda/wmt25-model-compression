# WMT25 Model Compression

This repository provides setup scripts and baseline tools for the WMT25 Model Compression shared task.  
For task details, visit the [official page](https://www2.statmt.org/wmt25/model-compression.html).

---

## Announcements

- **2025-04-12:** Code refactored to facilitate separate submission from eval pipeline
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
pip install -e .
```

### 2. (Optional) Hugging Face Login

```bash
huggingface-cli login
```
*Required for gated models (e.g., [aya-expanse-8b](https://huggingface.co/CohereLabs/aya-expanse-8b)).*

### 3. Download Models & Test Sets

```bash
python -m modelzip.setup
```
- Default work directory: `./wmt25-compression`
- Change with `-w` or `--work` argument

**Directory Structure:**
```
$WORK_DIR/
├── models/
│   └── aya-expanse-8b-base
└── tests/
  ├── ces-deu/
  └── jpn-zho/
```
*Note: Test sets here are for development only. Official evaluation uses different data.*

---
## Compression: demo

```bash
python -m modelzip.compress
```

## Running Baselines

```bash
for m in wmt25-compression/models/aya-expanse-8b-*; do
  python -m modelzip.eval -m $m
done

python -m modelzip.report
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


## Submission Requirements

To participate, submit both a `Dockerfile` and a Docker image containing all necessary software and model files for translation.

- **Naming:**
    Include your team’s short name (no spaces) in both the Dockerfile and image name. E.g: `$Team-Dockerfile` and `$Team-dockerimage.tar`

- **Model Directory:**
    The image must contain a model directory at `/model/$submission_id` with all required files (model, vocabulary, etc.).  l
    > **Note:** Keep `$submission_id` short; it will appear in reports.

- **File Restrictions:**
    You may include additional files, but **do not** use any paths starting with `/wmt`—these are reserved for the evaluation system.

- **Execution Script:**
    Each model directory must include a `run.sh` script with the following interface:

    ```bash
    /model/$submission_id/run.sh $lang_pair $batch_size < input.txt > output.txt
    ```
    - `$lang_pair`: Language pair in the format `eng-deu`
    - `$batch_size`: Positive integer
    - The script must run without accessing Internet.

## Example Usage

```bash
image_name="$(docker load -i ${image_file_path} | cut -d ' ' -f 3)"
container_id="$(docker run -itd ${opt_memory} --memory-swap=0 ${image_name} bash)"
(time docker exec -i "${container_id}" /model/$submission_id/run.sh $lang_pair $batch_size < input.txt > output.txt 2> stderr.txt)
```

