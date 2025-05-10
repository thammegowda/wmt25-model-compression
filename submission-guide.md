# Docker Submission Guidelines

To participate, submit both a `Dockerfile` and a Docker image containing all necessary software and model files for translation.

## Submission Requirements

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