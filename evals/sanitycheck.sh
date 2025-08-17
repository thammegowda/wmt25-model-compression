#!/usr/bin/env bash
# Created by: TG Gowda on 2025-07-31

# This script sanity checks a participant's docker submission
# Steps to sanity check:
# 1. iterate through run scripts at /model/*/run.sh
# 2. dirname is the submission ID. Print the directory size
# 3. run the script as "echo -e "Sentence1\nSentence2\nSentence3" | bash run.sh $langs $batchsize" where $langs=eng-deu and $batchsize=1
# 4. Check that output is not empty and has same number of lines as input
# 5. If the command does not finish, timeout after 15 minutes
# 6. Decide success or failure for each submission. If a submission fails, print the status and continue to the next one
set -euo pipefail  # Exit on error, undefined variable, or failed command in a pipeline

# Set the timeout duration in seconds
TIMEOUT_DURATION=900  # 15 minutes


# wget https://data.statmt.org/wmt25/general-mt/wmt25.jsonl
# cat wmt25.jsonl | jq -c  'select(.src_lang == "cs" and .tgt_lang == "de_DE")' > wmt25.cs-de.jsonl
# cat wmt25.cs-de.jsonl | jq  .src_text  | sed 's/\\n/ /g' | jq -r . > wmt25.cs-de.txt


# Iterate through each run.sh script in the /model/*/ directory
for run_script in /model/*/run.sh; do
    # Get the submission ID from the directory name
    submission_id=$(basename "$(dirname "$run_script")")
    echo "===Sanity checking submission ID: $submission_id==="
    cd "$(dirname "$run_script")" || {
        echo "Failed to change directory to $(dirname "$run_script")";
        continue
    }
    echo "Submission directory: $(pwd)"
    ls -lh # List the directory contents and size
    echo "Directory size:"
    du -sh . # Show the size of the submission directory

    # Prepare the input sentences
    #input_sentences="Sentence1\nSentence2 Word2 word3\nSentence3"
    input_sentences="Věta1\nVěta2 slovo2 slovo3\nVěta3"
    # Run the script with a timeout
    start_time=$(date +%s)
    langs="ces-deu"
    output=$(echo -e "$input_sentences" | timeout $TIMEOUT_DURATION bash "$run_script" $langs 1) || {
        echo "ERROR: Command failed for submission ID: $submission_id"
        continue
    }
    end_time=$(date +%s)
    elapsed_time=$((end_time - start_time))
    echo "Command executed in $elapsed_time seconds for submission ID: $submission_id"
    echo "Output from run.sh for submission ID $submission_id:"
    echo -e "=====\n$output\n======"
    # Check if the output is empty
    if [[ -z "$output" ]]; then
        echo "ERROR: Output is empty for submission ID: $submission_id"
        continue
    fi
    # Check if the number of lines in the output matches the number of input sentences
    output_lines=$(echo "$output" | wc -l)
    input_lines=$(echo -e "$input_sentences" | wc -l)
    if [[ "$output_lines" -ne "$input_lines" ]]; then
        echo "ERROR: Output line count ($output_lines) does not match input line count ($input_lines) for submission ID: $submission_id"
        continue
    fi
    echo "SUCCESS: Submission ID $submission_id passed the sanity check."
done

echo "Sanity check completed for all submissions. See ERROR messages above for any failures."
exit 0
