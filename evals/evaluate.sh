#!/usr/bin/env bash

work=workdir
warmup_runs=3
full_runs=3
# assuming participant name is set as Amulet job name
backup=/mnt/tg/data/projects/wmt25-model-compression/evals/backup-v2/$AMLT_JOB_NAME

python -m modelzip.setup -t eval -w $work

eval_baseline=0
if [[ $eval_baseline == "1" ]]; then
    # baseline + run compression
    python -m modelzip.setup -t model -w $work
    # pip install bitsandbytes --no-deps
    python -m modelzip.compress -m $work/models/aya-expanse-8b-base
    models=($(echo $work/models/aya-expanse-8b*))
    backup=$(dirname $backup)/baseline-models
else
    models=($(ls -d /model/*))
fi

echo "Models: ${models[@]}"
echo "Backup: $backup"

metrics="chrf wmt22-comet-da wmt22-cometkiwi-da wmt23-cometkiwi-da-xl"
for m in ${models[@]}; do
    echo "====warming up $m====="
    python -m modelzip.evaluate -B $backup -r $warmup_runs -M $metrics -m $m -l ces-deu -b 1 -t warmup;

    for batch_size in 1 8 16 32 128 256; do
        echo "=====Full eval on $m with batch size $batch_size====="
        python -m modelzip.evaluate -B $backup -r $full_runs -M $metrics -m $m -b $batch_size
    done
done

