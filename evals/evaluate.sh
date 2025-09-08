#!/usr/bin/env bash

work=$PWD/workdir
warmup_runs=3
full_runs=3
# assuming participant name is set as Amulet job name
backup=/mnt/tg/data/projects/wmt25-model-compression/evals/backup-v2/$AMLT_JOB_NAME


if [[ "$1" == "--baseline" ]]; then
    eval_baseline=1
else
    eval_baseline=0
fi

python -m modelzip.setup -t eval -w $work

if [[ $eval_baseline == "1" ]]; then
    # baseline + run compression
    python -m modelzip.setup -t model -w $work
    # pip install bitsandbytes --no-deps
    python -m modelzip.compress -m $work/models/aya-expanse-8b-base
    models=($(ls -d $work/models/aya-expanse-8b*))
    #if [[ $(basename "$backup") != *baseline* ]]; then
    # trying to run baseline evaluation inside a participant's amulet job?
    #    backup=$(dirname $backup)/baseline-01
    #fi
else
    models=($(ls -d /model/*))
    # one participant placed their run.py script here and
    # their run.sh wrapper expects run.py in PWD, so we cd here
    if [[ -f /work/wmt25-model-compression/run.py ]]; then
        cd /work/wmt25-model-compression
    fi
fi

echo "Models: ${models[@]}"
echo "Backup: $backup"

metrics="chrf wmt22-comet-da wmt22-cometkiwi-da wmt23-cometkiwi-da-xl"
# get outputs on all supported lang pairs for each model so we can do quality assessment
for m in ${models[@]}; do
    echo "=====Full eval on $m with batch size $batch_size====="
    python -m modelzip.evaluate -w $work -B $backup -r 1 -M $metrics -m $m -b 8 # -l all -t all
done

# this is for speed benchmark; try different batch sizes
for batch_size in 1 16 64 256 512; do
    for m in ${models[@]}; do
        echo "====warming $m====="
        python -m modelzip.evaluate -w $work -B $backup -r $warmup_runs -M $metrics -m $m -b 1 -l ces-deu -t warmup;

        echo "=====Speed eval for $m with batch size $batch_size on ces-deu wmt25====="
        python -m modelzip.evaluate -w $work -B $backup -r $full_runs -M $metrics -m $m -b $batch_size -l ces-deu -t wmt25
    done
done
