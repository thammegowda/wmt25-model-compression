#!/usr/bin/env bash

# Debug node setup

bash -c "$(curl -fsSL https://raw.githubusercontent.com/thammegowda/dotfiles/master/setup.bash)"

# --no-deps so we dont mess up torch+huggingface+cuda deps
pip install -e . --no-deps
pip install sacrebleu mtdata[hf] pymarian

mkdir -p ~/.cache
# for pymarian-eval
ln -sf /mnt/tg/data/cache/marian ~/.cache/marian

# baseline model
mkdir -p ~/.cache/huggingface/hub/
cp -rv  /mnt/tg/data/cache/huggingface/hub/models--CohereLabs--aya-expanse-8b ~/.cache/huggingface/hub/
