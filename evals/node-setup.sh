#!/usr/bin/env bash

# Debug node setup; should be done automatically in amulet setup

# bash -c "$(curl -fsSL https://raw.githubusercontent.com/thammegowda/dotfiles/master/setup.bash)"

pip install --upgrade pip
pip install -e . --no-deps
pip install sacrebleu mtdata[hf] pymarian

mkdir -p ~/.cache/huggingface/hub/
cp -r /mnt/tg/data/cache/huggingface/hub/models--CohereLabs--aya-expanse-8b ~/.cache/huggingface/hub/

ln -sf /mnt/tg/data/cache/marian ~/.cache/marian