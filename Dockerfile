FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
LABEL description="Dockerfile for WMT25 Model Compression Challenge"
LABEL maintainer="Thamme Gowda and WMT25 Organizers"
LABEL version="1.0"
LABEL date="2025-04-29"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install -y git python3 python3-pip emacs vim wget curl ncdu tmux htop
RUN python3 -m pip install --no-cache-dir --upgrade pip

RUN git clone https://github.com/huggingface/transformers && cd transformers && git checkout $REF

COPY requirements.txt /work/requirements.txt
# COPY run.py /work/run.py  # we might update this file, so please get the latest file from the repo

SHELL ["/bin/bash", "-c"]
RUN wget https://raw.githubusercontent.com/ohmybash/oh-my-bash/master/tools/install.sh -O - | bash

WORKDIR /work
RUN python3 -m pip install --no-cache-dir -r requirements.txt
