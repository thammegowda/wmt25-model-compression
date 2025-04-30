FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
LABEL description="Dockerfile for WMT25 Model Compression Challenge"
LABEL maintainer="Thamme Gowda and WMT25 Organizers"
LABEL version="1.0"
LABEL date="2025-04-29"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install -y git python3 python3-pip
RUN python3 -m pip install --no-cache-dir --upgrade pip


RUN git clone https://github.com/huggingface/transformers && cd transformers && git checkout $REF

# latest stable version of PyTorch at the time of writing
ARG PYTORCH='2.7.0'
#     see https://pytorch.org/get-started/locally/
ARG CUDA='cu126'

ARG TRANSFORMERS="4.44.0"
# transformers version from https://huggingface.co/CohereLabs/aya-expanse-8b/blob/main/config.json

COPY requirements.txt /work/requirements.txt
WORKDIR /work
RUN python3 -m pip install --no-cache-dir -r requirements.txt
