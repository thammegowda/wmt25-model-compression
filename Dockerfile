FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
LABEL description="Dockerfile for WMT25 Model Compression Shared Task"
LABEL maintainer="Thamme Gowda and WMT25 Organizers"
LABEL version="1.0"
LABEL date="2025-04-29"

ARG DEBIAN_FRONTEND=noninteractive

# !!! Follow the instructions in README.md before building this image !!!

# Install default packages
RUN apt update
RUN apt install -y git python3 python3-pip emacs vim wget curl ncdu tmux htop tree
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Install requirements
COPY requirements.txt /requirements.txt
RUN python3 -m pip install --no-cache-dir -r /requirements.txt

# Copy model files with execution scripts
COPY workdir/models/aya-expanse-8b-base /model/base
COPY workdir/models/aya-expanse-8b-bnb-8bit /model/bnb-8bit
COPY workdir/models/aya-expanse-8b-bnb-4bit /model/bnb-4bit

RUN chmod +x /model/base/run.sh
RUN chmod +x /model/bnb-8bit/run.sh
RUN chmod +x /model/bnb-4bit/run.sh

# Test run.sh scripts
RUN /model/base/run.sh eng-deu 1 <<< "This is a test with the base model."
RUN /model/bnb-8bit/run.sh eng-deu 1 <<< "This is a test with the 8-bit model."
RUN /model/bnb-4bit/run.sh eng-deu 1 <<< "This is a test with the 4-bit model."

# For development
#SHELL ["/bin/bash", "-c"]
#RUN wget https://raw.githubusercontent.com/ohmybash/oh-my-bash/master/tools/install.sh -O - | bash