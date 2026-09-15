#!/usr/bin/env bash
# Runs once on first boot of the Brev instance (passed via --startup-script).
set -euxo pipefail

# Brev images ship Docker + the NVIDIA container toolkit; verify rather than assume.
nvidia-smi
docker run --rm --gpus all nvcr.io/nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi

# Log in to NGC if you need gated containers: docker login nvcr.io -u '$oauthtoken'
docker pull nvcr.io/nvidia/tritonserver:25.08-py3
docker pull nvcr.io/nvidia/tritonserver:25.08-py3-sdk

sudo apt-get update -y && sudo apt-get install -y jq make git
