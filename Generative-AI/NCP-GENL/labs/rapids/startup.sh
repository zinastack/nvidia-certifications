#!/usr/bin/env bash
# Runs once on first boot of the Brev instance (passed via --startup-script).
set -euxo pipefail

nvidia-smi
docker pull nvcr.io/nvidia/rapidsai/notebooks:25.08-cuda12.5-py3.12
sudo apt-get update -y && sudo apt-get install -y jq make git
