# =============================================================================
# Lab: RAPIDS / cuDF
# Reached from Generative-AI/NCP-GENL/Makefile as `make rapids-<target>`.
# =============================================================================

SHELL         := /bin/bash
.DEFAULT_GOAL := help

RAPIDS_IMAGE  ?= nvcr.io/nvidia/rapidsai/notebooks:25.08-cuda12.5-py3.12
WORKDIR       ?= $(CURDIR)
PORT          ?= 8888

.PHONY: help notebook stop logs smoke clean

help: ## Show this help
	  grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) \
	    | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

notebook: ## Start the RAPIDS notebook container on $(PORT)
	docker run -d --gpus all --name rapids \
	  -p $(PORT):8888 -v $(WORKDIR):/home/rapids/work \
	  $(RAPIDS_IMAGE)

logs: ## Tail the notebook log (the token URL is in here)
	docker logs -f rapids

stop: ## Stop and remove the container
	-docker rm -f rapids

smoke: ## Confirm cuDF sees the GPU
	docker run --rm --gpus all $(RAPIDS_IMAGE) \
	  python -c "import cudf; print(cudf.__version__); print(cudf.DataFrame({'a':[1,2,3]}).sum())"

clean: stop ## Remove containers
