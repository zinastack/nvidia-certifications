# =============================================================================
# Lab: Triton Inference Server
# -----------------------------------------------------------------------------
# These targets run ON the Brev instance (make triton-shell first), or are
# reached from Generative-AI/NCP-GENL/Makefile as `make triton-<target>`.
#
# Goal of the lab: read every knob in config.pbtxt and every server flag, then
# watch the numbers move in perf_analyzer.
# =============================================================================

SHELL         := /bin/bash
.DEFAULT_GOAL := help

# Check the current tag: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver
TRITON_VERSION ?= 25.08
SERVER_IMAGE   := nvcr.io/nvidia/tritonserver:$(TRITON_VERSION)-py3
SDK_IMAGE      := nvcr.io/nvidia/tritonserver:$(TRITON_VERSION)-py3-sdk

WORKDIR        ?= $(CURDIR)
MODEL_REPO     ?= $(WORKDIR)/model_repository
MODEL          ?= densenet_onnx
GPUS           ?= 1

# Server flags worth knowing - see README "Server flags" table.
SERVER_FLAGS   ?= --model-repository=/models \
                  --model-control-mode=explicit \
                  --load-model=$(MODEL) \
                  --log-verbose=1 \
                  --strict-readiness=true \
                  --exit-on-error=true

.PHONY: help fetch serve serve-bg logs stop health ready meta config stats metrics client perf analyze clean

help: ## Show this help
	  grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) \
	    | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

fetch: ## Clone the example model repository (densenet_onnx, inception, simple)
	test -d $(MODEL_REPO) || ( \
	  git clone -b r$(TRITON_VERSION) --depth 1 https://github.com/triton-inference-server/server.git /tmp/triton-server && \
	  cp -r /tmp/triton-server/docs/examples/model_repository $(MODEL_REPO) && \
	  cd $(MODEL_REPO)/.. && bash /tmp/triton-server/docs/examples/fetch_models.sh )
	find $(MODEL_REPO) -maxdepth 2 -name config.pbtxt

serve: ## Run tritonserver in the foreground (Ctrl-C to stop)
	docker run --gpus=$(GPUS) --rm --net=host \
	  -v $(MODEL_REPO):/models \
	  --name triton \
	  $(SERVER_IMAGE) tritonserver $(SERVER_FLAGS)

serve-bg: ## Run tritonserver detached
	docker run --gpus=$(GPUS) -d --net=host \
	  -v $(MODEL_REPO):/models \
	  --name triton \
	  $(SERVER_IMAGE) tritonserver $(SERVER_FLAGS)

logs: ## Tail the server log
	docker logs -f triton

stop: ## Stop and remove the server container
	-docker rm -f triton

# ---- inspection: every one of these maps to a KServe v2 endpoint -------------
health: ## GET /v2/health/live  - is the process up
	curl -sf -o /dev/null -w 'live: %{http_code}\n' localhost:8000/v2/health/live

ready: ## GET /v2/health/ready - are the models loaded
	curl -sf -o /dev/null -w 'ready: %{http_code}\n' localhost:8000/v2/health/ready

meta: ## GET /v2 and /v2/models/<MODEL> - server + model metadata
	curl -s localhost:8000/v2 | python3 -m json.tool
	curl -s localhost:8000/v2/models/$(MODEL) | python3 -m json.tool

config: ## GET /v2/models/<MODEL>/config - the config Triton actually resolved
	curl -s localhost:8000/v2/models/$(MODEL)/config | python3 -m json.tool

stats: ## GET /v2/models/<MODEL>/stats - queue vs compute time breakdown
	curl -s localhost:8000/v2/models/$(MODEL)/stats | python3 -m json.tool

metrics: ## Prometheus metrics on :8002
	curl -s localhost:8002/metrics | grep -E '^nv_inference_(request|queue|compute)' | head -30

# ---- clients and benchmarking ----------------------------------------------
client: ## Run the example image client against <MODEL>
	docker run -it --rm --net=host $(SDK_IMAGE) \
	  /workspace/install/bin/image_client -m $(MODEL) -c 3 -s INCEPTION /workspace/images/mug.jpg

perf: ## perf_analyzer sweep: concurrency 1..8, the core latency/throughput view
	docker run -it --rm --net=host $(SDK_IMAGE) \
	  perf_analyzer -m $(MODEL) \
	    --percentile=95 \
	    --concurrency-range 1:8:1 \
	    --shared-memory none

analyze: ## model_analyzer profile - searches instance_count x batch size
	docker run -it --rm --gpus=$(GPUS) --net=host \
	  -v /var/run/docker.sock:/var/run/docker.sock \
	  -v $(MODEL_REPO):/models \
	  $(SDK_IMAGE) model-analyzer profile \
	    --model-repository=/models \
	    --profile-models=$(MODEL) \
	    --triton-launch-mode=docker \
	    --output-model-repository-path=/tmp/analyzer-output \
	    --override-output-model-repository

clean: stop ## Remove the fetched model repository
	rm -rf $(MODEL_REPO) /tmp/triton-server
