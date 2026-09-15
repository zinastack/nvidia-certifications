# =============================================================================
# NCP-GENL - lab environment control
# -----------------------------------------------------------------------------
# One entry point for every hands-on lab under labs/.
#
#   <lab>-up      create the Brev GPU instance for that lab
#   <lab>-shell   SSH into it
#   <lab>-fwd     forward the lab's ports to localhost
#   <lab>-push    copy the lab folder up to the instance
#   <lab>-stop    stop the instance (keeps the disk, stops the billing clock)
#   <lab>-rm      delete the instance permanently
#
# The actual lab work lives in labs/<lab>/Makefile and runs *on the instance*.
# Every variable below can be overridden on the command line, e.g.
#   make triton-up TRITON_GPU=A100 TRITON_VRAM=40
#
# Brev CLI reference: https://docs.nvidia.com/brev/cli/cli-overview
# =============================================================================

SHELL         := /bin/bash
.DEFAULT_GOAL := help

BREV          ?= brev
LABS_DIR      := labs

# ---- shared instance defaults ----------------------------------------------
MIN_DISK      ?= 256        # GB of disk
TIMEOUT       ?= 900        # seconds to wait for the instance to become ready
CREATE_FLAGS  ?=            # extra flags, e.g. CREATE_FLAGS=--dry-run

# brev_create: $(1)=name $(2)=gpu $(3)=min-vram $(4)=startup script (optional)
define brev_create
	$(BREV) create $(1) \
	    --gpu-name $(2) \
	    --min-vram $(3) \
	    --min-disk $(MIN_DISK) \
	    --timeout $(TIMEOUT) \
	    $(if $(wildcard $(4)),--startup-script @$(4)) \
	    $(CREATE_FLAGS)
endef

# ---- account / fleet --------------------------------------------------------
.PHONY: login ls refresh stop-all help

login: ## Authenticate the Brev CLI and install SSH keys
	$(BREV) login

ls: ## List every instance in the active org
	$(BREV) ls

refresh: ## Re-sync instances created in the web console into the CLI
	$(BREV) refresh

stop-all: ## Stop every running instance (do this before you walk away)
	$(BREV) stop --all

help: ## Show this help
	echo "NCP-GENL labs"
	echo
	grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) \
	    | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# =============================================================================
# Lab: triton - Triton Inference Server (model repository + config.pbtxt)
#   labs/triton/
# =============================================================================
TRITON_DIR      := $(LABS_DIR)/triton
TRITON_INSTANCE ?= genl-triton
TRITON_GPU      ?= L4
TRITON_VRAM     ?= 24
TRITON_PORTS    ?= 8000 8001 8002   # http / grpc / metrics

.PHONY: triton-up triton-shell triton-fwd triton-push triton-stop triton-rm triton-%

triton-up: ## Create the Triton GPU instance
	$(call brev_create,$(TRITON_INSTANCE),$(TRITON_GPU),$(TRITON_VRAM),$(TRITON_DIR)/startup.sh)

triton-shell: ## SSH into the Triton instance
	$(BREV) shell $(TRITON_INSTANCE)

triton-fwd: ## Forward Triton HTTP/gRPC/metrics ports to localhost
	$(BREV) port-forward $(TRITON_INSTANCE) $(foreach p,$(TRITON_PORTS),-p $(p):$(p))

triton-push: ## Copy labs/triton/ onto the instance
	$(BREV) copy $(TRITON_DIR) $(TRITON_INSTANCE):~/triton-lab

triton-stop: ## Stop the Triton instance
	$(BREV) stop $(TRITON_INSTANCE)

triton-rm: ## Delete the Triton instance
	$(BREV) delete $(TRITON_INSTANCE)

# Anything else (triton-serve, triton-perf, ...) is handed to labs/triton/Makefile.
triton-%:
	$(MAKE) -C $(TRITON_DIR) $*

# =============================================================================
# Lab: rapids - RAPIDS / cuDF accelerated data prep
#   labs/rapids/
# =============================================================================
RAPIDS_DIR      := $(LABS_DIR)/rapids
RAPIDS_INSTANCE ?= genl-rapids
RAPIDS_GPU      ?= L4
RAPIDS_VRAM     ?= 24
RAPIDS_PORTS    ?= 8888             # jupyter

.PHONY: rapids-up rapids-shell rapids-fwd rapids-push rapids-stop rapids-rm rapids-%

rapids-up: ## Create the RAPIDS GPU instance
	$(call brev_create,$(RAPIDS_INSTANCE),$(RAPIDS_GPU),$(RAPIDS_VRAM),$(RAPIDS_DIR)/startup.sh)

rapids-shell: ## SSH into the RAPIDS instance
	$(BREV) shell $(RAPIDS_INSTANCE)

rapids-fwd: ## Forward the Jupyter port to localhost
	$(BREV) port-forward $(RAPIDS_INSTANCE) $(foreach p,$(RAPIDS_PORTS),-p $(p):$(p))

rapids-push: ## Copy labs/rapids/ onto the instance
	$(BREV) copy $(RAPIDS_DIR) $(RAPIDS_INSTANCE):~/rapids-lab

rapids-stop: ## Stop the RAPIDS instance
	$(BREV) stop $(RAPIDS_INSTANCE)

rapids-rm: ## Delete the RAPIDS instance
	$(BREV) delete $(RAPIDS_INSTANCE)

rapids-%:
	$(MAKE) -C $(RAPIDS_DIR) $*
