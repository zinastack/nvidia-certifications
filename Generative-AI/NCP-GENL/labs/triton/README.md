# Lab: Triton Inference Server

**Goal** — deploy a model on Triton and understand what *every* knob does: the
`config.pbtxt` fields, the `tritonserver` CLI flags, and the KServe v2 endpoints
that let you read back what Triton actually resolved.

Run everything through the [Makefile](Makefile) (`make help`), or from
`NCP-GENL/` as `make triton-<target>`.

---

## 1. Reading list

The official quickstart gets a container running but says almost nothing about
parameters. These are the resources worth the time, best first:

| Resource | Why |
| --- | --- |
| [Triton Conceptual Guide, Parts 1–6](https://github.com/triton-inference-server/tutorials/tree/main/Conceptual_Guide) | **Start here.** NVIDIA's own tutorial series, and the cleanest thing that exists. Part 1 model deployment, [Part 2 dynamic batching + concurrent execution](https://github.com/triton-inference-server/tutorials/tree/main/Conceptual_Guide/Part_2-improving_resource_utilization), Part 3 Model Analyzer, Part 4 accelerating models, Part 5 ensembles. Each part is a runnable repo, not prose. |
| [SoftwareMill — Triton Inference Server Tips & Tricks](https://softwaremill.com/triton-inference-server-tips-and-tricks/) | The clean third-party blog you were looking for. `config.pbtxt` cheat sheets, dynamic batching, instance groups, model warmup, response cache, TensorRT/OpenVINO acceleration, and real `perf_analyzer` / `model_analyzer` runs with benchmark numbers on MobileNetV2. |
| [Model Configuration](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html) | The reference. Not a tutorial — use it to look up a field once you know it exists. |
| [Optimization](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/optimization.html) + [Performance Tuning](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/performance_tuning.html) | The decision tree for *which* parameter to reach for when latency or throughput is wrong. |
| [The Neural Base — Triton beginner course](https://theneuralbase.com/triton/learn/beginner/config-pbtxt-per-model/) | 49 short lessons, one concept each. Good for filling gaps; lighter than the guides above. |
| [Quickstart](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/getting_started/quickstart.html) | Where you started — keep it only for the exact `docker run` invocation. |

---

## 2. The model repository

Triton is driven entirely by a directory layout. Get this wrong and nothing loads.

```text
model_repository/
└── densenet_onnx/
    ├── config.pbtxt          # the parameters (this lab)
    ├── densenet_labels.txt   # optional label file
    └── 1/                    # version directory - must be an integer
        └── model.onnx        # name is backend-specific (see default_model_filename)
```

```bash
make fetch     # clones the example repo and downloads the models
```

---

## 3. `config.pbtxt` parameters

### Required / near-required

| Field | What it controls | Notes |
| --- | --- | --- |
| `name` | Model name | Must match the directory name. Omit it and Triton infers it. |
| `platform` / `backend` | Which backend loads the model | `onnxruntime_onnx`, `tensorrt_plan`, `pytorch_libtorch`, `tensorflow_savedmodel`, `python`, `vllm`, `ensemble`. |
| `max_batch_size` | Largest batch Triton will form | `0` = model does **not** support batching; the first dim of `dims` is then a real dimension, not the batch dim. This is the #1 source of shape errors. |
| `input` / `output` | Tensor contract | `name`, `data_type` (`TYPE_FP32`, `TYPE_INT64`, `TYPE_STRING`…), `dims`. Names must match the model's own tensor names exactly. |

Inside `input`/`output`:

| Field | Meaning |
| --- | --- |
| `dims` | Shape **excluding** the batch dim when `max_batch_size > 0`. `-1` = variable. |
| `reshape` | Bridges a shape mismatch between what the client sends and what the model wants (e.g. dropping a size-1 dim). |
| `format` | `FORMAT_NCHW` / `FORMAT_NHWC` for image inputs. |
| `optional: true` | Input may be omitted by the client. |
| `allow_ragged_batch` | Lets variable-length inputs batch without padding. |
| `is_shape_tensor` | TensorRT shape tensors. |

### Performance knobs — the ones the exam and real life care about

| Block | Field | Effect |
| --- | --- | --- |
| `instance_group` | `count` | How many copies of the model run concurrently. More instances overlap memory transfer with compute. Start at 1, raise while throughput climbs. |
| | `kind` | `KIND_GPU`, `KIND_CPU`, `KIND_AUTO`, `KIND_MODEL`. |
| | `gpus: [0,1]` | Pins instances to specific devices. |
| | `rate_limiter` | Per-instance resource budget; only active with `--rate-limit=execution_count`. |
| `dynamic_batching` | *(empty block)* | Enables server-side batching with defaults — often the single biggest throughput win. |
| | `preferred_batch_size` | Batch sizes Triton tries to form, e.g. `[4, 8]`. |
| | `max_queue_delay_microseconds` | How long to wait for more requests before dispatching. **The latency/throughput dial.** |
| | `preserve_ordering` | Responses returned in request order (costs throughput). |
| | `priority_levels` / `default_priority_level` | Multiple priority queues. |
| | `default_queue_policy` | `timeout_action` (`REJECT`/`DELAY`), `default_timeout_microseconds`, `max_queue_size`, `allow_timeout_override`. |
| `sequence_batching` | `max_sequence_idle_microseconds`, `control_input`, `oldest`/`direct` | Stateful models: routes a correlation ID to the same instance. `START`, `END`, `READY`, `CORRID` control tensors. |
| `ensemble_scheduling` | `step` | Wires models into a DAG (pre-process → model → post-process) with `input_map`/`output_map`. |
| `optimization` | `execution_accelerators` | TensorRT or OpenVINO acceleration of an ONNX/TF model, with `precision_mode: FP16`. |
| | `cuda { graphs: true }` | CUDA graph capture — cuts launch overhead for small models. |
| | `input_pinned_memory` / `output_pinned_memory` | Pinned host buffers for faster H2D/D2H. |
| | `gather_kernel_buffer_threshold` | Fuses batched input copies. |
| `model_warmup` | `batch_size`, `inputs`, `count` | Runs sample inferences at load time so the first real request is not the slow one. |
| `response_cache` | `enable: true` | Caches identical requests. Needs `--cache-config` on the server. |
| `version_policy` | `{ latest: { num_versions: 2 } }` | Or `all {}` / `specific { versions: [1,3] }`. |
| `model_transaction_policy` | `decoupled: true` | One request → many responses. **Required for streaming LLM token output.** |
| `parameters` | `{ key: "..." value: { string_value: "..." } }` | Free-form, backend-specific (how the vLLM and TensorRT-LLM backends are configured). |
| `default_model_filename` | | Override `model.onnx` / `model.plan` / `model.pt`. |

### A worked example

```protobuf
name: "densenet_onnx"
platform: "onnxruntime_onnx"
max_batch_size: 8

input [
  {
    name: "data_0"
    data_type: TYPE_FP32
    format: FORMAT_NCHW
    dims: [ 3, 224, 224 ]
    reshape { shape: [ 1, 3, 224, 224 ] }
  }
]

output [
  {
    name: "fc6_1"
    data_type: TYPE_FP32
    dims: [ 1000 ]
    reshape { shape: [ 1, 1000, 1, 1 ] }
    label_filename: "densenet_labels.txt"
  }
]

instance_group [
  { count: 2, kind: KIND_GPU, gpus: [ 0 ] }
]

dynamic_batching {
  preferred_batch_size: [ 4, 8 ]
  max_queue_delay_microseconds: 100
}

model_warmup [
  { name: "warmup", batch_size: 1,
    inputs { key: "data_0" value: { data_type: TYPE_FP32, dims: [3,224,224], zero_data: true } } }
]
```

---

## 4. `tritonserver` flags

| Flag | Effect |
| --- | --- |
| `--model-repository` | Repeatable — Triton can serve from several repositories. |
| `--model-control-mode` | `none` (load all at start), `poll` (re-scan the directory), `explicit` (load/unload over the API). |
| `--load-model` | With `explicit` mode, which models to load at start. `*` = all. |
| `--repository-poll-secs` | Poll interval for `poll` mode. |
| `--strict-model-config` | `false` lets Triton auto-generate config for backends that can describe themselves. |
| `--backend-config` | Per-backend settings, e.g. `--backend-config=python,shm-default-byte-size=16777216`. |
| `--http-port` / `--grpc-port` / `--metrics-port` | Default `8000` / `8001` / `8002`. |
| `--allow-http`, `--allow-grpc`, `--allow-metrics` | Turn protocols off. |
| `--pinned-memory-pool-byte-size` | Host staging buffer. |
| `--cuda-memory-pool-byte-size` | Per-device pool, e.g. `0:512000000`. |
| `--rate-limit` | `off` or `execution_count`, enables `instance_group.rate_limiter`. |
| `--cache-config` | e.g. `local,size=1048576` — required for `response_cache`. |
| `--buffer-manager-thread-count` | Threads for input/output copies. |
| `--log-verbose` | `1`+ prints the resolved config and per-request scheduling. Indispensable. |
| `--exit-on-error`, `--strict-readiness` | Startup and `/health/ready` semantics. |

---

## 5. Walkthrough

```bash
# from NCP-GENL/
make triton-up && make triton-push && make triton-shell
# on the instance
cd triton-lab
make fetch
make serve-bg && make logs        # --log-verbose=1 prints the resolved config

# in a second local terminal
make triton-fwd

make health ready                 # 200/200
make config                       # the config Triton resolved, defaults filled in
make client                       # example inference
make perf                         # baseline: concurrency 1..8
make stats                        # queue time vs compute time
```

Then iterate — change **one** parameter, `make stop serve-bg perf`, record the number:

1. Baseline, no `dynamic_batching`, `count: 1`.
2. Add `dynamic_batching {}` — throughput should jump, p95 latency should rise slightly.
3. Raise `max_queue_delay_microseconds` — watch throughput and latency trade off.
4. `instance_group.count: 2` — watch queue time in `make stats` fall.
5. `make analyze` — let Model Analyzer search the same space and compare its answer to yours.

### Reading `make stats`

`queue` time high, `compute_infer` low → not enough instances, or batching is waiting
too long. `compute_infer` dominant → the model itself is the bottleneck; go to
`optimization` / TensorRT. `compute_input` + `compute_output` significant → pinned
memory and `gather_kernel_buffer_threshold`.

---

## 6. Troubleshooting

| Symptom | Cause |
| --- | --- |
| `failed to load ... unexpected shape` | `max_batch_size > 0` but `dims` includes the batch dimension. |
| `model not found` | Version directory isn't an integer, or the file name doesn't match `default_model_filename`. |
| `/v2/health/ready` returns 400 | A model failed to load — read `make logs`, not the HTTP code. |
| Throughput flat as concurrency rises | Dynamic batching not enabled, or one instance saturating the GPU. |
| Client hangs on an LLM backend | `model_transaction_policy { decoupled: true }` missing for streaming. |

---

## 7. What I learned

<!-- fill in as you go: the number that surprised you, the parameter you had backwards -->
