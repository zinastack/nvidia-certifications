# NCP-GENL

NVIDIA-Certified Professional: Generative AI LLMs.

## Contents

- [labs/](labs/) — hands-on labs, each with its own Makefile and Brev GPU instance
- [Makefile](Makefile) — one entry point to spin every lab environment up and down

## Labs

| Lab | Focus | Status |
| --- | --- | --- |
| [Triton Inference Server](labs/triton/) | Model repository, `config.pbtxt` parameters, dynamic batching, instance groups, `perf_analyzer` | In progress |
| [RAPIDS / cuDF](labs/rapids/) | GPU-accelerated data prep | Planned |

## Quick start

```bash
make help        # every target
make login       # authenticate the Brev CLI once
make triton-up   # spin up the Triton lab instance
make ls          # what is still running (and billing)
make stop-all    # stop everything
```

See [labs/README.md](labs/README.md) for the full workflow.

## References

- [Generative AI LLMs Professional exam page](https://www.nvidia.com/en-us/learn/certification/generative-ai-llm-professional/)
- [NVIDIA Brev docs](https://docs.nvidia.com/brev/)
