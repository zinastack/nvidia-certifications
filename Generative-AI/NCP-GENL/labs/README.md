# NCP-GENL Labs

Hands-on labs for the NVIDIA-Certified Professional: Generative AI LLMs exam.
Every lab runs on a GPU instance provisioned through [NVIDIA Brev](https://docs.nvidia.com/brev/),
driven from the [Makefile](../Makefile) one level up.

## Labs

| Lab | Focus | Folder | Instance | Status |
| --- | --- | --- | --- | --- |
| Triton Inference Server | Model repository layout, `config.pbtxt` parameters, dynamic batching, instance groups, `perf_analyzer` | [triton/](triton/) | `genl-triton` | In progress |
| RAPIDS / cuDF | GPU-accelerated data prep for LLM datasets | [rapids/](rapids/) | `genl-rapids` | Planned |

## Running a lab

All lab environments are controlled from `Generative-AI/NCP-GENL/Makefile`:

```bash
cd Generative-AI/NCP-GENL

make help              # every target
make login             # once, to authenticate the Brev CLI

make triton-up         # create the GPU instance for the Triton lab
make triton-push       # copy labs/triton/ onto it
make triton-shell      # SSH in, then: cd triton-lab && make fetch serve
make triton-fwd        # in a second terminal: 8000/8001/8002 -> localhost

make triton-stop       # stop the clock when you take a break
make triton-rm         # delete the instance when the lab is done
make stop-all          # panic button: stop everything
```

Targets that are not part of the instance lifecycle are forwarded to the lab's own
Makefile, so `make triton-perf` from `NCP-GENL/` is the same as `make perf` inside
`labs/triton/`.

> Brev bills per running hour. `make ls` shows what is still up — check it before
> you close the laptop.

## Lab format

Each lab folder carries:

- `README.md` — goal, parameter reference, walkthrough, what I learned
- `Makefile` — the commands, so a lab is repeatable instead of remembered
- `startup.sh` — first-boot provisioning handed to `brev create --startup-script`

## Shared references

- [NVIDIA Brev docs](https://docs.nvidia.com/brev/) · [CLI reference](https://docs.nvidia.com/brev/cli/cli-overview)
- [NGC catalog](https://catalog.ngc.nvidia.com/) — container tags used by the labs
- [Generative AI LLMs Professional exam page](https://www.nvidia.com/en-us/learn/certification/generative-ai-llm-professional/)
