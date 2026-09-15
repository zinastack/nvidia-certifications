# Lab: RAPIDS / cuDF

**Goal** — GPU-accelerated data preparation for LLM datasets: cuDF as a pandas
drop-in, and where the GPU actually wins.

Status: **planned** — scaffolding only, see the [Makefile](Makefile).

```bash
# from NCP-GENL/
make rapids-up && make rapids-shell
cd rapids-lab && make smoke notebook logs   # token URL is in the log
make rapids-fwd                             # 8888 -> localhost
```

## References

- [RAPIDS docs](https://docs.rapids.ai/)
- [cudf.pandas accelerator](https://docs.rapids.ai/api/cudf/stable/cudf_pandas/)
- [RAPIDS notebooks container on NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/rapidsai/containers/notebooks)

## What I learned

<!-- fill in as you go -->
