# Architecture Notes

Use this file for durable mental models, diagrams, and system-level summaries.

## AI Infrastructure Stack

```text
AI workloads and validation suites
Slurm, Enroot, Pyxis, Docker, and cluster tooling
Base Command Manager and operating system
NVIDIA GPU drivers, DOCA drivers, CUDA, container toolkit, NGC CLI
GPU servers, BlueField, switches, cabling, transceivers, storage
Facilities, rack layout, power, cooling, BMC, OOB, TPM, firmware
```

## Questions To Answer

- What is the expected deployment and validation sequence for a GPU server?
- Which checks prove power, cooling, cabling, and firmware are correct?
- How do BCM, Slurm, Enroot, and Pyxis fit into the control plane?
- What do HPL, NCCL, ClusterKit, and storage tests each validate?
- Which failures are hardware, driver, network, storage, or workload-level?
