# Sample10 psana pipeline versus cold read baseline

Measurement in progress. Scope: [scope.md](scope.md).

This harness uses the accepted 10k synthetic MFX fixture and the existing
expansion, calibration and 100-bin/99-active cube kernels unchanged. Each run
is a fresh MPI process group: one SMD0, one EB and N BDs. All BDs see exactly
one physical GPU selected by UUID before MPI initialization. Each BD is pinned
to four distinct physical CPU cores; each helper has one. BDs increase total
CPU allocation and KvikIO workers (four per BD), not just process count.
Device PCI/mask, CPU affinity, runtime hashes and effective settings are saved.

Both cold and warm cases have fresh GPU contexts. Whole-loop time includes the
start barrier, event delivery, lazy first-call allocations/calibration setup,
and all BD GPU drains, but excludes DataSource setup, explicit cache conditioning,
NIC sampling, acceptance readback and cleanup. There is no warm GPU timer,
per-event D2H, full-cube export, or cross-BD cube reduction in these rates.
Scaling is aggregate input/processing, not a complete distributed-cube product.

Before cold runs, only the ten owner-controlled synthetic bigdata files are
fsync'd, FADV_DONTNEED-evicted and checked with mincore. All must have zero pages
resident. SMD buffers may already be primed during DataSource setup. Warm cases
explicitly read the ten files outside timing and require >=99.9% residency.
Weka backend caches are uncontrolled. No global cache flush or data mutation.
Physical ethtool counters bracket the loop, with their own timing interval;
idle counter endpoints are also retained. Ports remain separate; NIC includes
all detectors and other host traffic. Do not infer exact cache-hit percentages.

Every event has 3,358,200 JF datagram bytes. The 10k JF numerator is
33,582,000,000 bytes (31.275675 GiB); the ten bigdata streams also contain
4,090,281,368 event bytes for CPU-routed ancillary detectors. Physical NIC
bytes and file sizes (including transitions/SMD) are distinct denominators.

A read/parse-only control exercises field descriptors and XTC parsing but skips
user expansion/calibration/cube kernels and their constants/banks. This changes
both compute and memory demand; it is a bottleneck diagnostic, not scientific
output or a pure single-factor CPU/GPU attribution.

The optional diagnostic wraps existing framework hooks with nested exclusive
main-thread wall and thread-CPU timers and CUDA events. It is separate from
clean throughput cases. CUDA intervals can overlap and include submission gaps;
their sum is not device busy time. Host exclusive categories leave uninstrumented
Python/framework work explicitly outside attribution.

1/2 BDs compare one and two streams; 4 BDs use one stream. Each stream's cube
bank is 6.25 GiB, so tested configurations have at most four banks (~25 GiB)
with an aggregate >=8 GiB non-bank reserve gate. Snapshots are not continuous
memory peaks. The 4 BD/2 stream configuration would need 50 GiB of banks alone
and is outside this A100-40GiB test. Framework budget remains 4 GiB per BD,
nominal EB batch 7, SMD grouping 20, bulk read target 512 KiB, forced KvikIO CPU
fallback, four workers/4 MiB tasks/16 MiB bounce/no auto-direct per BD.

Native evidence is under
`/sdf/data/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx/pipeline_scaling_20261109_r3`.
Job 43977007, exclusive sdfampere023. Pilot maximum is 400 events. The campaign
runs two pilots, a same-node standalone read control, 18 clean full-10k cases,
and three diagnostics. There are two cold/warm repetitions for full 1BD/2stream,
2BD/2stream and 4BD/1stream; other controls have one cold/warm pair. This is a
bounded scaling screen, not an exhaustive sweep or confidence-interval study.

Preserved failed attempts: job 43976569 (`..._r1`) had insufficient advertised
MPI slots (no read); job 43976777 (`..._r2`) lacked the process-local synthetic
reader registration before DataSource setup (no event loop). Corrected by
requesting six Slurm task slots and registering the existing adapter. No core
psana or detector kernel changes were made.
