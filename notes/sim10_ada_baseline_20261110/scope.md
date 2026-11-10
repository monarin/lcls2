# Corrected Sim10 Ada baseline

Run pinned PR #298 head 67c327a148a08b2aaf972e67e1e02e367c2b298f on S3DF Ada,
with source-identical existing native extensions and exact PR Python sources.
Only harness/evidence changes; no framework, data-format or calibration edits.
Reuse accepted 10k unique-event fixture, five JF files plus five SMD. Copy and
SHA-verify ~31.3 GiB to owner-controlled Weka FFB, never NVMe; remove only this
campaign's temporary FFB copy at completion. Shared scratch stores results.

Exclusive node (Slurm reserves all GPUs), one selected scheduler-assigned L40S, one BD plus SMD0 and EB. All ranks
share the same fixed 24 physical CPU allowance, NUMA interleave. Eight KvikIO
workers, fallback ON, task 1 MiB, bounce 16 MiB, auto-direct OFF. Batch100,
SMD1000, bulk512KiB, 6GiB framework budget. Descriptor staging GpuTask; no
expansion/calibration/cube. Two CUDA stream counts in order 1,2,2,1; precede
with GPU ownership/transition/task tests (no skips) and 1000-event pilots.

Require exact timestamps, 50k completed JF reads / 33,582,000,000 bytes per10k,
100 callbacks of100, requested stream use, no CPU payload reads or pressure
trims, zero pending/held resources. Cold gate after DataSource setup: fsync,
owner-file FADV_DONTNEED, mincore zero. Physical NIC counters and idle baseline
recorded separately per NIC; no summing interfaces or deriving cache fractions.
Whole event-loop rate includes lazy pipeline setup, excludes cache conditioning,
staging and tests. Hardware cache beyond local page cache is not flushed.

Default independent Claude review loop: max3 rounds. Review harness before job;
review measured evidence afterward. GPU performance is pending until job runs.
No cross-hardware speedup claim. No more BD/batch/KvikIO sweep in this stage.
