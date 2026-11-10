# Corrected Sim10 baseline on S3DF Ada

Purpose: establish the post-bookkeeping-fix single-GPU baseline on Weka before
further BD/batch/KvikIO tuning. This is descriptor staging through psana, including
XTC metadata handling; it does not run expansion, calibration, cube or payload D2H.

Runtime: [PR #298](https://github.com/slac-lcls/lcls2/pull/298), pinned to
`67c327a148a08b2aaf972e67e1e02e367c2b298f`. PR Python runs with the existing SDF
install-B native extensions. Every tracked native source file in psana/xtcdata/psalg
matches that build's source; module paths/hashes are checked during each case.
CPU unit suite: 1,023 passed. GPU acceptance is a required first phase of the job.

Configuration: one selected L40S, one BD, batch100, SMD1000, bulk512KiB,
6GiB budget, KvikIO fallback ON / 8workers / 1MiBtask / 16MiBbounce / auto-direct OFF.
The exclusive Slurm node reserves all GPUs; only one selected GPU is used.
Three MPI ranks share a fixed allowance of 24 physical cores, NUMA interleave.

Input: accepted 10k-event sample10 fixture, five JF files and five SMD files.
A temporary verified copy goes to owner-controlled Weka FFB. Each full pass
requests 33,582,000,000 JF datagram bytes (31.2756747007 GiB), 50,000 reads,
100 callbacks of 100 events. Two 1k-event pilots precede full passes in CUDA
stream order 1,2,2,1. Cold conditioning occurs after DataSource setup and requires
zero mincore-resident pages. Per-interface physical NIC counters and idle rates
remain separate from the application-loop timer. Backend storage caches are not
flushed. There is no NVMe input staging.

Artifacts:
`/sdf/scratch/users/m/monarin/gpu-validation/sim10-ada-baseline-20261110/`

The Slurm script runs GPU acceptance, staging/hash verification, all six cases,
temporary-input cleanup, then a separate audit. A nonzero job exit is not a
measurement pass. Inspect `results/gpu-tests.json`, `results/complete.json`,
`results/cleanup.json`, and `audit.json` before using rates. Read per-case logs
when any phase fails. Failed attempts remain separate from accepted evidence.

Review and scheduling state: see status.json. Ada results must be reported
separately from historical Ampere and Perlmutter A100 results.

JF bigdata files alone are cold-gated. SMD files are not evicted and are expected
to be page-cache warm; backend storage caches are uncontrolled.

The 20-minute allocation is deliberately bounded: allow up to 5 minutes for GPU
tests, about 3 for staging/hash checks, and roughly 6 for pilots/repetitions and
cleanup, leaving about 4 minutes before the TERM warning. This is an estimate,
not a measured Ada runtime; timeout leaves an incomplete attempt and no accepted
rates. Do not interpret a scheduling timeout as a framework regression.

Shared-scratch tmp, CuPy compilation cache and pytest temp directories are retained
for diagnosis until the attempt is reviewed. If a hard scheduler kill interrupts
staging, check the owner-only FFB sim10-ada-baseline-<jobid> directory for an orphan
and remove only that attempt's directory after confirming no readers remain.

The runtime loader intentionally omits the optional psexp.parallel_pread extension
(debugtools-only in this checkout), and asserts that absence. Native source
comparison includes psana, xtcdata and psalg; build configuration hashes are
recorded separately. This reuses an existing build, not a fresh PR rebuild.
