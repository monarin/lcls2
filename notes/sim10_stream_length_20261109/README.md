# Perlmutter Sim10 stream and duration comparison

Native validation is in progress; no benchmark acceptance is claimed yet.
The implementation is on `work/perlmutter-sim10-stream-length`, currently
`97399922b`. Review of the initial port `cfadd496a` passed implementation readiness
only; native fixes and results require re-review.

Use the transferred 100k JF-only fixture on Lustre in place. The same five bigdata
files serve both the 10k prefix and full 100k case. One BD uses one allocated GPU
UUID; helpers are CUDA-masked empty. Perlmutter's exclusive-node allocation grants
four GPUs, while each MPI step requests one and verifies the selected UUID.

Fixed controls: batch100, SMD1000, bulk524288B, budget6GiB, KvikIO compatON,
8 workers, 1MiB tasks,16MiB bounce, auto direct I/O OFF. Descriptor staging only:
framework reads/walking/metadata preparation, no dense gather, expansion,
calibration, cube or payload D2H. Each event contains 3358200 datagram bytes.

The mode is local-page-cache-cold on the five bigdata files, verified by
fsync/FADV_DONTNEED and zero mincore residency after DataSource setup. SMD and
Lustre server caches remain uncontrolled. Raw llite/CXI diagnostics are retained;
remote traffic is unverified. No network-byte ratio or cache-hit fraction is claimed.
No Weka reproduction or direct explanation of the earlier SDF discrepancy follows.

Two 1000-event pilots precede the streams1/2 × events10k/100k matrix, then its
reverse. Startup, allocations, fill, drain and barriers remain timed, affecting
short cases proportionally more. Validation reads the whole fixture before the
pilots; reversed 10k cases follow all full-file reads. Reversal does not equalize
backend caches. Report individual repeats and paired ratios alongside medians.

Attempt records under `/pscratch/sd/m/monarin/psana2-gpu/sim10-stream-length-20261109`:

- `attempt01`, job61064869: all eleven SHA256 checks passed; validation stopped
  because the checker mixed sim10 and config Names. Corrected in f0bc14cef.
- `attempt02`, job61064979: failed before input reads on the initial single-granted-
  GPU assumption. Exclusive-node Slurm grants four; corrected in 97399922b.
- `attempt03`, job61065065: independent hashes, SMD/Configure checks and full
  DataSource traversal, then pilots/matrix and native audit tests in progress.

[Scope](scope.md), [review response](perlmutter-review-01/responses.md),
[status](status.json), and [preflight evidence](preflight-check.txt) retain the
implementation and interpretation contract. The original SDF handoff and transfer
receipts remain historical evidence; no SDF allocation was resumed.
