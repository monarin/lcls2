# Matched 100k-event Ada baseline

Run the accepted sample10_100k fixture on sdfada016, one L40S, one BD,
24 physical CPU cores, batch_size=100, SMD chunk=1000, bulk target=512 KiB,
6 GiB input budget. KvikIO fallback ON, eight workers, 1 MiB tasks, 16 MiB
bounce buffer, automatic direct I/O OFF. Use the exact PR #298 Python commit
67c327a148a08b2aaf972e67e1e02e367c2b298f and verified source-compatible native
extensions used in the completed 10k baseline.

Run 1000-event pilots for one and two streams, then four 100k passes in stream
order 1,2,2,1. Stage five JF and five SMD files (~313 GiB) on owner-controlled
Weka FFB, verify SHA256 and size, remove only this job's copy after completion.
No NVMe, dataset generation, calibration writes, additional BDs or cube kernels.

Each full pass must preserve 100k identities, 500k physical reader requests,
335820000000 bytes, 1000 callbacks of 100 events, clean drain, correct runtime
identity, mincore-zero JF after initialization and conditioning, physical NIC
before/after plus idle interval. Backend caches are uncontrolled and SMD warm.
Report repeats and medians without inferring overlapping kernels or a ceiling.

Close prior review ADA-08 and ADA-09: fail-closed exact build-config mismatch
allow-list with persisted mismatch receipt; checked-in submission helper;
canonical source paths, tested against symlink paths. Preserve 10k evidence.

Session-managed Claude review, maximum three rounds; review the committed
implementation before submission, and audit results after execution. A queued
job may be handed back with precise paths and resume instructions.
