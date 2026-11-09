# 100k sim10: 1–4 BDs on one GPU, cold Weka FFB

Preparation in progress. See [scope](scope.md) and [status](status.json).
Two CUDA streams per BD,512KiB bulk target,batch100,8 KvikIO workers per BD,
1MiB tasks,16MiB bounce buffers,compatibilityON,automatic directIOoff. Same
psana-GPU descriptor staging baseline, with no user kernels or payload D2H.

Use100k events from the accepted scratch fixture, copied and SHA-verified on FFB.
Two cold repetitions per topology, order1,2,3,4,4,3,2,1, after small preflights.
Full-loop GiB/s numerator is335820000000bytes(312.756747GiB), not the sum of
NIC traffic or bytes expanded by the hypothetical user kernel.

Node payload-cache cold is established by per-file fsync/FADV_DONTNEED and zero
mincore residency after setup. Physical NIC counters bracket the loop; SMD and
server caches remain uncontrolled. No NVMe input staging. Increasing BDs also
increases KvikIO threads(8–32) and CUDA contexts, with the same shared physical
CPU allowance and one GPU. Two streams exercised does not prove overlap; no
Nsight concurrency measurement is part of this sweep. Memory snapshots and
submitted-not-retired request peaks are not simultaneous storage concurrency.
