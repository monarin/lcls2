Reproduce uncompressed JF raw-read performance with the preceding task's strict
local-cache definitions and physical NIC counters. Existing owned run387 five
JF prefixes on Weka FFB; no source edits or global cache flush. Exact 10k SMD
identities, 335571760000 datagram bytes, 50000 reads. Same frozen B runtime.
Two distinct workloads: standalone KvikIO (no CUDA kernels) and established
psana dense input adapter (XTC parser/gather kernels, no calibration/user task).
Historical baseline used local NVMe, so include a private node-local copy and
identify the storage path in every result. Hash-verify once on compute node;
never stage into shared persistent input. Remove only newly created local copy
on exit. Cold zero mincore pages; warm >=99.9% per file after setup; only five
owned files, excluding primed SMD. Weka backend caches uncontrolled.
One exclusive A100 node, one physical GPU UUID, <=60min, 700GiB RAM to permit
313GiB warm input plus staging/cache headroom. Physical NICs retained separately.
Psana: 1/4 BDs on Weka, 4BD local, batch20, one execution stream, bulk on,
1MiB target/task, eight KvikIO workers, 6GiB explicit budget/BD, no D2H in timed
loop. Common physical-core affinity across ranks matching historical placement;
record it rather than conflate with the preceding task's four-core/BD placement.
Two cold/warm repetitions for each psana topology; 200-event raw-pixel preflight
before throughput. Standalone: four workers/4MiB/depth35 on Weka to bridge recent
sample10; eight workers/1MiB/depth35 Weka and local to match historical tuning,
one cold/warm pair each. All fresh processes; cache conditioning excluded.
Validate short reads, global unique timestamp hash, bytes/counts, raw preflight
hashes, no CPU bigdata reads, budget/lifetime drains, single GPU and cache/NIC
endpoints. Standalone sampled GPU/CPU byte witnesses outside timing. No new
scientific/cube kernels. Report loop and NIC intervals separately; no exact
cache-hit fraction or causal across-node regression claim. Independent Claude
review up to three rounds, scoped commits/push to work/synthetic-mfx-scan.
