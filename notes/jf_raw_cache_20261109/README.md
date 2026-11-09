# Uncompressed Jungfrau cold/warm reproduction

Status: **queued, no new GPU measurements yet**. Slurm job 43983799 requests
one exclusive A100 node, one active GPU, 700 GiB RAM and a 60-minute limit.
The current queue has no start estimate. Independent harness-readiness review
is pending; this is not performance acceptance.

User request: reproduce the earlier uncompressed JF raw-read performance using
the preceding sample10 investigation's strict cold/warm definitions.
Two interpretations are covered: standalone KvikIO without kernels, and the
maintained psana input-staging benchmark without calibration/user kernels.
Psana staging does run the XTC parser and dense raw-gather kernels. No GpuTask,
calibration, cube computation or timed D2H is included.

## Historical reference and comparison limits

The [maintained staging report](../../psana/psana/gpu/docs/performance/read_staging.md)
used mfx101210926 run387, 10k events, full 32-panel JF, five files, batch20,
one execution stream, eight KvikIO workers, 1MiB tasks/targets and CPU fallback.
The September one-GPU/four-BD bulk-on medians were 351.64 events/s cold and
489.12 warm. October campaigns gave materially different absolute rates on
different nodes. Those inputs were staged on **node-local NVMe**, not Weka.
This reproduction includes both Weka and node-local storage on one new node;
it cannot assign historical differences to a code change across nodes/builds.

Input: existing owner-controlled Weka prefixes at
`/sdf/data/lcls/drpsrcf/ffb/users/monarin/jf-feespec-bulk-38995226/xtc`.
Only streams 005–009 participate. Exact event datagram numerator:
**335,571,760,000 bytes**, 50,000 requests for 10,000 events. Five staged files
including transitions/padding occupy 335,656,487,644 bytes. Physical NIC bytes,
whole-file bytes and datagram payload are separate quantities.
The copied reference.json, pixels.json and expected-sha256.json come from
`/sdf/scratch/users/m/monarin/gpu-validation/pr187-ab-1gpu4bd-20261019-r1`.
CPU preparation reproduced the 10k timestamp hash, five SMD hashes and exact
request/byte totals. Full bigdata SHA and raw-pixel checks run on the GPU node.

## Planned matrix and definitions

- Psana: Weka 1/4 BDs and local 4 BDs; cold/warm, two fresh-process repeats each.
  Batch20, one stream, bulk on, eight workers, 1MiB task/target, 6GiB explicit
  per-BD budget, 16MiB bounce, forced fallback, auto-direct off, SMD1000.
  All ranks share the allocation's physical-core affinity, as in the historical
  common-affinity harness. This differs from four dedicated cores/BD in the
  recent sample10 experiment and is recorded explicitly.
- Standalone: Weka four workers/4MiB and eight workers/1MiB; local eight
  workers/1MiB. One cold/warm pair each, depth35, 16MiB bounce, fallback,
  auto-direct off. Pin to four/eight physical cores respectively.
- Two 200-event psana preflights compare three gathered raw arrays with frozen
  CPU hashes. Timed cases check exact event partition/bytes/reads and drained
  resources. Standalone validates every completed read length and 105 sampled
  GPU/CPU byte witnesses after timing.
- Cold: fsync + file-specific FADV_DONTNEED, then **zero resident pages** in all
  five owned bigdata files after setup/warmup. Warm: preload outside timing,
  then **>=99.9% residency in each file**. SMD setup may already prime buffers;
  Weka backend caches are uncontrolled. No global cache flush.
- Read physical ethtool counters before/after each completed loop and separately
  during idle. Keep ports and counter intervals separate; no inferred exact
  cache-hit fraction. Local-NVMe reads should not generate payload-sized Weka
  NIC traffic, which helps distinguish the storage paths.
- Psana whole-loop time includes start barrier, lazy input initialization,
  delivery and all GPU drains. Setup, conditioning, NIC sampling, validation
  and cleanup are outside timing. Standalone times submissions through all
  futures and final device synchronization, excluding initial warmup/witnesses.
- Stage a private local copy, hash both source and copy against the historical
  manifest, and remove only that newly created directory in campaign cleanup.
  Evict the other inode set before switching storage paths to reduce cache
  competition. No persistent new dataset or calibration writes.

There are 20 planned cases: two preflights, twelve psana timed samples, six
standalone timed samples. Reverse topology/storage order in repeat two;
cold precedes warm for each pair. This is a bounded reproduction, not an
exhaustive sweep or a confidence-interval study.

## Records and execution

Code commit before submission: `1e526c602`. Frozen runtime is the same B install
used by the sample10 investigation; runtime.sh records the absolute prefix and
venv. Scripts/runtime hashes and device placement accompany each case.
Seven CPU cache/NIC audit tests and Python/shell syntax checks pass. GPU checks
and the independent result audit remain pending.

Expected artifacts:
`/sdf/data/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx/jf_raw_cache_20261109_r1`
and sibling `jf_raw_cache_20261109_r1_logs/43983799.{out,err}`.
No output directory exists until Slurm starts the job.

After completion, run with the Python3.11 environment from runtime.sh:

```bash
python notes/jf_raw_cache_20261109/analysis/audit.py \
  /sdf/data/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx/jf_raw_cache_20261109_r1 \
  --output /tmp/jf-raw-cache-audit-20261109.json
```

The auditor refuses incomplete campaigns, failed cache/NIC checks, false rates,
incorrect timestamp sets/pixel hashes/read counts, mixed GPU placement, source
hash changes and missing local cleanup. Preserve failure logs and use a fresh
artifact root for any rerun. Do not describe submission or host checks as a
successful reproduction. See [scope](scope.md), [status](status.json) and
[CPU tests](tests.log).
