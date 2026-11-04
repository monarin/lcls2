# psana2-gpu Perlmutter scaling test plan

Status: proposed execution plan; no new bandwidth measurement yet. Baseline code
is PR262 head **5b082e0fdfcefc36ea208c0954aab6548aa6bdb7**. Work branch:
**codex/psana2-gpu-scaleout**, personal fork **monarin/lcls2**. Keep this code SHA
fixed even if PR262 moves; record every later harness and optimization SHA.

## 1. Questions and success criteria

Measure the sustainable client-cache-cold Lustre read ceiling on a GPU node,
then find the least CPU/GPU capacity that lets psana2 approach it. Report three
workloads separately: host file reads, GPU read/parse/gather staging, and Jungfrau
calibration plus a compact reduction. Full-image D2H is a separate optional case.

Define B_storage as the best reproducible sequential host-read plateau for a
frozen file layout. B_pattern is the separately confirmed concurrency plateau
of exact host replay of the application's physical descriptor ranges, stream
interleaving, batch boundaries and completion boundary. Sweep replay reader
concurrency with the same plateau rule as B_storage; keep the logical trace fixed
while allowing independent reads to overlap. Coalesced replay is a labeled
diagnostic, never a substituted reference. Record fetched/useful amplification.
Freeze the exact replay variant and its effective request plan with every result;
a change in the application's bulk/read plan requires a newly matched replay.
If no replay plateau is reached, report a lower-bound reference and withhold the
near-saturation claim, even if the numerical ratio exceeds 0.90.
Use B_pattern for the pattern-matched target and report the ratio to B_storage
too. Neither is a hardware guarantee. Four 200-Gbit/s NICs imply 100 GB/s aggregate
link rate, but routing, OSTs, PCIe and CPU copies can lower file throughput.

Proposed decision rules, fixed before collecting results:

- Confirm a storage plateau only if two successive useful concurrency increases
  each improve paired throughput by <5%, with the upper 95% confidence bound below
  5%, while cache and storage-traffic checks pass. If resources run out first,
  report the best measured lower bound, not a proven plateau.
- Near saturation requires the lower 95% confidence bound of paired
  application/B_pattern throughput to be >=0.90. Replay and application use the
  same useful ranges, layout and completion boundary; count over-read separately.
- Retain a Pareto frontier of active GPUs, assigned physical CPU cores, CPU
  core-seconds per useful GB, and throughput. Among qualifying points, default
  to fewest active GPUs, then fewest assigned cores; retain alternatives when
  objectives trade off. Staging and calibration qualify separately.
- Weak scaling holds workload per worker node fixed: E(N)=B(N)/(N*B(1)). Target
  lower 95% confidence bound >=0.80 through eight worker nodes. Report storage
  efficiency and application/storage ratio at each N to distinguish filesystem
  contention from software scaling loss. This is a target, not a promise.

## 2. Source and data discovery

The baseline's maintained jf_scaling harness measures staging without
GpuTask/calibration. It hardcodes OpenMPI environment names, run387, one EB,
batch20 and a single-node GPU mapping. Adapt its correctness, request-count and
provenance checks; do not run it unchanged under Cray MPICH or call historical
SDF timings Perlmutter Lustre bandwidth. The older gpu_mpi_benchmark also needs
a current API/metric audit before reuse.

KvikioGpuReader.io_stats total_ns is wait-only; summed request-to-completion
intervals overlap. Neither is a node-throughput denominator. Its io_path label
alone does not prove that a particular Lustre read used GDS. Independently verify
installed KvikIO/cuFile settings and per-file path. Start with explicit CPU
fallback; evaluate verified GDS separately if available.

Pilot data: mfx101232725 run442, Jungfrau streams004-008, 10,000 events and
335,571,760,000 useful descriptor bytes. About 335.6 GB of staged files remain
under the recorded prior campaign, with SMD, pixel references and calibration
snapshot. The source run's nine XTC files total 545,970,697,927 bytes. Only metadata
and old manifests were inspected now; revalidate payload hashes, Configure,
segment order, transitions and calibration before use. Do not substitute run387
constants. Keep existing inputs read-only and use campaign-owned copies for
eviction/layout changes.

Sample staged/source s004 files each use one OST and 1 MiB stripes, on different
OSTs. Inventory all streams before generalizing. Quota snapshot: about 1.05 TiB
used, 20 TiB soft, 30 TiB hard. Budget against the soft limit with reserve.
No large weak-scaling corpus is prepared yet. See baseline.json for exact paths.

## 3. Freeze baseline before tuning

1. Build a detached checkout of the pinned PR SHA with checkout-local
   build_psana.sh and the psana-build environment. Give the working branch a
   separate build/install. Record compiler, MPI, Python, NumPy, CuPy, CUDA,
   driver, KvikIO/cuFile, loaded modules, effective settings and library hashes.
2. Verify actual Python imports and native library paths on the compute node.
   The working branch path is ~/lcls2_worktree/codex/psana2-gpu-scaleout, matching
   `source ~/activate_psana_gpu.sh codex/psana2-gpu-scaleout` after building.
3. Freeze the measurement harness, dataset/range manifests and calibration
   independently of baseline code. Measurement adaptation does not silently
   redefine the production baseline.
4. Run CPU invariant tests, applicable GPU pixel-exact/integration cases and
   real-run pixel/ownership preflights before timing. Validate timestamp coverage,
   segment order, missing fields, partial batches, BeginStep/EndRun drains,
   bounded ownership and final synchronization. Match CPU reference semantics;
   document any accepted numeric tolerance. Unexplained skips are not passes.
5. Preserve baseline results before optimization. Later A/B uses separate builds,
   identical data/layout/workload/settings and the same allocation.

## 4. Measurement contract

Use exclusive GPU compute nodes of a single recorded A100 memory class; do not
pool 40/80 GB results. NERSC documents 64 physical cores, four GPUs and 256 GB
host memory. Record actual NUMA/PCIe/NIC topology. Host-read controls use this
same node class with zero active GPUs. Reserving four GPUs for isolation does
not mean using one GPU saves allocation charges: report active and reserved
resources separately.

Start with a coordinated barrier. Headline rate is total useful bytes divided
by the maximum rank elapsed time covering submission through all reads, GPU
work and tail drains. Record barrier skew and setup/load, first-batch, loop,
final-drain and total job times separately. A secondary steady-state window
must count bytes completed within that same window. Do not sum overlapping
stage intervals or use average rank time. Warm compilation separately, then
reestablish cache state; lazy setup inside run.events remains timed and visible.

Use decimal GB/s for data and GiB for memory budgets. Record useful SMD bytes,
submitted physical bytes, completed bytes, requests, over-read/re-read
amplification and events. Input bytes derive from SMD, not the calibrated image
size. The pilot averages 33,557,176 input bytes/event. Raw contiguous tests count
completed requested bytes as useful. Validate bounds and short reads.

Primary runs consume distinct ranges in a single pass. Require <=1% residency
for every potential measured range on every reader node before timing,
including all shared-file ranges that dynamic application scheduling could send
to that node (not just initially planned node-private ranges),
verified with mincore or equivalent. Flush staged writes outside timing; use
file-scoped advisory eviction on campaign files only. Recheck after eviction;
reject or replace ranges when it fails. Never drop global caches. A dataset
larger than RAM or O_DIRECT alone does not prove an application cold path.
Client-cache-cold is not server-cache-cold; repeated extents after verified
eviction must retain that backend-cache caveat.

Capture client OSC/RPC-level read-byte and RPC-count deltas (or an independently
validated equivalent wire-level source), plus all relevant NIC RX/TX deltas,
without resetting shared counters. Record counter names, units, scope and
semantics and validate them with cold/warm controls before use. llite read_bytes
is application-level and may include page-cache hits: it cannot certify storage
traffic. Measure quiet-node background and distinguish MPI/SMD traffic.
Expected storage traffic is submitted bigdata bytes plus separately measured or
bounded page-rounding/readahead overhead plus SMD file bytes; report each term.
SMD is excluded from the useful detector-byte numerator. Freeze the overhead
allowance from preflight observations, never fit it afterward to make a gate pass.
Default cold evidence requires attributable wire-level read bytes >=95% of that
expected total; unexplained excess >20% or deficits require investigation. NIC
>=95% is corroboration only when other traffic could explain it. If trustworthy
Lustre counters are unavailable, retain native read evidence plus verified
residency, label storage attribution limited, and defer claims of verified
physical-bandwidth saturation. GDS needs suitable cuFile evidence as well.

Record CPU affinity/utilization/core-seconds and all worker threads; GPU UUID,
bus ID and utilization; host/pinned/GPU-owned/physical-GPU memory high-water;
queue/inflight bytes; PCIe traffic where available; stage times; SMD/EB queues
and service times. Use Nsight/native traces/Darshan in separate diagnostic runs.
Do not include instrumented rates in clean comparisons; keep ordinary telemetry
identical across controls and candidates.

## 5. Adaptive test sequence

| Phase | Experiment | Initial sweep and gate |
|---|---|---|
| A | Correctness and pilot | 200-event references and 10k-event baseline; one GPU/one BD; then two-node placement/coverage gate. Short pilot rates are not saturation results. |
| B | Host Lustre, zero GPUs | Reusable-buffer POSIX pread/preadv: 1,2,4,8,16,32 reader cores; 1/4/16 MiB reads. Add 64 cores only if needed. Direct-I/O is a separately labeled aligned diagnostic. |
| C | Pattern replay | Sweep 1/2/4/8/16/32 reader cores (64 only if needed) over exact SMD replay to confirm B_pattern with the same plateau rule. Coalescing is a diagnostic. Count fetched/useful bytes. |
| D | GPU ingestion | Read-to-device only, then read/parse/gather. One GPU first, 1/2/4 BDs per GPU and 1/2/4/8 KvikIO workers per BD. Bound total cores. Add two/four GPUs only if needed. |
| E | Application | Fixed Jungfrau calibration plus compact reduction; test winners/neighbors. Full-image D2H remains separate, counting every requested result. |
| F | Weak scaling | Fixed per-worker bytes/events/resources at 1/2/4/8 nodes. Interleave host replay and application; extend 16/32 after gates and capacity check. |

Before sustained Phases B-E, prepare the D_node-sized corpus defined below:
fully written raw files for B and valid real/replayed XTC with SMD for C-E.
The 335.6 GB pilot lasts only about 16.8 seconds at 20 GB/s; pilot-sized runs
under 30 seconds cannot confirm either plateau or the saturation gate. Large
valid XTC preparation is therefore a single-node acceptance dependency too.

Avoid a Cartesian sweep. First vary host reader count at 4 MiB, then transfer
size near the knee. Test stripe counts 1/4/8/16 on new private data with fixed
stripe size; test sizes 1/4/16 MiB near the winner. Inventory actual OST placement
and distribution across files. Preserve originals. Freeze layout for comparisons;
layout changes create a new storage reference, not a code speedup. For weak
scaling, freeze the explicit N-dependent layout policy below, rather than a
constant total OST count.

GPU starting settings: batch20, depth1, bulk on, 1 MiB task/target. Sweep workers
then BDs, keeping total cores explicit. Pin SMD0/EB, BD main and KvikIO threads
within recorded CPU sets; count all assigned/occupied cores, not just MPI ranks.
Reject oversubscribed thread configurations and disable unintended BLAS/OpenMP
threads. Near the best resource point try batches10/20/50/100, depths1/2 and
1/4/16 MiB KvikIO tasks independently of bulk target. Bulk off is a matched
control. Retain only memory-feasible points; budget each GPU across sharing BDs,
constants, raw/dense/calibrated buffers, held consumers, caches and reserve.
OOM/retry throughput is invalid.

## 6. Weak-scaling topology and dataset

Keep these experiments distinct:

- Storage/worker scaling uses disjoint physical file ranges per node with the
  same byte/request distribution and compute. Independent per-node psana jobs
  can diagnose worker capacity, but are not single-DataSource application scaling.
- Application scaling uses one MPI DataSource, globally unique events and
  disjoint descriptors across the global run. Start with one Smd0, one EB, no
  server ranks; increase PS_EB_NODES only if telemetry shows EB starvation.
  node.py assigns roles via communicator membership: rank>=2 and global modulo
  GPU mapping are invalid after topology changes. Derive local BD ordinal from
  role/shared-memory node communicators and map to allocated GPU UUIDs before
  CUDA initialization. Verify every rank's role/host/device at each N.

PS_EB_NODE_LOCAL=1 is a separate topology candidate supported by node.py.
It creates an EB per shared-memory node; configure/verify PS_EB_NODES consistently
with the observed EB membership and reserve its additional cores on every node.
Compare this variant to the single-EB topology with separate N=1 references and
full control-resource accounting. Do not assume a flag alone guarantees correct
rank placement, GPU assignment, data locality or improved throughput.

Initially co-locate control ranks on worker node0 on reserved cores. Keep
identical BD/GPU/core quotas and disjoint data per worker at every N, including
node0; reserve control cores even at N=1 and include them in total resource cost.
If a dedicated control node is required, start a separate series with its own
N=1 reference; report N workers and N+1 allocated nodes plus cost-adjusted
efficiency. Smd0 remains unchanged and calibration IPC stays node/device local.
Set max_events=N*E globally and record assigned/completed events and bytes per
node. If load balancing causes >5% work imbalance, retain the diagnostic but
qualify the fixed-work-per-node interpretation. Locality-aware scheduling changes
are separately reviewed optimizations, not an unreported harness adjustment.

The storage provisioning rule must scale *active* OST capacity with N. Let the
selected one-node five-stream layout use s0 stripes per stream with fixed stripe
size and P1 distinct OSTs. For the primary series, grow each stream's file size
by N and stripe count to N*s0; provision P(N)=N*P1 distinct OSTs across the five
streams. This increases capacity even while all nodes consume the same five
active stream files. A mere sequence of more chunks on the original five OSTs
does not satisfy this rule. Keep stream/request distributions and stripe size
fixed. Cap each file at 128 stripes; if available targets or quota cannot satisfy
the policy, reduce N or label a separate layout-constrained experiment.

Create only new campaign files, selecting nonoverlapping OST lists where needed.
Save lfs getstripe for every file, full OST/OSS mapping and distinct-OST counts.
Reject accidental overlaps that reduce the intended P(N); prepare a corrected
private layout before accepting the sample. Distinct OSTs may still share OSS
and fabric resources: record that mapping and use matched storage controls,
rather than claiming storage independence. Diagnostics must verify the measured
range and concurrent request window actually exercise the expanded layout; a
large allocated stripe set that is never active does not qualify. Alternative
chunk/file-per-worker layouts require proof of simultaneous active target growth
and valid psana multi-chunk consumption, not just more filenames.

At each N the host replay and single-DataSource application use exactly the same
files, extents and layout; raw synthetic tests alone cannot be their denominator.
All N-dependent layout changes follow the frozen rule and are reported alongside
rates. If the replay-data builder or multi-chunk reader cannot satisfy it,
application weak scaling remains blocked; independent-reader storage results
retain their narrower scope. Do not disguise this as a software scaling failure.

Prepare a hashed range manifest and size the unique corpus per worker as:

    D_node >= max(2 * host_RAM, 1.2 * B_est * T)
    D_total = N_workers * D_node + metadata/warmup/reference space

Target T=60 seconds, with 30 seconds minimum for an accepted sustained sample.
Expand data if the actual run finishes sooner. B_est comes from the pilot, not
link specifications. Hold D_node, event count and work fixed across N in a
series. Size for both host and application controls; enlarge and restart a
series if necessary. At 20/40/80 GB/s, 60-second sizing requires respectively
1.44/2.88/5.76 TB per node, or 11.52/23.04/46.08 TB at eight nodes. These are
examples, not forecasts. The current 20 TiB soft quota cannot fit every case.
Maintain >=20% soft-quota headroom including existing usage, preparation copies
and retained layouts. Stage one large layout at a time. Reduce N or obtain
project storage before exceeding that budget; never reduce timing below the
minimum to make a larger scaling chart. Refresh quota and file presence first.

Raw data must be fully written, nonsparse, non-reflinked independent extents;
verify allocated blocks and deterministic sampled checksums. Hardlinks/symlinks
and repeatedly reopening the same pilot are not additional data. Generation,
writes and checksum passes are separate from timed reads. Five-stream-shaped
raw files can validate storage, not psana correctness.

For application weak scaling prefer additional compatible real XTC chunks/runs
and their calibration. Otherwise implement a separately tested replay-data
builder: retain Configure/Names and physical segment order, rewrite unique
monotonically ordered event timestamps, rebuild SMD file/offset/size indexes
for new files, and preserve transition/step semantics. Label this replicated
payload data. Never concatenate XTC bytes or blindly copy SMD indexes. CPU-read
shard boundaries, partial events, every transition and deterministic interior
samples; compare payload/reference and validate global event/extent coverage
plus GPU checksums. Without a valid large XTC corpus, only storage can scale;
application weak scaling remains blocked rather than inferred.

## 7. Statistics, execution limits and outputs

Screen each configuration twice; screens select candidates only. Confirm the
winner, smaller-resource neighbor and baseline with at least five balanced
paired repetitions on the same allocation. Randomize/reverse order and
interleave a fixed sentinel for storage-load drift. Reserve a separate
confirmation set to avoid selecting a ceiling from favorable screening noise.
Report every sample, median, mean, SD and 95% confidence intervals. Fix the gate
method now: for each independent balanced pair/block form the log throughput
ratio r_i, then exponentiate mean(r) +/- t_(0.975,n-1)*sd(r)/sqrt(n). Use at least
five independent balanced blocks; small-sample intervals are provisional and
assume approximately independent, normal log ratios. Publish log-ratio plots,
sentinel drift and leave-one-block-out sensitivity; if assumptions are doubtful
or a threshold conclusion changes in that sensitivity check, call it inconclusive
and increase independent blocks rather than substituting a narrower bootstrap.
Do not treat repeated events within one run as replicates. Apply the same method
to plateau increments, application/replay ratios and weak-scaling ratios.

For weak scaling, each block pairs the N-node rate with a one-worker reference
on the same allocation, using a predeclared rotating worker-node selection; form
log(B_N/(N*B_1)). State elapsed separation and node identities, balance N order
across blocks and retain all valid nodes. Storage and application each get their
own paired reference. Do not cherry-pick the fastest N=1 node. Keep slow valid samples. Failed
cache/correctness/provenance checks invalidate samples with recorded reasons;
speed alone never does. An interval too wide for a gate is inconclusive and
requires an explicit follow-up, not unlimited automatic benchmarking.

Initial scheduling caps: correctness/pilot <=1 node-hour; host and GPU screens
each <=2 node-hours; confirmation <=3 node-hours. Weak scaling gets a separate
resource estimate after the pilot. These caps are proposed scheduling bounds,
not runtime forecasts; checkpoint incomplete stages. Data preparation has its
own bytes/write-rate estimate and quota gate. Use account lcls_g, premium QoS
per standing preference, explicit cores/GPUs/memory/walltime, verified Cray MPI
and affinity. Check Iris personal/project balances before each allocation and
alert below 20%. Record actual charged hours and reserved versus active capacity.
No staging, rebuild, optimization or benchmark jobs are launched by this plan.

Each results.jsonl sample links code/harness/build hashes; data/range/layout
hashes; job/node/time; rank roles and CPU/GPU maps; workload/effective settings;
cache/counter evidence; bytes/events/timing; memory/queue highs; per-node rates;
correctness and invalidation reasons. Keep bulk logs on scratch and compact
reports in Git; preserve failed attempts and raw outputs.

Deliver: frozen PR262 baseline table; sequential and pattern ceilings;
resource/throughput Pareto tables for staging and calibration; weak-scaling
aggregate/per-node rates, efficiency, stragglers and allocation cost; reproducible
data/launch commands; next bottleneck supported by diagnostics. Each optimization
gets implementation, verification, committed-SHA Claude review, response/fix and
re-review. Unresolved correctness or provenance failures prevent acceptance.

## References checked

- [PR262](https://github.com/slac-lcls/lcls2/pull/262), API head verified and fetched.
- Baseline source: psana/psana/gpu/scripts/jf_scaling/{README.md,bench.py},
  gpu_kvikio_read.py, psexp/node.py and issue159_stage7_performance_20261029.md.
- [NERSC architecture](https://docs.nersc.gov/systems/perlmutter/architecture/).
- [NERSC Lustre](https://docs.nersc.gov/performance/io/lustre/): proposed striping
  sweep <=16 OSTs, within the documented maximum guidance of 128.
- [NERSC scratch](https://docs.nersc.gov/filesystems/perlmutter-scratch/).
