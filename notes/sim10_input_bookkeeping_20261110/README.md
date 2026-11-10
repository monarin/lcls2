# Sim10 input bookkeeping optimization

The matched Perlmutter 10k test confirms that removing repeated input-pool scans
eliminates most of the extra two-stream host cost. This change preserves physical
reads, callback sizes, memory ownership and CUDA scheduling. Final review pending.

## Measured result

Job **61098604**, node **nid003900**, one BD / one A100. Each case reads
10,000 Sim10 events: **50,000 reads, 33,582,000,000 bytes (31.2757 GiB),
100 callbacks of 100 events**. Two clean repetitions per combination, in
forward/reverse order; medians below. Individual values are in
`native/attempt03/summary.json` and the full case receipts.

| GPU streams | Old seconds | New seconds | Old GiB/s | New GiB/s |
|---|---:|---:|---:|---:|
| 1 | 14.0240 | 13.6756 | 2.2302 | 2.2871 |
| 2 | 24.7121 | 14.3197 | 1.2656 | 2.1841 |

Two-stream throughput improves **72.6%** (elapsed time falls **42.1%**).
The new two-stream result remains about **4.5% below** the new one-stream rate.
This fixes the large bookkeeping penalty; it does not establish beneficial
stream scaling or maximum storage bandwidth. One-stream change is small (~2.6%)
and these two repetitions are not a statistical uncertainty estimate.

Separate timing cases sum startup and steady exclusive phases (seconds):

| Host phase | Old 1 stream | New 1 stream | Old 2 streams | New 2 streams |
|---|---:|---:|---:|---:|
| Input pool polling | 0.6377 | 0.1356 | 9.2247 | 0.1810 |
| Input issue bookkeeping | 1.2097 | 0.4564 | 2.7128 | 0.4731 |
| Reader group bookkeeping | 3.3504 | 3.2310 | 3.1204 | 3.3799 |
| Read wait | 1.0595 | 1.5047 | 0.8774 | 1.1074 |
| Entire measured loop | 16.1722 | 15.0597 | 26.3070 | 15.5796 |

New issue bookkeeping includes both `input.issue` and `input.issue_groups` self
time, excluding nested reader calls. Poll calls fall from **50,101 to 101** for
both stream counts. The new algorithm visits only eligible candidates; the
`retained_states_at_poll` counter sums pool sizes at entry and must **not** be
interpreted as candidate visits. Remaining reader pending-list scans and reader
bookkeeping are outside this change. Timing instrumentation adds overhead;
throughput claims use clean cases only. Individual phase differences are not
precise causal measurements of all remaining runtime costs.

## Implementation

`GpuEventManager._issue_group_reads` submits the admitted group plan once through
`InputGroupPool.issue_groups`. The pool validates all keys, slots and small-stream
credits before I/O and submits reads without per-group polling or full-pool slot
scans. A busy-slot set provides constant-time membership checks. The standalone
`issue()` API remains available.

Each bound `InputWindow` registers one notification. When closed with no remaining
planned/live references, it enqueues its state using `SimpleQueue`. Consumer
threads only enqueue; the BD thread updates the pool. Polling checks the existing
CUDA dependencies for candidates and releases the raw hold before returning a
slot or stream credit. Failed completion/release checks retain ownership for
retry. State-identity checks reject stale notifications after key reuse. Partial
batch submission remains pool-owned and drainable through existing cleanup.

## Controls and limits

- Same physical GPU, CPU affinity and NUMA interleave for all cases; three MPI
  ranks (SMD0, EB, BD), helpers masked off GPU. Framework batch 100, pool depth
  1 or 2, 512 KiB bulk target, 6 GiB budget, SMD batch 1,000.
- KvikIO fallback ON, 8 workers, 1 MiB tasks, 16 MiB bounce, auto-direct OFF.
- Input remains on Perlmutter Lustre at
  `/pscratch/sd/m/monarin/psana2-gpu/datasets/mfx101555026-r0010-sim10-100k-jf`.
  Owner-scoped fadvise and mincore establish zero local page residency after
  DataSource setup for every case. Backend/network cache state is unverified.
- Descriptor staging only: no blow-up, calibration, cube or user payload D2H.
- Three-file Python overlay on frozen install-B; unchanged modules/native files
  resolve to the frozen installation. Per-case path/hash checks enforce identity.
  The original installation and dataset are untouched. No native rebuild needed.
- All event identities, bytes, reads, callback sizes/counts, stream counts and
  cleanup checks pass. No CPU payload reads or pressure drains. All 11 dataset
  file metadata match before/after; content hashes reuse prior transfer checks.

## Validation and review

- Full CPU GPU-framework suite: **970 passed in 8.77 s**, Python 3.9,
  `PS_PARALLEL=mpi`, checkout Python over existing SDF native modules.
- Native GPU acceptance: **31 passed, zero failures or skips**. Covers actual
  device ownership, delayed consumers, transitions and task batches, including
  pixel comparisons. This is not a scientific calibration validation.
- Eight clean and four timing cases passed. Slurm job and all steps completed
  with exit 0. `summarize.py` reproduces the summary from full receipts.
- Round 1 reviewed 95e6c3788 and requested native evidence before approval;
  four nonblocking suggestions were implemented in c13d3df31. Final review pending.

Failed attempts remain disclosed. Attempt 01 (61098117) passed 30 GPU tests but
failed a stale assertion expecting BeginStep to refresh device constants. The
frozen runtime explicitly retains constants per run. An unchanged-runtime
diagnostic in successful attempt 03 reproduces that exact historical failure.
The corrected integration assertion retains consumer drain and pixel checks and
verifies stable constant identity/value plus step-generation changes. Calibration
runtime behavior is unchanged. A standalone diagnostic (61098227) failed before
pytest because activation inside srun scrubbed MPI library setup. Attempt 02
(61098414) failed before collection due to a pytest hook parameter typo, fixed
in b95b789ca. No performance conclusions come from those failed attempts.

## Records

- Branch: `work/sim10-input-bookkeeping-20261110`.
- Base: `1805e1d9522a4e45aa611a59d83a4cf051678104`.
- Native tested source: `b95b789ca` (full SHA in provenance); runtime/test changes
  last changed in `c13d3df31`. Later result documentation does not change runtime.
- Perlmutter artifacts:
  `/pscratch/sd/m/monarin/psana2-gpu/sim10-input-bookkeeping-20261110/attempt03`.
- Local records: `scope.md`, `status.json`, `cpu-tests.log`, `review-01/`,
  `review-02/`, `native/attempt03/complete.json`, `provenance.json`, `slurm.txt`.
- `native/attempt03/SHA256SUMS.remote` verifies copies of authoritative result
  JSON files. `summary.json` is derived locally; Slurm receipt is queried directly.
