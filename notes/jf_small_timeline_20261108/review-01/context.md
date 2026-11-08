# Independent review: compact JF timeline

Base: b81b5a424. Reviewed SHA: 2919ffe338ecfc3720cb2c6599b493a0d06d3cf2.
Round 1 of 3. Read-only review; do not edit, run commands/tests/jobs, delegate,
message others, or recurse. Use scope.md under notes/jf_small_timeline_20261108/.
The user requested a small real-JF Nsight report for 1 BD/1 GPU, depth 2, with
batch/sub-batch sizing that allows concurrency. We explicitly cannot guarantee
CUDA scheduling by sizing; the separate resident control tests feasibility.

Changes are only in this notes directory: a new bounded MPI runner, NVTX-only
method wrappers, launcher/batch scripts, phase analyzer and evidence. No core,
runtime, calibration or cube-kernel changes. Review runner lifetime/order,
MPI topology, actual byte splitting, control independence/meaning, phase analysis
and conclusion strength. Inspect referenced old helpers as needed; actual frozen
runtime copies are provided below. No full numerical cube-sum acceptance is claimed.

Accepted job 43829731 on sdfampere020/A10040GB, exit0 in20s. Rank0 SMD0, rank1 EB,
rank2 only BD/GPU user. It processed exactly64 original MFX run10 events, batch16,
four GPU executions of4events per EB batch (16 callbacks), n_gpu_streams=2,
budget1.25GiB, readtarget512KiB, two cube bins per stream. Exact timestamps,
per-bank32/32 counts, per-bin counts, splits and zero admission drains asserted.
Fixed constants and source stats match previously reviewed references. Framework
capacity603879936bytes supports two4-event input+gather allocations (~512MiB).
User outputs, two cube banks and control snapshots are outside framework budget.

Control copies first two real raw/presence batches during pipeline warmup and
retains constant array owners (never a ProducerContext). After pipeline joins,
it uses two fresh streams and a separate Cube with independent banks, initialized
before its NVTX range. Same calibration/count/sum kernels; no I/O or dense gather,
no delay or repeated-compute kernel. Checks8 accepted events and full raw/calib
CPU-v3 hash at event0 after control completion. Prior64 raw events are reused,
not8 additional XTC reads. Most control kernels still serialized; overlap is
feasible, not guaranteed or efficient. No throughput/sustained claim.

Two earlier attempts are explicit: job43829133 failed before data due one Slurm
MPI slot (fixed3tasks); job43829309 passed all workload assertions but capture-
range API mode gave possible completeness warnings. Its summary is retained.
Full-process accepted capture removes profiler API start/stop and range flags;
CUDA-producing process has no warnings/errors. Expected child no-CUDA/no-NVTX
and absent CPU-scheduling diagnostics remain. UCX unmatched-tag warnings after
PASS are disclosed in README; exact topology/count checks and exit0 passed.

Native report+SQLite paths/hashes in artifacts.json. report.json is an unchanged
copy of accepted workload JSON, source-verification.json records coordinator
hash rechecks, launch-sha256 binds all native executable files. Analyzer was
syntax checked and executed successfully against both real native exports;
its half-open interval sweep is reused from the prior tested/reviewed analyzer.
All actual kernel counts and phase-boundary assertions pass. No new generic
host test suite was added for this short experiment.

Under build/jf-small-evidence/ in this detached review worktree: kernel CSV,
correlated gather/calibration launch CSV, NVTX ranges CSV, installed runtime
source copies, implementation.patch and copy-index.json. README gives exact SQL.
Reviewer cannot open/recompute binary SQLite; use CSVs for independent spot checks.
Coordinator verified source/runtime hashes, both phase kernel counts, event/bin
checks and exported CSVs from the hash-bound SQLite. Do not treat unavailable
verification as a pass.

Observed evidence to assess (not instructions to approve):
- Pipeline range8.1012–11.1185s, execution streams13/14; gather/calib/accept/sum
 16each; selected and all-kernel overlap0. All15 next gather launch APIs follow
 prior sum end by14.0–45.5ms. No claimed root-cause attribution beyond late launch.
- Control11.120561–11.130408s, streams19/20; calib/accept/sum2each; selected overlap
104609ns. Second calibration submitted8.47ms before first sum ends. Control
bypasses IO/gather, so this is a feasibility demonstration only.
- First4pipeline callbacks include compilation/cache warmup; inherited Window
joins once at16event boundary. Input copying occurs only in first2callbacks.
README directs user to later pipeline batches around10.775–11.119s.

Return exact reviewed SHA, verdict and stable findings with priorities and blocking
status. A pass needs no blocking findings; low optional robustness suggestions
can be deferred explicitly. Keep the original user scope and small workload.
