# Independent review: repeated-calibration pipeline concurrency

Base 18198d703; reviewed SHA is supplied by the coordinator invocation. Round1/3.
Read-only review, no commands, edits, delegation or nested loops. Inspect only
notes/jf_compute_overlap_20261108 and its inherited helpers as needed. The scope
is scope.md: same bounded real 64-event MPI 1BD/1GPU experiment, twelve calibration
passes instead of one, a native Nsight report demonstrating concurrent CPU
submission, and no framework fixes or representative performance claims.

Changes derive from notes/jf_small_timeline_20261108 (previous reviewed capture).
runner.py adds RepeatedCalibration: base.calibrate creates/retains output once,
then the same cached kernel is replayed eleven times with the same raw input,
constants, presence and output. Verify the argument ABI and launch shape against
psana/psana/gpu/examples/jungfrau_calibration.py (used via the inherited adapter),
including unused raw pointers for status/extra/valid and disabled masking.
All launches on supplied stream, no new synchronizations; count/sum once. First
final calibrated image copied on stream and retained for validation after join.
Inherited first-two raw copies and resident control retained; control uses the
base single-pass calibration, never the repeated wrapper. Constants and outputs
have batch-owned lifetimes. Wrapper uses a private example ABI deliberately and
is documented as diagnostic only.

Job43837236 on sdfampere040/A10040GB, COMPLETED0:0 in23sec, first attempt. Actual
three MPI ranks SMD0/EB/BD, only BD GPU visible, frozen runtime same as baseline,
CPU fallback,16-event batches, four4-event subbatches each, depth2,budget1.25GiB,
target512KiB. Exact64timestamps,16callbacks,per-bin counts,32/32bank events,zero
admission drains, full first repeated calibrated image CPU-v3 hash all pass.
The single-pass resident control again passes full raw/calibrated sample hashes.
No new full-pixel numerical cube-sum check; no throughput claim. Different node
from baseline explicitly prevents a controlled speed comparison.

analyze.py changes only expected192 pipeline calibrations (16 each gather/count/
sum), records CPU launch completion and kernel bounds, and asserts at least one
post-warmup adjacent cross-stream gather CPU API completes before previous sum
finishes. Correlation IDs uniquely match CPU launches. Native analyzer passes.
Seven of eleven steady adjacent handoffs qualify. Inspect the exported CSVs to
independently check several pairs, especially 11.16–11.27s: stream14 sum ends
11224653271ns; stream13 gather CPU API returns11202744607ns, first calibration
returns11202965574ns, all12 calibration APIs end11203119262ns, sum API ends
11203302820ns. This provides actual task submission ahead of previous completion.
Selected device overlap292481ns; all-kernel1205029ns. These aggregate over the
whole pipeline phase, not a single interval. Single-pass resident control is a
separate phase and is not the concurrent pipeline submission evidence.

Readable independent exports in build/jf-compute-evidence/{kernels,launches,nvtx}.csv
from hash-bound SQLite. Current source/runtime SHA hashes rechecked, plus23
inherited source files from prior manifest; see source-verification.json.
Coordinator ran native checks and AST/shell syntax checks; reviewer has no shell
and should not claim to execute them. No new generic analyzer fixture tests added;
interval sweep inherits the prior reviewed/tested analyzer. Native SQLite/nsys
are binary; reviewer should use supplied CSVs and declare limitations.

No producer-process capture warning/error; global CPU scheduling disabled and
child no-CUDA/no-NVTX diagnostics retained. UCX unmatched-tag warnings occurred
afterPASS, exit0; cause uninvestigated. Job log formatting normalized, native
log hashes preserved. All artifact paths/sizes/SHA are in artifacts.json. Scope
requires no rerun merely for review; request concrete fixes if evidence fails.
Return exact SHA, verdict, stable finding IDs and blocking flags. Do not approve
missing evidence or infer successful independent checks that were not performed.
