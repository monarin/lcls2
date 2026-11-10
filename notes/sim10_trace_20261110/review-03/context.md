Review this completed bounded bottleneck investigation, not a framework optimization.
Read-only review, no tools beyond the restricted helper, no delegation or nested review.

Base accepted workload: cc98b8440e00123bbe86a1d82e6be7986f28a447.
Runtime harness implementation: ec0941a31222045e31e5a8ea8fb49c02c4a63a28.
Current results/analysis commit: 3b1e9b3ac2d4e0d45c660be8097b1c9c3111c76b.
Round 3 of 3. Round 1 timed out; round 2 passed implementation with no blockers.
No runtime harness changes since that pass. New code is offline analyze_trace.py only.

Review README.md claims against native evidence in notes/sim10_trace_20261110,
especially analysis-summary.json, complete.json, slurm.txt, timeline-analysis.json,
the two timing phases JSON files (self_ns vs total_ns), and analyze_trace.py.
Do not load all large case timestamp arrays by default. The six case JSON receipts
are available if a focused field check is necessary. Frozen source copies of
InputGroupPool, InputWindow, GpuEventManager are identical to repo source;
use these to assess the proposed mechanism and distinguish inference from measurement.
Inspect review-02/review.json and responses.md for prior nonblocking T1-T4.
Review against scope.md and all requested acceptance criteria.

Coordinator validation evidence:
- Job 61072191 completed on nid001129; all six Slurm steps COMPLETED 0:0; 5:43.
- Clean, timing, nsys pairs: all exact 10k events, 50k reads, 33582000000 bytes,
  100 callbacks size100, requested stream count, zero initial local page residency,
  zero CPU payload reads and pressure drains, cleanup/retirement pass.
- Same host/GPU/CPU mask/settings, unchanged input metadata before/after campaign.
- Pure CPU timer tests previously 10 passed; native installation now succeeds.
- Independently recomputed six case invariant fields and phase exclusive sums;
  host exclusive totals match elapsed within 0.001 s for timing-only cases.
- Nsight files (34MB each) exported to SQLite read-only and analyzed by supplied script;
  100 walk,100 init_locators,100 locate_fields each. Two-stream split50/50 on13/14.
- Copied reports locally and SHA256 matched Perlmutter. SHA256SUMS covers native/frozen.
- git hooks AST/conflicts/format pass. Analysis script ran successfully on both exports.

Artifacts at /pscratch/sd/m/monarin/psana2-gpu/sim10-trace-20261110/attempt01/results
are not accessible from your restricted SDF worktree. Raw nsys-rep and SQLite are
not in Git. Treat supplied timeline-analysis.json as coordinator-derived evidence;
review SQL/math and acknowledge inability to independently query the binaries.
The exact compact native result receipts and instrumented source hashes are in Git.

Important limits to check:
- 99.6% attribution is for the host-timed pair, not exact clean-run attribution.
- input.issue self time includes slot linear scan, but the scan was not separately timed.
- Explicit ~2ms synchronization is BD host phases; KvikIO worker sync is separate.
- Nsight outer elapsed includes profiler start/stop and final barrier, while bd.loop
  is16.91/28.03s; do not claim exact start-vs-stop breakdown or profiled throughput.
- NVTX bd.loop label startup spans entire loop; whole-loop totals are used.
- GPU activity union is traced kernels/copies, not hardware utilization/network time.
- Local-page-cache-cold only; backend cache/network bytes unverified; noNVMe staging.
- One matched pair per mode, with previous independent repetition as context.
- No framework fix or measured optimized speedup yet. Recommendations must preserve
  input ownership/small-stream credit/deferred retirement/error cleanup.

Return structured verdict/findings on current commit, stable IDs, no forced issues.
