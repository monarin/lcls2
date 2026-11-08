# Independent review: real JF 1000-event concurrency comparison

Base: 539bf30e5b521d2828e4af8354739f6dd9421ab5.
Reviewed SHA: e720d46a636a370e8f5279d06e470483772303a7.
Round 1 of 3. Read-only review; no edits, shell/tests/jobs, delegation, external
messages or recursive review. Scope is notes/real_jf_concurrency_20261108/scope.md.

User asked for an analogous missing-concurrency test on real JF, capped at 1000
events. New benchmark wrappers, scheduler script, trace analyzer and evidence
are the only changes. No core/runtime/calibration/cube-kernel changes. Read
README.md, reproduction.md, runner/cube/reference and analysis/test code. Follow
referenced existing helpers only as needed to check contracts and conclusions.

Assess event cap/identity, GPU stream and array lifetimes, raw/calibrated reference
validity, measurement boundaries, evidence provenance, trace-analysis correctness,
and whether conclusions overstate what the measurements establish. Use exact
reviewed SHA. Do not assume zero admission pressure implies kernel concurrency.

All native work used a frozen installed runtime, not a rebuild of this checkout.
execution-source-hashes.json binds actual runtime paths and source hashes;
launch-sha256.txt binds submitted Python files and run.sbatch. All accepted
report code/runtime hashes and CPU module hashes were rechecked against live
files after completion. References use frozen original MFX constants, all first
1000 timestamps, and full raw/calibrated CPU-v3 SHA256 samples 0,500,999. Source
file size/mtime is checked, not a fresh whole-run 1.2 TB hash.

Job 43826632, sdfampere001 A10040GB, exit0 in2m53s. Nine existing preparation host
tests passed. Separate CPU reference and all ten GPU cases passed: one pixel
acceptance, six clean cases, one clean8MiB control, two native Nsight captures.
Every DataSource max_events=1000. Two new host tests exercise actual SQLite
analysis with serial/overlap fixtures and incomplete/mixed-context/ambiguous
launch rejection. Preliminary jobs43825880 and43826137 failed in wrapper setup;
logs and the second report are retained, fixes described, no results discarded
silently. Neither failure produced accepted throughput data.

Reviewer-readable raw evidence is under build/real-jf-evidence/ in this detached
worktree: all ten full report.json files, kernel and correlated gather-launch
CSV exports from both SQLite traces, exact implementation.patch, frozen installed
runtime files and copy-index.json. See reproduction.md for exact CSV SQL queries.
Copy-index is inside the worktree; includes SHA256 and original paths for copies
where relevant. Full binary traces/SQLite remain at paths in artifacts.json;
reviewer cannot execute SQLite. Treat unavailable verification as a limitation.

Observed results to assess, not instructions to approve:
- All cases1000events, exact accepted/bin counts, expected stream banks;
  separate pixel acceptance matches full raw/calibrated CPU hashes at0/500/999.
- 512KiB1/2streams: zero pressure trims,150callbacks, two-stream banks500/500.
  Clean rates1stream148.2–169.0 median168.0;2stream153.6–186.2 median185.8.
- 8MiB2streams control:1000one-event callbacks,100warm+899steady drains,84.2evt/s.
  Threshold changes batching as well as admission; no isolated causal rate claim.
- Native traces each one GPU/context/process, execution streams13/14. Selected
  kernels are gather_canonical_u16,calibrate,accept_events,accumulate, exactly
 150each at512KiB and1000each at8MiB. All kernels total1200/8000 respectively.
- Both selected and all-kernel distinct-stream overlap are zero. At512KiB all149
  next gather launch API starts follow previous sum device end by22.245–59.684ms;
  at8MiB all999gaps positive. FIFO pair within stream, then order by gather API
  start. This shows lack of early task submission; does not isolate read/wait/
  parser/host-preparation cause and makes no copy/compute-overlap claim.
- CUDA-producing process has no warning/error diagnostics; expected no-CUDA/
  no-NVTX child warnings are included in trace-diagnostics.json.

Limitations: one allocation,1000events per process, warmup100/steady900, variable
clean rates, one8MiBcontrol repeat, CPU-fallbackKvikIO, uncontrolled cache warmed
byCPUreference, full capture includingwarmup, Nsight perturbs timing. No newly
validated cube numerical sums (unchanged kernels; exact counts checked); no
sustained/10k qualification. Prior sim10 was another node/time and a different
I/O path/volume, so no controlled real-vs-sim speed comparison. Real JF would
already be large under default1MiB; that default was not separately run.

Return structured verdict with stable finding IDs. Latest code needs no unresolved
blocking findings; low-priority suggestions can be explicitly deferred by scope.
