Review the scoped timing attribution implementation and conclusions at commit
27975c173a08bd69ba6b03c87c70cc31fcc8914f, base 2913e479c.
User asks: for one and two streams, break down which sections dominate timing.
This session requests independent Claude review for implementation/analysis.
Do not launch another review, execute commands, or edit files.

Read notes/mfx_expand10_sections_20261108/{scope.md,README.md,analyze.py,
test_analysis.py,tests.log,cpu-table.md,summary.json,source-verification.json,
run.sbatch}. Relevant unchanged instrumentation:
notes/mfx_expand10_performance_20261106/{timing.py,perf_cube.py,runner.py}.
The old clean baseline is notes/mfx_expand10_baseline10k_20261108/README.md.

Acceptance: both native runs pass 10k original event/bin checks, 1000 warmup,
9000 measured, no pressure drains, one/two expected banks. Same frozen sources,
no kernel/framework changes, no file staging. Diagnostic rates aren't baseline
replacement. CPU partition must handle nested NVTX spans without double counting.
GPU execution and worker/copy activity must be separate from host wall time.
Boundary exclusions and uninstrumented remainder must be explicit. Avoid claims
of Weka saturation, measured CPU utilization, or causal speedup from one profile.

Job43863315 completed0:0 in2m14s on sdfampere026 A10040GB. Five interval tests pass.
Coordinator checked all source/runtime SHA256s after execution and report code,
runtime and calibration equality across both cases. Native files bound in summary.
Initial GPU activity 6373MiB/100%, gone at both before_setup samples (~452MiB);
no continuous isolation claim. No sampling, profiling overhead quantified only as
limitation; warmed callback-window differs from NVTX envelope by <4ms.

Evidence readable inside build/sections-evidence: byte-identical report.json
copies (one/two streams), profile-aggregates.json projections (omit raw intervals),
and copy-index.json with native paths/hashes and projection rules. These copies
are present before review. Full native SQLite/Nsight reside at:
/sdf/data/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx/sections_20261108_r1/.
With read-only file tools you cannot recompute SQLite queries; say so as a review
limitation. Judge source/interval algorithm, reported evidence consistency and
claims, not an invented independent rerun. No GPU/software changes are proposed.
Review source code for hidden attribution errors as well as README conclusions.
Give stable finding IDs SEC-01 etc and the exact reviewed SHA.
