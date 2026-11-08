# Small real-JF timeline: one BD, one GPU, depth two

Open the native report with Nsight Systems (captured by version 2025.3.1):

`/sdf/data/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx/jf_small_timeline_20261108_r3/jf-64-b16-s4-pool2.nsys-rep`

This is an actual three-rank MPI launch: rank 0 SMD0, rank 1 EB, rank 2 the only
BD. Only the BD can see/use the one A100. Nsight profiles only that rank. The
source is the first **64 original MFX run-10 events**, retaining all ten XTC
files and the five real JF streams. No input is rewritten or staged.

| Setting | Value |
|---|---|
| EB communication batch / PS_SMD_N_EVENTS | 16 / 16 |
| Execution pool depth | 2 (`n_gpu_streams=2`) |
| Actual GPU sub-batch | 4 events |
| Batches / GPU executions | 4 / 16, each batch split 4+4+4+4 |
| Framework memory budget | 1.25 GiB |
| Read-group target | 512 KiB (all JF datagrams large) |
| JF raw per event | 32 × 512 × 1024 uint16 = 32 MiB |
| Cube bins | 2 per stream; 256 MiB total cube state |
| I/O mode | KvikIO CPU fallback |

There is no new public sub-batch-size option. The existing byte planner produces
four-event executions under this budget, and the runner asserts every actual
split. After constants/setup and the 10% margin, measured admission capacity is
603,879,936 bytes, divided across two executions. A four-event input plus dense
raw gather costs approximately 256 MiB per slot (plus parser/metadata), fitting
two slots; five events per slot would exceed the allowance. User calibration
outputs, cube banks and control snapshots are outside this framework budget.
The A100 has ample room for this small task state. Admission capacity, allocated
bytes, observed ranges and pool depth are recorded in report.json.

Multiple independent executions are available within each EB batch, so lack of
work and a single-slot configuration cannot explain serialization. Batch sizing
alone cannot guarantee CUDA scheduling or meaningful overlap of kernels that
already consume most device resources. The separate resident control measures
whether overlap is possible for the same work on this hardware.

## Viewing the timeline

Expand the BD process's NVTX track and GPU CUDA kernel tracks. Select/zoom these
named ranges rather than the whole process startup:

- `PIPELINE_64_EVENTS`: the real file-reading pipeline. Nested `read_issue`,
  `read_wait`, `submit_gather_task`, `admission_split` and `task events=4 ...`
  ranges identify host work. Kernel names show `gather_canonical_u16`,
  `calibrate`, `accept_events`, and `accumulate`. Calibration and cube host calls
  share the enclosing task NVTX range; their host submission durations are not
  separately instrumented. Use the named GPU kernels and correlated CUDA launch
  APIs to distinguish them. Nested host spans are a possible future refinement.
- `RESIDENT_CONTROL_8_EVENTS`: two four-event batches copied from the first eight
  real events, already resident and ready before launch. It uses the same
  calibration and cube kernels, with two independent streams and cube banks;
  I/O and dense gathering are bypassed. No synthetic delay/repeated-compute
  kernel is added. This is a feasibility control, not an alternate pipeline.

The first four pipeline executions warm up compilation/caches. The inherited
benchmark joins once at the callback boundary after 16 events. It also saves the
two control inputs during the first two callbacks, labeled
`save_control_inputs_warmup_only`. Inspect the later pipeline batches for the
repeated gap, not that initial setup or deliberate join. Pool depth refers to
the two execution streams; KvikIO/parser helper streams and the control's two
fresh streams may also appear. Both top-level NVTX ranges end after explicit
stream joins. Pixel D2H validation happens after the control range.

## Validation and provenance

Every event timestamp must match the previously verified original-source
reference. All 64 pipeline events must appear exactly once in the proper bins,
with 32 events in each stream bank, zero admission drains and exact 4+4+4+4
splits. The resident control checks eight accepted events and full raw/calibrated
CPU-v3 SHA256 for event zero using unchanged frozen MFX constants. It does not
newly validate full numerical cube sums. Existing calibration/cube kernels and
core/runtime files are unchanged. No throughput number is claimed.

Job 43829133 failed before data: one advertised MPI slot for three ranks. The
batch allocation was corrected to three tasks. Job 43829309 passed all workload
checks, but profiler-controlled capture emitted possible completeness warnings;
its windowed summary is retained, and a full-process capture supersedes it.
The Nsight process executes within MPI; helper ranks are outside the capture.
Review records and final accepted measurements are described below.

Reproduce with `run.sbatch SOURCE_DIR NEW_OUTPUT_DIR`, setting scheduler log paths
to an existing sibling directory. The script invokes the frozen MPI runtime,
profiles only BD rank 2, and exports SQLite. To reanalyze on a CPU host:

```bash
python3 notes/jf_small_timeline_20261108/analyze.py \
  /sdf/data/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx/jf_small_timeline_20261108_r3/jf-64-b16-s4-pool2.sqlite \
  /sdf/data/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx/jf_small_timeline_20261108_r3/report.json \
  --output /tmp/jf-small-new-summary.json
```

The analyzer requires one CUDA process/device/context, both streams in each
phase, exact expected kernel counts and unique matching launch APIs. Its
half-open interval sweep (reused from the prior reviewed analyzer) counts time
with at least two distinct active streams. Phase boundaries come from the two
fully joined NVTX ranges. Submission gaps pair gather-to-sum within each pipeline
stream, or calibration-to-sum in the resident control, then order by launch API
start. No copy/compute overlap or critical-path attribution is inferred.

## Accepted capture

Job **43829731**, sdfampere020, A100-SXM4-40GB, completed exit 0 in 20 seconds.
All workload assertions passed. The native report is **1,182,379 bytes**.
The full capture has no warning/error diagnostics for the CUDA-producing
process. Child-process no-CUDA/no-NVTX warnings and the expected lack of CPU
scheduling information (CPU sampling/context switching disabled) are preserved
in timeline-summary.json. Both MPI captures printed UCX unmatched-tag warnings
after the PASS line; these are retained in the job logs. They did not prevent
completion or the exact event/count checks; their cause was not investigated.
Committed job logs are normalized copies: Nsight progress lines, carriage-return
formatting, NULs and trailing whitespace were removed. The byte-exact originals
are the sibling `_logs` files bound by artifacts.json.

| NVTX range | Time in report | GPU execution streams | Selected-kernel overlap |
|---|---|---|---|
| PIPELINE_64_EVENTS | 8.1012–11.1185 s | 13, 14 | 0 ns |
| RESIDENT_CONTROL_8_EVENTS | 11.120561–11.130408 s | 19, 20 | 104,609 ns (~105 µs) |

For the regular pipeline, start looking around **10.775–11.119 s**, after the
first 16 events. All 15 next gather launch APIs begin after the preceding sum
kernel finishes, by **14.0–45.5 ms**. Four expected kernels per execution are
present (64 total). No admission trim occurred. The resident control has six
expected calibration/count/sum kernels. Its second calibration was submitted
8.47 ms before the first stream's sum ended, and device kernel overlap is
measurable. Most resident kernel execution still serialized: this establishes
that overlap is possible, not that the kernels overlap substantially or that
batch sizing guarantees it. The pipeline also has zero overlap when considering
all recorded kernels in its NVTX range.

[report.json](report.json) is an unmodified copy of the native workload report.
[timeline-summary.json](timeline-summary.json) contains the derived phase and
handoff measurements, with selected-kernel and all-kernel interval summaries.
[artifacts.json](artifacts.json) binds reports, native traces, SQLite and all
attempt logs by path/size/SHA256. [source-verification.json](source-verification.json)
records rechecked source/runtime hashes; inherited helper source hashes were
also checked against the previous reviewed manifest. No full-source 1.2 TB hash
is claimed. Original source stat and all 64 reference timestamps were checked.

The review CSVs are generated with Python csv.writer from read-only SQLite:

```sql
-- kernels.csv: start,end,stream,name,correlation,globalPid
SELECT k.start,k.end,k.streamId,s.value,k.correlationId,k.globalPid
FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName
ORDER BY k.start;
-- launches.csv: start,end,name,correlation,globalTid
SELECT r.start,r.end,s.value,r.correlationId,r.globalTid
FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId
WHERE r.correlationId IN (
  SELECT k.correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL k
  JOIN StringIds n ON n.id=k.demangledName
  WHERE n.value IN ('gather_canonical_u16','calibrate')) ORDER BY r.start;
-- nvtx.csv: start,end,text,globalTid
SELECT start,end,text,globalTid FROM NVTX_EVENTS
WHERE text IN ('PIPELINE_64_EVENTS','RESIDENT_CONTROL_8_EVENTS','admission_split',
              'read_issue','read_wait','submit_gather_task','admission_pressure_trim')
   OR text LIKE 'task events=%' OR text LIKE 'resident_task%' ORDER BY start;
```

Independent restricted Claude review **passed** code commit `2919ffe33`, with no
blocking findings. The reviewer independently recomputed overlaps, submission
gaps and kernel counts from the readable CSV records. See
[review responses](review-01/response.md) for four low findings, documentation
clarifications and two deferred robustness suggestions. Closure changes only
notes/evidence; the tested executable files and native capture are unchanged.
