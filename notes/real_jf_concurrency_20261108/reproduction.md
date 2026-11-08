# Reproduction and evidence boundaries

`run.sbatch` is a single-node, single-GPU matrix with fresh processes for every
case. Submit it with this directory's absolute path and a **new** project scratch
output directory; set scheduler stdout/stderr to an existing sibling logs dir.
It invokes the frozen native runtime through the earlier stage-2 wrapper. Every
DataSource has `max_events=1000`, including CPU preparation and pixel acceptance.

The script first runs nine existing preparation host tests, prepares a CPU
reference, runs separate GPU pixel acceptance, then three alternating one/two
CUDA-stream clean repeats at 512 KiB, one two-stream clean control at 8 MiB, and
two native two-stream Nsight captures. The captures use the same throughput
mode, without extra Python timing instrumentation. The only always-on pressure
instrumentation counts failed-admission trims; it adds no CUDA events or clocks.
The pixel-acceptance rate is not a throughput result because that case performs
full-image D2H comparisons. The clean/traced cases do not perform those copies.

Each trace can be analyzed on a CPU host with Python 3.9+:

```bash
python3 notes/real_jf_concurrency_20261108/analyze_real.py \
  /sdf/data/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx/real_jf_concurrency_20261108_r3/trace-524288.sqlite \
  /sdf/data/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx/real_jf_concurrency_20261108_r3/traced-524288/report.json \
  --output /tmp/real-jf-524288-new.json
```

Replace `524288` with `8388608` for the control. Output must not already exist.
Run host analysis checks with:

```bash
python3 -m unittest discover -s notes/real_jf_concurrency_20261108 -p test_analysis.py -v
```

The analyzer checks one GPU/context/process, expected gather/calibrate/accept/sum
counts against callbacks, and both execution streams. It sweeps half-open device
kernel intervals, counting distinct stream IDs. `task_kernels` includes the
framework's dense raw gather. `all_kernels` additionally includes parsing and
the CuPy presence-to-error conversion. The captures include warmup. Neither
copy/compute overlap nor a steady-only native interval is claimed.

Within each execution stream, the analyzer pairs dense gather and cube sum FIFO.
It matches the gather CUDA launch API using correlationId, orders executions by
that API's start, and subtracts the preceding sum kernel's device end. Positive
gaps show the next gather was not submitted before the prior sum completed;
they do not identify the responsible host/I/O stage. Runtime API duration totals
are inclusive, not critical-path attribution.

The review CSV exports use Python `csv.writer` and these read-only SQLite queries:

```sql
-- Header: start,end,stream,name,correlation,globalPid
SELECT k.start,k.end,k.streamId,s.value,k.correlationId,k.globalPid
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON s.id=k.demangledName ORDER BY k.start;

-- Header: start,end,name,correlation,globalTid
SELECT r.start,r.end,s.value,r.correlationId,r.globalTid
FROM CUPTI_ACTIVITY_KIND_RUNTIME r
JOIN StringIds s ON s.id=r.nameId
WHERE r.correlationId IN (
  SELECT k.correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL k
  JOIN StringIds n ON n.id=k.demangledName WHERE n.value='gather_canonical_u16'
) ORDER BY r.start;
```

The review worktree gets copies of reports and CSVs plus a copy-index inside its
ignored build directory. Native binary traces/SQLite remain on project scratch,
with paths, sizes and SHA256 in artifacts.json. Execution source hashes are
rechecked against all case reports and the launch hash list. Source XTC files
are bound by size/mtime, all 1,000 event identities, and three full raw/calibrated
sample hashes; we do not claim a fresh full-run hash of the 1.2 TB input.

The CPU reference pass reads the first 1,000 source events and warms cache;
cache residency is not controlled or measured. KvikIO uses CPU fallback.
This is a short concurrency diagnostic, not sustained performance qualification.
