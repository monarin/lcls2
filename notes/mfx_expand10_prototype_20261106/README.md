# Small blow-up + cube GpuTask prototype

Run `prototype.py` on the existing 1,000-event sample10 fixture. It reuses the
reviewed [Cube task](../mfx_expand10_stage3a_20261105/cube.py) and
[expansion adapter](../mfx_expand10_stage2_20261105/adapter.py); no new XTC data is generated.
The processing flow is:

```python
# Outline; prototype.py supplies fixture checks and frozen calibration setup.
task = Cube(timestamps, nbins=30, active_bins=29, max_streams=2)
ds = DataSource(..., gpu_fn=GpuTask(task, inputs=task.inputs,
                                  calibconst=task.calibconst))
try:
    for event in run.events():
        check_input_errors(event)
    cube, counts = task.finish()
finally:
    task.close()
```

Each callback expands sampled uint16 words tenfold, trims each panel to 512×1024,
calibrates all 32 panels, and adds images to the executing CUDA stream's private
float32 sum bank. Bin = global event ordinal modulo 29; bin 29 stays empty.
State persists across callbacks. After EOF, `finish()` synchronizes the streams,
exports one bank/bin at a time, and merges into a float64 CPU cube. Only small
error flags return per event. This uses today's callable GpuTask interface;
application code owns initialization/finalization, not automatic lifecycle hooks.

From this checkout on SDF, choose a **new** output directory on scratch:

```bash
prototype_dir="$PWD/notes/mfx_expand10_prototype_20261106"
result_root=/sdf/scratch/users/m/monarin/gpu-validation/mfx-expand10/my-prototype
mkdir -p "$result_root/logs"
sbatch --output="$result_root/logs/%j.log" "$prototype_dir/run.sbatch" \
    "$prototype_dir" "$result_root/pilot01"
```

The launcher requests one A100 and the recorded Python/CuPy/KvikIO environment.
Output is `cube.npy`, `counts.npy`, and `report.json`; reserve about 4.1 GB for
output plus caches. Cube shape is `(30, 32, 512, 1024)`. Two GPU banks consume
4.03 GB, additional to the framework input budget; total GPU usage includes
constants, expansion, calibration, and cached buffers. Frozen calibration is
injected only during serial Run construction, as in the reviewed acceptance
harness. This remains a local test fixture, not a published run/DB deployment.

The smoke check validates event order, input errors, counts, finite sums, and
empty-bin behavior. Full pixelwise CPU numerical validation is the earlier
[Stage 3A evidence](../mfx_expand10_stage3a_20261105/README.md), not rerun here.
This small run does not establish 35 kHz, MPI scaling, GDS performance, or native
multi-step behavior. Large-dataset preparation is deferred until after this
prototype; no large dataset generation has started.

Smoke result (November 6): job `43527071` passed on `sdfampere002`, one A100
40 GB, KvikIO compatibility ON. All 1,000 events were consumed, split 501/499
across the two banks. Counts were 35 in bins 0–13, 34 in bins 14–28, and zero
in bin 29. One final export; saved arrays independently reopened and checked.
Observed CuPy pool peak was 6,064,654,848 bytes (not total device memory).
The 18.79 s script elapsed time includes hashing, setup and export, and is not a
throughput measurement. [Full report](smoke-report.json) and
[artifact hashes/readback](evidence.json) preserve the checks. Arrays remain at
`/sdf/scratch/users/m/monarin/gpu-validation/mfx-expand10/20261106-prototype-r1/pilot01/`.
