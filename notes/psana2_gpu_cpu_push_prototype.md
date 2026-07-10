# psana2 GPU/CPU Push Prototype

## Objective

Prototype a psana2 MPI path where SMD0 is unchanged, EB splits each BigData
batch into CPU and GPU work, and BD overlaps GPU BigData fetch/kernel work with
CPU BigData fetch/compute.

The first prototype should prove the scheduling model and data plumbing. It
does not need to make the normal `Run.events()` API expose GPU futures yet.

## Current Path To Preserve

Normal offline MPI flow:

```text
SMD0
  SmdReaderManager / ParallelReader
  sends unchanged SMD chunks to EB

EB
  EventBuilderNode.start()
  EventBuilderManager.batches()
  repack_for_bd(...step history...)
  sends one SMD bytearray batch to one requesting BD rank

BD
  BigDataNode.start()
  Events(get_smd=...)
  EventManager._get_offset_and_size()
  EventManager._fill_bd_chunk()
  EventManager._get_next_dgrams()
```

Keep the existing path as the fallback whenever GPU mode is disabled.

## Prototype Knobs

Add explicit prototype controls, preferably guarded by an environment flag while
the API is unstable:

```python
DataSource(..., gpu_det=["jungfrau"], cpu_det=["ebeam", "epicsinfo"])
```

Suggested fields on `DsParms`:

- `gpu_det`: detector names assigned to the GPU lane.
- `cpu_det`: optional detector names assigned to the CPU lane. If omitted, CPU
  gets every stream not assigned to GPU.
- `gpu_stream_ids`: derived from `dsparms.det_stream_id_table`.
- `cpu_stream_ids`: derived from `dsparms.det_stream_id_table`.
- `gpu_enabled`: true only when `gpu_det` is non-empty and the prototype flag is
  enabled.

Use the existing `DgramManager` metadata:

- `det_stream_id_table`: detector name to stream index.
- `stream_id_to_detnames`: stream index to detector names.

For the first prototype, treat an XTC stream as indivisible. If a stream contains
both CPU and GPU detector names, assign the whole stream to GPU and log that the
CPU detector shares a GPU-owned stream. A later version can split within a
stream only if the XTC layout makes that safe.

## EB Design

SMD0 stays unchanged. EB remains the first place that knows how to split work.

In `EventBuilderNode.start()`:

1. Receive the SMD chunk from SMD0 exactly as today.
2. Build normal EB batches with `EventBuilderManager.batches()`.
3. Pick a requesting BD rank exactly as today.
4. Apply `repack_for_bd()` before lane splitting so step history semantics stay
   identical.
5. If GPU mode is disabled, send the normal batch.
6. If GPU mode is enabled, split the repacked batch into two PacketFooter-valid
   batches and send them to the same BD.

Prefer a single MPI payload containing both batches over two independent MPI
sends. The current BD request protocol expects one response per request; a
single envelope keeps EB/BD flow control simple.

Envelope sketch:

```text
GpuCpuBatchEnvelope
  magic/version
  batch_id
  n_events
  cpu_batch_nbytes
  gpu_batch_nbytes
  event_table_nbytes
  cpu_smd_batch
  gpu_smd_batch
  event_table
```

The event table should include at least:

```text
batch_id
event_index
timestamp
has_cpu_work
has_gpu_work
```

### Split Rules

For L1 events:

- CPU batch contains only CPU-owned stream dgrams.
- GPU batch contains only GPU-owned stream dgrams.
- Unowned streams are represented by zero-sized packets in the event footer.
- Event count and event order are identical in both lane batches.

For non-L1 transitions and step-history events:

- Replicate full transition events into both lane batches for the prototype.
- BD must apply envstore/user-visible transition handling only once.
- Internal GPU/CPU lane managers may still consume transitions for chunk-file
  switching and state updates.

This avoids breaking psana's current invariant that non-L1 transitions are
complete across streams.

## BD Design

Add a GPU-aware batch executor beside the existing `Events/EventManager` path.
It can be a new class initially, for example:

```text
GpuCpuBatchExecutor
  parse envelope
  create CPU EventManager for cpu_smd_batch
  create GPU descriptor manager for gpu_smd_batch
  schedule GPU descriptors first
  run CPU lane while GPU is in flight
  join in event order at batch fence
```

Do not request the next EB batch until the current batch fence is complete. That
keeps the first prototype simple and matches the proposed model.

### GPU Descriptors

BD already has the data needed in `EventManager._get_offset_and_size()`. Reuse
that logic rather than reparsing dgrams elsewhere.

Represent one GPU event as:

```python
GpuEventDesc(
    batch_id: int,
    event_index: int,
    timestamp: int,
    reads: list[GpuReadDesc],
)

GpuReadDesc(
    stream_id: int,
    fd: int,
    offset: int,
    size: int,
    detnames: list[str],
)
```

Keep all GPU-owned streams for one event on the same GPU for the first
prototype. That makes event-level join logic and detector kernels simpler. A
later version can split one event across devices if needed.

### GPU Scheduling

Default ownership should avoid cross-rank GPU oversubscription:

- If there are multiple BD ranks on a node, map each BD rank to one visible GPU,
  usually `local_bd_rank % n_visible_gpus`.
- Within a BD rank, round-robin GPU events across a ring of slots/streams on
  that GPU.
- Only enable multi-GPU scheduling inside one BD rank when explicitly requested.

Each slot owns:

```text
pinned host input buffer
device input buffer
device output buffer
pinned host output buffer, if D2H is needed
nonblocking CUDA stream
done CUDA event
```

Per descriptor:

```text
wait for reusable slot if necessary
fetch BigData into device input
launch detector kernel(s)
optionally enqueue async D2H
record done event
return GpuTicket(event_id, slot, done_event, result_ref)
```

For CPU safety, only call `event.synchronize()` when BD must consume D2H output
or reuse host/device buffers. Use CUDA stream ordering for H2D/kernel/D2H within
one slot.

### BigData Fetch Backends

Support two backends behind the same descriptor API.

No-GDS backend:

- Read XTC bytes from `fd, offset, size` into pinned host memory.
- Prefer `preadv` or a small C/Cython helper that writes directly into the
  pinned buffer.
- Enqueue async H2D on the slot stream.
- A pure Python `os.pread()` fallback is acceptable for correctness but adds an
  extra host copy and will hide less CPU/GPU overlap.

GDS backend:

- Register/open the XTC file handle for cuFile.
- Read directly from file offset into device memory.
- Fall back to the no-GDS backend for unsupported filesystems, unaligned reads,
  or cuFile errors.
- Record backend, alignment, fallback count, and bytes read in debug metrics.

### CPU Lane

Once GPU tickets are enqueued for the batch, BD processes CPU work:

```text
for cpu_event in cpu_event_manager:
    fetch CPU BigData through current EventManager logic
    run CPU compute/callback for CPU detectors
    store CpuResult(batch_id, event_index, timestamp, result)
```

For the first prototype, make CPU compute a small internal callback or benchmark
hook. The existing public `Run.events()` loop is pull-based; if user analysis
stays outside BD, there is little opportunity to overlap CPU compute with GPU
work.

### Join

At the batch fence:

```text
for event_index in batch order:
    cpu_result = cpu_results.get(event_index)
    gpu_ticket = gpu_tickets.get(event_index)
    if gpu_ticket:
        wait for gpu_ticket.done_event
        materialize D2H result only if needed
    emit/record joined result
```

Events with only CPU work do not wait on GPU. Events with only GPU work still
participate in ordered join.

The first prototype can record joined results and timing statistics rather than
changing the public psana event object. A later version can decide whether the
public API should yield joined events, GPU result handles, or detector-specific
arrays.

## Implementation Touch Points

- `psana/psana/psexp/ds_base.py`
  - Add prototype kwargs and `DsParms` fields.
  - Derive stream partitions after configs are available.
- `psana/psana/psexp/node.py`
  - Extend `EventBuilderNode.start()` after `repack_for_bd()`.
  - Add envelope send path guarded by `dsparms.gpu_enabled`.
  - Keep existing send path unchanged for normal mode.
- `psana/psana/psexp/event_manager.py`
  - Factor offset/size table access so a GPU descriptor builder can reuse it.
  - Avoid changing normal `_get_next_dgrams()` behavior.
- New prototype module, suggested:
  - `psana/psana/psexp/gpu_cpu_batch.py`
  - Envelope encode/decode, batch splitting, descriptor building, scheduler.
- Optional debug entry point:
  - Extend `psana/psana/debugtools/ds_gpu_stream_profile.py` or add a focused
    MPI prototype driver.

## Metrics To Add

Per BD rank:

- received batches/events
- GPU descriptors/events/bytes
- CPU events/bytes
- GDS bytes and fallback bytes
- GPU enqueue time
- CPU lane time
- GPU wait-at-join time
- total batch fence time
- slot reuse waits

Per EB rank:

- split time
- CPU/GPU batch bytes
- events with CPU only, GPU only, and both

## Validation Plan

1. GPU disabled: existing MPI event count and transition behavior unchanged.
2. GPU enabled with empty `gpu_det`: identical to normal mode.
3. One GPU detector:
   - CPU/GPU lane event counts match the envelope event table.
   - GPU descriptors match `EventManager` offsets/sizes for the same streams.
   - Joined event order matches original EB batch order.
4. Transition/chunking run:
   - Enable chunk switches still update file descriptors before following L1s.
   - EnvStore updates happen once, not once per lane.
5. Performance smoke test:
   - GPU enqueue occurs before CPU lane.
   - CPU lane overlaps kernel/D2H in Nsight Systems or CUDA event timeline.
   - Join wait is small when CPU work is long enough to cover GPU work.

## Open Questions

- What is the first real GPU kernel target: raw byte checksum, detector raw
  extraction, calibration, or image construction?
- Should first public output be a benchmark/statistics mode, or a joined psana
  event carrying GPU result handles?
- Should GDS be part of the first code path, or should the first merge use
  pinned-host reads with a GDS-compatible interface stub?
- How should BD rank to GPU mapping interact with Slurm GPU allocation and
  `CUDA_VISIBLE_DEVICES` on psana nodes?
