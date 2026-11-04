# Persistent GPU task state for XPP cubes

**Proposed, not implemented.** Reviewed against checkout `87bce00cb` on
2026-10-28. API names below are a design sketch, not executable examples for the
current release. The implemented interfaces remain in [user kernels](../user_kernels.md)
and [current design](../design.md).

**Discussion update — 2026-11-02:** Keep
[PR #172's Design 1](https://github.com/slac-lcls/lcls2/pull/172): different
steps may overlap across BDs and across execution slots within one BD, with
one step per execution subbatch/occupied slot. This supersedes this sketch's
local BD step-boundary drain. Retaining multiple XPP bins is an open memory
and lifecycle discussion, not a decision to restrict overlap or add a drain
option. Follow #172's payload scope too: GPU task context carries run/step
identity, not scan/step payload values or SlowUpdate values. Those values stay
on the ordered CPU path. The lifecycle API below remains proposed.

## Purpose

Support image sums that survive many GPU task invocations, with explicit
initialization, finalization, memory accounting, and result delivery. The first
application is an XPP cube:

- Step scan: one scan step identifies one bin. Accumulate its images, export
  the completed partial bin, and reuse storage for the next step.
- Fly scan: events belonging to a bin can occur anywhere in the run. Accumulate
  several bins, then export final bins or mergeable partial bins.

The original pseudocode used a global `pixel_array_sum` and proposed
`beginrun`/`endrun` methods. The desired feature is **run-owned, per-stream task
state**. Python module globals and Python destructor timing do not provide the
required ownership or completion rules.

The historical XPP workload was `xpp101605526`, run 216, with `jungfrau1M`
panels of shape `(2, 512, 1024)`. It calibrated with `cmpars=(7, 0, 10)`, applied
a threshold of 5, and accumulated both panel arrays and assembled images,
together with counts. It used ordinary BD distribution, not bin-directed
`set_destination`. This proposal supports that accumulation pattern; it does
not claim scientific equivalence or a measured throughput for a new GPU cube.
The existing [GPU calibration example](../../examples/jungfrau_calibration.py)
does not implement common-mode correction.

## Fit with the current implementation

| Existing mechanism | Extension needed |
| --- | --- |
| Host-only `GpuTask(function, inputs, calibconst)` declaration | Optional factory for a stateful implementation |
| One callback per nonempty selected execution subbatch | Preserve batching; invoke `process(batch, stream)` |
| Stable CUDA execution streams in `EventPool` | One task instance per stream and run instance |
| Input leases and producer completion events | Retain persistent state through submitted work and aggregate export |
| Per-event `batch.publish()` and automatic D2H | Separate keyed step/run aggregate delivery |
| Framework allocation ledger | Charge declared persistent state and reserve execution/export headroom |
| Transition dispatch and pipeline close | Ordered lifecycle hooks and finalization exactly once |

Relevant source: [task declaration](../../gpu_task.py),
[execution pool](../../gpu_stream.py), [producer dispatch](../../gpu_producer.py),
[event manager](../../gpu_events.py), and [allocation ownership](../../gpu_allocation.py).

A callable object can already retain application-owned arrays between calls.
Today the same callable is invoked on different streams, its arrays are outside
framework accounting, and there are no automatic task lifecycle hooks. Such a
prototype is possible, especially with one stream, but it is not this managed
state feature.

## Ownership and concurrency

State identity is:

```text
(run instance, task identity, BD worker, CUDA execution stream)
```

Use a run-instance identity, not just the numerical run number: reopening a
run must not reuse an old accumulator. Here “stream” means a CUDA execution
stream, not an XTC file stream.

```text
Run
  BD 0
    CUDA stream A -> task instance A -> sums and counts
    CUDA stream B -> task instance B -> sums and counts
    combine contributions with matching keys -> BD partial bins
  BD 1
    ...
  writer/application -> combine BD partial bins -> final cube
```

Instances survive execution-slot retirement. Borrowed input arrays, batch
contexts, and raw pointers do not become persistent merely because the task
does. Never save them for a later invocation. Task state owns its allocations
independently of input windows and slot leases.

All instance methods that enqueue GPU work run on the assigned current stream.
Updates to one instance are stream ordered. Different instances have disjoint
mutable arrays; no cross-stream atomics are needed for this ownership scheme.
An accumulation kernel still needs a correct reduction or atomic strategy when
multiple threads in the *same invocation* contribute to the same bin/pixel.

A shared accumulator per BD is an alternative requiring explicit synchronization
or suitable atomic kernels. It is not the initial design. The existing CUDA-IPC
sharing of immutable calibration constants must not be reused as an implicit
sharing mechanism for mutable sums.

## Proposed declaration and lifecycle

Preserve existing `GpuTask(function=...)` behavior. Add a mutually exclusive
`factory=...` form. The factory constructs a fresh instance for each execution
stream, on the assigned GPU worker. Configuration and factory construction are
host-only; device allocation starts in `begin_run`.

Provide a proposed `GpuTaskWorker` base class for user implementations.
`process` is abstract and required; lifecycle hooks default to no-ops so users
override only the methods their analysis needs. These are Python overridable
methods, analogous to virtual methods. The framework owns hook ordering and
completion; inheriting the base class alone does not provide those guarantees.
Keep `GpuTask` as the host-only declaration and `GpuTaskWorker` as the mutable
worker implementation.

```python
from abc import ABC, abstractmethod

class GpuTaskWorker(ABC):
    def begin_run(self, ctx, stream):
        pass

    def begin_step(self, ctx, stream):
        pass

    @abstractmethod
    def process(self, batch, stream):
        ...

    def end_step(self, ctx, stream):
        pass

    def end_run(self, ctx, stream):
        pass

    def close(self):
        pass
```

The name and export location remain proposed. Constructors store host-only
configuration, allocation starts in `begin_run`, and `close` is idempotent
reference cleanup after GPU work and export readers finish. Required science
output belongs in finalization hooks, never in a Python destructor.

The proposed declaration also specifies aggregate names/merge operations and
a persistent-state byte limit per stream. The following fields are all new:

```python
from functools import partial

task = GpuTask(
    factory=partial(XppCube, mode="step", binning=binning, science=science),
    inputs=["jungfrau1M.raw", *binning.input_selectors],
    calibconst=science.constant_selectors,
    state_bytes_per_stream=state_capacity_bytes,
    aggregate_outputs={
        "cube": {"pixel_array_sum": "sum", "count": "sum"},
    },
    aggregate_sink=local_cube_writer,
)
# ds = DataSource(..., gpu_fn=task, n_gpu_streams=2)
```

`binning` and `science` are host-only configurations for user algorithms. Each
instance must create its own mutable algorithm state; only immutable
configuration may be shared. Detector and constant selection retain today's
validation rules. `aggregate_sink` is a host consumer on each BD (or the serial
worker), not a callback automatically run on MPI rank zero.

| Method | Invocation and guarantee |
| --- | --- |
| `__init__(...)` | Host configuration only; no CUDA allocation |
| `begin_run(ctx, stream)` | Once per instance, after placement and input metadata are available |
| `begin_step(ctx, stream)` | Correct ordered step identity is available; no scan/step payload values |
| `process(batch, stream)` | Once per nonempty execution subbatch; borrowed inputs retain existing contracts |
| `end_step(ctx, stream)` | All admitted contributions for this instance's step have completed; export is allowed |
| `end_run(ctx, stream)` | All admitted run contributions have completed; export remaining state exactly once |
| `close()` | Idempotent reference cleanup, after all GPU work/export readers complete; no science output or MPI |

The lifecycle context supplies an immutable run/step identity, termination
status, a budgeted state allocator, and aggregate emission. It does not expose
scan/step payload values, SlowUpdate values, or the live mutable Run/EnvStore.
Use the shared source-step identity from #172, qualified by run identity
and based on the BeginStep timestamp; a BD-local generation is not a cross-BD
bin key. The process batch exposes the same run instance and step identity as
its captured scope, plus its existing aligned inputs and timestamps. A stable
stream ID may be exposed for diagnostics; applications should not key state by
a raw CUDA pointer.

Initial input shape/layout metadata remains available for allocation; it is
not changing scan/step payload state. Requested calibration constants are
initialized once per run and remain unchanged across steps. The CPU application
associates exported step identities with physical scan/bin values separately.

Instances with no events still participate in lifecycle completion. Their
initialized sums/counts are zero; the reducer can omit their numerical payload
while recording completion. Shapes must therefore come from validated input
metadata, not solely from the first event.

## Refined pseudocode

This version implements one accumulator bank per stream. For step scans the
bank contains the current step's label bins. For fly scans it contains all
configured `(bin, label)` combinations and must fit the declared state budget.
The bounded-cache extension for larger cubes is described below.

```python
class XppCube(GpuTaskWorker):
    def __init__(self, mode, binning, science):
        # Host-only configuration; one new instance per run/BD/CUDA stream.
        self.mode = mode
        self.binning_config = binning
        self.science_config = science
        self.sums = self.counts = None
        self.active_step = None

    def begin_run(self, ctx, stream):
        self.binning = self.binning_config.make_worker(ctx)
        self.science = self.science_config.make_worker(ctx)
        event_shape = ctx.input_spec("jungfrau1M.raw").event_shape
        # Includes label combinations; one label gives one bank row per bin.
        self.keys = self.binning.bank_keys(mode=self.mode)
        self.sums = ctx.state.zeros(
            (len(self.keys), *event_shape), dtype="float64")
        self.counts = ctx.state.zeros((len(self.keys),), dtype="uint64")

    def begin_step(self, ctx, stream):
        if self.mode == "step":
            assert self.active_step is None
            self.active_step = ctx.step_id
            # Scope identity only; no scan/step payload values enter the task.
            # Previous end_step exported and cleared the bank.

    def process(self, batch, stream):
        if self.mode == "step":
            assert batch.step_id == self.active_step

        # User science: calibration, common mode if required, threshold,
        # masks, and event selection. Scratch uses admitted workspace and
        # is retained before kernels are launched, through batch completion.
        images, valid_pixels, accepted_events = self.science.prepare(
            batch, stream=stream)
        # Step mode selects label rows within this step's bank using declared
        # event inputs/static configuration; step_id keys the exported bin.
        # Fly mode assigns scientific bins from declared aligned event fields.
        bin_rows = self.binning.assign(batch, stream=stream)

        # Updates persistent state IN PLACE on the supplied stream.
        # Invalid bin rows are excluded. No event image is published/copied.
        accumulate_by_bin(
            images, bin_rows, valid_pixels, accepted_events,
            out_sums=self.sums, out_counts=self.counts, stream=stream)

    def _export_and_clear(self, ctx, stream):
        # emit snapshots each keyed record with bounded host staging.
        # Baseline contract: all reads from these device arrays finish
        # before emit returns. The runtime records source/flush identity.
        for row, key in enumerate(self.keys):
            ctx.emit(
                "cube",
                key=ctx.bin_key(step_id=self.active_step, bin_label=key),
                pixel_array_sum=self.sums[row],
                count=self.counts[row:row + 1],
            )
        # Safe because export no longer reads either allocation.
        self.sums.fill(0)
        self.counts.fill(0)

    def end_step(self, ctx, stream):
        if self.mode == "step":
            assert ctx.step_id == self.active_step
            self._export_and_clear(ctx, stream)
            self.active_step = None

    def end_run(self, ctx, stream):
        if self.mode == "fly":
            self._export_and_clear(ctx, stream)
        # The controller finalizes an open step before calling end_run,
        # including a truncated step marked incomplete. Never export it twice.

    def close(self):
        # Runtime has already joined method work and export consumers.
        # Runtime-owned allocation charges survive until storage is released.
        self.sums = self.counts = None
        self.science = self.binning = None
        self.active_step = None
```

`make_worker`, `prepare`, `assign`, and `accumulate_by_bin` represent application
algorithms, not proposed built-in Jungfrau calibration/binning routines. Their
memory requirements must be included in admission. `ctx.state.zeros`,
`ctx.input_spec`, `ctx.bin_key`, `ctx.emit`, and `batch.step_id` are proposed
framework interfaces. The current `batch.step_generation` is not sufficient
until ordering is corrected.

For event `e`, bank row `b`, and pixel `p`, the kernel contract is:

```text
sum[b,p] += images[e,p] when accepted[e], bin_rows[e] == b, valid_pixels[e,p]
count[b] += 1          when accepted[e], bin_rows[e] == b
```

The event count increments once per event, not once per pixel. Define the
acceptance policy for missing segments and damaged data explicitly. A scalar
event count does not describe pixel-dependent exposure; applications requiring
pixelwise means must also accumulate valid-pixel counts with the same shape as
the sum. Bin edges, out-of-range/NaN exclusion, labels, and segment order must be
identical across BDs. Preserve sums and counts; do not merge partial averages.

For a single bin with all images valid, the operation is conceptually:

```python
# images: (N, segments, rows, columns); sum_images: (segments, rows, columns)
partial = workspace.empty(sum_images.shape, dtype=sum_images.dtype)
cp.sum(images, axis=0, dtype=sum_images.dtype, out=partial)
sum_images += partial
```

Here `workspace` denotes admitted scratch retained through producer completion.
`cp.sum(images, out=sum_images)` would overwrite the accumulator and, without
`axis=0`, reduce pixels as well. `cp.array((x, y, z))` creates three values;
`zeros((x, y, z), dtype=...)` creates the intended image storage. Packed Jungfrau
raw words include gain bits: summing raw uint16 words is not calibrated science.

## Transition ordering and step completion

This feature depends on the ordered transition work tracked by
[#153](https://github.com/slac-lcls/lcls2/issues/153). In the inspected source,
the EB may append BeginStep or EndRun after L1 events in a packet. The manager
currently calls `_handle_steps()` before submitting that packet's events.
Adding lifecycle hooks at that point would finalize too early or assign the
wrong step. EndStep itself currently does not require a drain.

Plan execution and transitions in source order. For the sketch's single-bank
worker, order its own lifecycle as follows; this is not a barrier across the
BD's streams:

```text
events for step A -> complete their contributions -> end_step(A)
                 -> begin_step(B) -> events for step B
last admitted run events -> complete -> end_step if open -> end_run -> close
```

Keep one step per execution subbatch. Discover compact event ranges and scope
boundaries during existing packet traversal, then intersect them with memory
admission ranges. Do not add per-field scheduling, repeated descriptor scans,
or rebuild bulk-read plans for each step. Input read grouping remains distinct
from execution boundaries.

Keep PR #172's overlapping-step baseline. Different execution slots within a
BD may hold different steps, and BDs advance independently. One execution
subbatch/occupied slot contains only one step; slot reuse waits for its
consumers. BeginStep/EndStep alone do not impose a blanket BD drain or a global
barrier. Persistent task state has a separate lifetime from slot storage.

For the single-bank sketch, a worker must finish exporting its old scope before
reusing that bank for the next scope. Other workers may still contribute to
the old step. The BD reducer retains that step's completion identity until all
participating streams finish. The exact hook scheduling and completion protocol
remain to be designed; replayed history and empty steps must not duplicate
output or hooks.

Supporting these hooks does not itself implement public `run.steps()` and
`step.events()`. Existing GPU step-iteration limitations remain until the
ordered controller supplies those adapters. Internal hooks must work while
users consume `run.events()`.

### Open discussion: XPP bin retention with overlapping steps

For a step scan with one bin per step, stream A can still accumulate bin A
while stream B accumulates bin B. Exported partials for A may also wait in the
reducer. Thus a BD can have several distinct bins live at once. Even with
step-boundary draining, one accumulator per stream already means multiple
arrays; they would simply contribute to the same local step/bin.

Overlap need not multiply GPU accumulator storage beyond one bank per stream
if export completes before each bank is reused. However, pending partials,
merge buffers, staging, completion metadata, and sink queues also consume
memory. A fixed stream/slot count alone does not bound retained output across
successive steps. For XPP's `(2, 512, 1024)` panels, each float64 image sum is
8 MiB, before counts, labels, assembled images, scratch, or export storage.

Discuss with the transition-design work:

- When can a worker export/reuse its bank while other workers are still on
  that step, and how is final step completion recorded?
- What byte limits cover live GPU banks, host partials, and sink retention?
- When those limits are reached, how should bounded partial export and
  resource backpressure preserve progress without a blanket step barrier?
- How should delayed streams/BDs and slow sinks be tested to demonstrate
  bounded retention and correct bin sums?

These are follow-up questions for the stateful task feature, not new approval
gates for #153/#172. No local-drain scheduling option is selected by this note.
Fly-scan bins can be interleaved throughout a run regardless of step overlap;
their dense-state or bounded-cache policy remains a separate requirement.

## Aggregate delivery and reduction

Persistent state is mutable working storage; aggregate output is a snapshot of
that storage. Do not call the state declaration `run_scope_outputs`: step and
periodic outputs may be produced from the same run-lifetime allocation.

`ctx.emit` has no event-row axis and does not attach results to an arbitrary
event. Each record contains run identity, logical bin/label, step identity when
applicable, array shape/dtype and segment identity, sums/counts, completion
status, and a runtime-assigned source/flush identity. Different partial flushes
for the same logical bin must be added, not overwritten. Runtime retries must
not redeliver an already accepted record; distributed crash recovery is outside
the initial scope.

The initial export path is deliberately bounded and synchronous with respect
to reading device source arrays. It stages one bin or bounded pixel tile at a
time, then permits reset/reuse only after that read finishes. Lifecycle methods
can enqueue clearing or other GPU work; the runtime records and joins their
completion too, even when there is no subsequent event callback.

Combine stream contributions within the BD before sending final step/run
partials to the writer where practical. A bounded host reducer is sufficient
initially; synchronize each source stream before its copy. It must track stream
completion independently of numerical records, including empty streams.
Charge its resident merge buffers and staging to an explicit host cap. Spill
mergeable partials to the sink when the cap is reached; never retain an
unbounded cube dictionary in a hidden output queue.

A later GPU merge can reduce one bin/tile at a time on a merge stream that waits
for all contributing producer events. That requires budgeted merge workspace
and the same export completion contract. Async snapshots require either an
immutable copy, buffer ownership transfer, or a completion token that gates
reuse; `keepalive` alone is not a reuse notification.

The host sink receives independent arrays plus metadata. It can write per-BD
partials or use an explicit application aggregation protocol. Any retained
arrays count against a bounded sink policy. No hidden `COMM_WORLD` collective
belongs in `end_step`, `end_run`, or `close`: other psana roles do not execute
these hooks, and BD early exits need not occur together. A collective reducer
requires an explicitly established participant communicator and completion
protocol, including workers with no contributions.

## Memory policy and large fly-scan cubes

For dense state, excluding small count/key arrays:

```text
state bytes per GPU = BDs sharing GPU * streams per BD
                    * resident bin/label combinations * pixels * sum itemsize
```

XPP's 1M panel stack needs 8 MiB per float64 sum. With 256 bins and two streams,
that is 4 GiB per BD before counts, additional labels, assembled images,
calibration constants, inputs, scratch, parser storage, and export workspace.
Ordinary event distribution can give every BD contributions to every bin.

Reserve each stream's declared state capacity during setup, before event
admission. Allocation through `ctx.state` consumes that reservation rather than
charging it twice. Actual backing allocations and aliases retain charges until
release. Validate the sum of all stream reservations against the per-BD budget
and preserve enough execution/export headroom to make progress. Fail with a
capacity diagnostic if the minimum working set cannot fit. Do not repeatedly
retry by shrinking batches when persistent state alone consumes the budget.

The baseline sketch uses a dense bank and rejects configurations that do not
fit. Periodically clearing that same dense bank does **not** reduce its memory
requirement. For larger fly-scan cubes, add a bounded resident-bin table:

1. Map selected events to scientific bin/label keys, preserving input ordering.
2. Partition work so each admitted execution's required resident keys fit, or
   process the execution in bounded key groups while retaining its input lease.
3. Before admitting a new key, export an existing key's mergeable partial sum
   and count, wait for its readers, and reuse the entry.
4. Permit an evicted key to reappear later. The sink adds all partials by key.

Track a flush sequence per source, and bound metadata, merge buffers, host
staging, and queued sink work as well as device arrays. The cache policy must
not introduce unbounded event-image retention or an event-by-event host loop.
Benchmark the additional grouping and export traffic separately from dense-bank
performance.

## Binning inputs and routing

For a step scan with one bin per step, use immutable step identity to separate
accumulation and key exported partials. The GPU task does not need the physical
scan value. CPU transition handling retains scan/step payload values; the
application can associate them with exported identities for labeling or
combining bins on the CPU. Scheduling look-ahead may export a partial before
public CPU delivery reaches its step, so this association must use the matching
recorded identity rather than the CPU's mutable current-step value. Any pending
association/output storage must obey the host/sink budget.

Fly binning needs a per-event value available before accumulation. Use declared
aligned device fields (for example a feespec field descriptor). Generic fields
are descriptors, not automatically dense arrays; kernels must validate
presence, locator status, types, and extents. A detector sharing an XTC stream
may require `hybrid_det`, with its existing duplicated-I/O cost. Static bin
edges and label definitions can be supplied through host configuration.

GPU EnvStore and extraction, retention, upload, or task-context exposure of
scan/step and SlowUpdate payload values are outside this proposal, as in #172.
An analysis requiring those values inside a GPU kernel needs a separately
designed input path; persistent state does not provide one. The GPU callback
runs before public event delivery, so a CPU event-loop calculation cannot
retroactively supply binning values to that callback.

Persistent state does not add `smd_callback`/`set_destination` support. Those
remain rejected with GPU routing in this checkout. Bin-directed ownership would
require coherent per-destination CPU/GPU packets and is a separate optimization.

## Termination and failure rules

- Normal EndRun: submit all preceding events, complete the final open step,
  export remaining run state, deliver accepted aggregate records, then close.
- EOF without EndRun, `max_events`, or explicit early close: finalize admitted
  work under a documented partial-result policy and mark affected scopes/run
  incomplete. Hooks must not imply that all recorded events were processed.
- A GPU task runs ahead of public event delivery. On early close its partial
  cube can include admitted events that the user loop has not consumed. Record
  contribution counts and scope status; do not describe it as the user's
  consumed-event prefix.
- Callback or GPU failure: stop admission, mark the result invalid, drain safely
  or retain/quarantine owners if completion cannot be proven. Do not export a
  partially updated sum as a successful result. Preserve the existing MPI fatal
  error policy; a lifecycle hook is not a recovery mechanism.
- Setup failure: close only instances/resources that were constructed, including
  partial allocations. No aggregate output is required from failed setup.
- Repeated cleanup or delivery failure: distinguish finalized, delivered, and
  released state so sums are not added twice. `__del__` performs no required
  export, synchronization, MPI, or lifecycle transition.

## Implementation sequence and acceptance

1. Establish ordered event/transition ranges and step identity. Test real EB
   output through the GPU manager, including trailing transitions and history.
2. Add factory instances, budgeted state allocation, and deterministic lifecycle
   calls without changing existing function tasks or no-task behavior.
3. Add bounded aggregate snapshots, completion records, and a host sink/reducer.
   Start with one stream, then combine multiple streams and multiple BDs.
4. Implement the application cube kernels and compare step/fly results with a
   CPU reference using matched calibration, masking, threshold, and bin policy.
5. Add bounded resident-bin eviction if dense fly-scan state does not fit;
   optimize GPU merging or async snapshots only after the ownership tests pass.

Required coverage includes empty runs/steps, missing detector segments, rejected
events, interleaved and skewed bins, duplicate bin IDs across partial flushes,
multiple subbatches per step, boundaries inside packets, final tails, multiple
runs, slow sink/export consumers, constrained budgets, partial initialization,
callback errors, and early/repeated close. Vary batch size and stream/BD counts:
counts must agree exactly, and floating sums must meet an explicit tolerance
because reduction order changes. Test label and pixel-validity policies too.

For psana core implementation, run the required CPU and longer MPI suites plus
GPU device and MPI lifecycle tests. This document adds no runtime behavior and
does not establish a cube performance result. Future measurements must report
input dataset, science policy, bin/label count, state size, batch/stream/BD/GPU
layout, I/O mode, cold/warm conditions, export traffic, and final drain time.

## Synthetic MFX scan dataset: calibrated pilot first

Plan revised November 4, 2026 after independent review. The first deliverable is
a small, calibrated dataset that can be inspected through the normal CPU DataSource pipeline.
The full dataset follows only after this pilot passes. This section records
planned work; it does not report generated files or deployed constants.

Use public `mfx101555026`, run 10, from
`/sdf/data/lcls/ds/prj/public01/xtc/`, including matching `smalldata/` files.
The inspected source has 36,037 events, ten streams, all 32 JF segments, and
supporting camera, QADC, beam-monitor, encoder, gas-detector, EPICS and timing
data. Its total XTC size is 1.224 TB. Preserve detector identities, segment
ordering, correlated recorded payloads, and original event timestamps; only
the scan boundaries and destination run identity are synthetic. No event
repetition is needed.

### Dataset stages and naming

| Stage | Proposed destination | Step event counts | Approximate XTC size |
| --- | --- | --- | --- |
| First pilot | `prjdatpsana`, run 1 | `[19, 20, 21]`, 60 events total | 2.04 GB |
| Full performance fixture | `prjdatpsana`, run 2 | 30 steps of 1,000 events | about 1.02 TB |

Sizes extrapolate the source's average across all streams; measure the actual
pilot output. The full fixture uses the first 36,000 source events and omits
the final 37. Keep the pilot as a reproducible fixture instead of overwriting
run 1 with the full dataset. Check run-number availability before publication.
Add empty-step and other boundary cases as separate small correctness fixtures
after the first inspection, without making them prerequisites for inspecting
the basic pilot.

Write under `/sdf/data/lcls/ds/prj/prjdatpsana/xtc/`:

```text
prjdatpsana-r0001-s000-c000.xtc2
... through s009 ...
smalldata/prjdatpsana-r0001-s000-c000.smd.xtc2
... through s009 ...
```

Retain ten streams and one initial chunk per stream. Steps are transitions
inside the files, not separate files. Update BeginRun `runinfo` in **all ten**
streams and assert that all decode to the destination experiment/run; psana's
`DataSourceBase._get_runinfo()` reads only the first stream. The source experiment
string field width must be read from ShapesData: Stage 3 measured **16 bytes**
in this source, correcting the earlier 32-byte assumption. Write `prjdatpsana`
with a terminating NUL, zero-fill the remaining bytes, preserve its shape/field
width, and change the
run number using its declared type. Verify the decoded result, not just bytes.

Configure declares `chunkinfo`, but no source Enable contains a `chunkinfo`
payload. Keep those Names and do not invent a filename payload. If another source
has such payloads, rewrite and validate them against the destination `c000`
filenames. Rebuild SMD offsets and preserve required small-detector content;
the original SMD files cannot simply be copied.

### Calibration is part of the pilot

Before accepting the pilot, freeze and deploy the complete resolved source
calibration snapshot. The November 3 metadata inspection identified collection
`jungfrau_000009` and these selections for source run 10:

| Types | Selection source |
| --- | --- |
| `pedestals`, `pixel_status`, `pixel_rms`, `pixel_min`, `pixel_max` | `cdb_mfx101555026`, records beginning at run 6 |
| `geometry` | `cdb_mfx101555026`, record beginning at run 7 |
| `pixel_gain`, `pixel_offset` | Shared detector DB fallback |

The seven arrays have shape `(3, 32, 512, 1024)` and dtype `float32` in the
inspected metadata: approximately 1.31 GiB total, plus geometry. This is a
metadata inventory, not validation of pixel-status quality or payloads.
Inventory supporting detectors too and copy any constants their normal
interfaces resolve. Retain original detector unique IDs in the XTC Configure.

1. Resolve the source constants using normal psana selection. Save their source
   DB/document/data IDs, selection parameters, shape, dtype, and payload hashes
   in a durable manifest. Save an exact local snapshot, including fallback
   types, rather than relying on future shared-detector selections.
2. Verify calibration-service write authorization for `cdb_prjdatpsana`.
   Filesystem permission does not establish this authorization. Check for
   existing destination records before inserting anything.
3. Copy payloads unchanged using `MDBWebUtils.add_data_and_doc()` into
   `cdb_prjdatpsana/<detector collection>`, with fresh IDs,
   `experiment="prjdatpsana"`, `run=1`, and `run_end=1`. Preserve relevant
   detector/data metadata and record source provenance. Do not reuse source
   GridFS IDs in the destination DB. Use experiment-only insertion: ordinary
   `deploy_constants()`/`cdb add` also writes to the shared detector DB.
4. Read back through normal run-1 lookup and verify every selected type is
   explicitly present in the destination, with matching payload hashes. Do not
   accept success that depends on unrelated shared-detector fallback constants.
5. Before generating run 2, extend the existing experiment records to validity
   1–2 and verify both runs select the same frozen payloads. This supersedes the
   original separate run-2 deployment plan; no duplicate payload upload is needed.

Allow roughly 5–10 GB of working space for the pilot, calibration snapshot and
validation artifacts, subject to the supporting-detector inventory. The full
calibration must be available even though the pilot contains only 60 events.

### Generate and inspect the pilot

Use the first 60 correlated source events. Retain source configuration and
necessary environment state/SlowUpdates. None of the ten source Configures has
scan Names, and each source BeginStep is header-only. Add a scan detector to
stream `s000` in both its bigdata and SMD Configure, preserving existing Names
and payloads. Use the DAQ convention from `psdaq/psdaq/control/BlueskyScan.py`:
`detname="scan"`, `dettype="scan"`, serial `"1234"`, algorithm `raw` version
`1.0.0`, segment 0. Define `step_value` as an integer scalar (INT64 for this
fixture) and `step_docstring` as a CHARSTR string. Use values 0, 1, 2 for the
pilot. The JSON docstring includes `detname: "scan"`,
`scantype: "synthetic_mfx"`, and `step` equal to `step_value`; source experiment,
run and event-index range may be additional provenance fields.

Allocate a free `(nodeId, namesId)` after inspecting all existing Names in that
stream; assert no collision and the NamesId bit limits. Use the same identity
and schema in SMD and bigdata. Append matching ShapesData to **every** BeginStep
in `s000`, including the reused opening BeginStep. Other streams retain their
own header-only BeginStep. Recompute all affected extents and decode each new
payload against the modified Configure before writing. Use DgramEdit or the XTC
Names/CreateData APIs, with buffers sized for the added content. A header service
change alone cannot create this scan payload. Insert this sequence in all streams:

```text
Configure -> BeginRun(prjdatpsana, 1)
for step_value, count in enumerate([19, 20, 21]):
    BeginStep(step_value, step_docstring) -> Enable
    copy count source events, retaining their original timestamps and payloads
    Disable -> EndStep
EndRun
```

Choose inserted transition timestamps within verified gaps in the full
retained datagram sequence, including SlowUpdates; preserve ordering and
cross-stream agreement. Fail generation if a required gap is unavailable
rather than silently changing event timestamps. Reuse legal transition/schema
construction, not just header edits. Record the source-event mapping and
transition schedule in the output manifest. Validate completed files before
publishing their final names; do not overwrite an existing run.

The [timestamp schedule audit](../../../../../notes/synthetic_mfx_stage1_20261103/timestamp_schedule_audit.json)
confirms that all ten source streams have identical service/timestamp sequences.
Both the 60-event pilot and the 36,000-event fixture can preserve every original
event timestamp. The detailed construction rule is:

- Reuse the opening Configure/BeginRun/BeginStep/Enable timestamps, updating
  destination identity and scan payloads as needed. Preserve the initial
  SlowUpdate and later retained SlowUpdates. Do not add a second opening step.
- Compute one shared transition schedule from the merged retained record
  timeline, then apply it to every stream. Corresponding transitions have
  identical service and timestamp, although their payloads may be stream-specific.
- Between steps, insert `Disable, EndStep, BeginStep, Enable`. After the last
  selected event, insert `Disable, EndStep, EndRun`. Discard the source's later
  events and closing transitions for the truncated dataset.
- Let `a` be the timestamp of the preceding retained record and `b` the next
  source record of any service across all streams. Convert both to integer
  nanoseconds, set `delta = min(1000, (b - a - 1) // k)` for `k` inserted
  transitions, and require `delta >= 1`. Assign `a + j * delta`, for `j=1..k`.
  This makes all new timestamps distinct, strictly between the bounding records,
  and avoids collisions with nearby SlowUpdates. The next unselected source
  record can also bound the final closing transitions.
- Convert back using `(seconds << 32) | nanoseconds`, with normalized
  nanoseconds below 1,000,000,000. Do not add an offset directly to the packed
  timestamp across a second boundary. No L1 or embedded timing payload shifts
  are needed for this source.
- Use identical outer timestamps/services in SMD and bigdata, regenerate
  offsets, and validate any Enable chunk payload against the destination's
  `c000` files. This source has no Enable chunk payload. A new step does not
  imply a new file chunk.

For each stream, construct inserted Disable, EndStep, BeginStep, Enable and
EndRun from that stream's own source datagram of the **same service**. Preserve
its complete `env` word (service, transition type, readout-group/control bits),
outer XTC `src`, `damage`, `contains` type/version, and any payload. Do not borrow
another stream's header, zero unknown fields, or turn an L1 header into a
transition. Require a same-service template to exist; otherwise stop.

Timestamp is the only changed field for copied header-only transitions. The
explicit exceptions are the scan ShapesData/ancestor extents in `s000`
BeginStep, scan Names/ancestor extents in Configure, and run identity in
BeginRun. Use matching per-stream templates for SMD and bigdata and compare
transition timestamp, env, src, damage and contains. Validate each extent
against that file's actual serialized record; do not require Configure or L1
extents to match between SMD and bigdata. The [size comparison](../../../../../notes/synthetic_mfx_stage1_20261103/review_20261104/configure_sizes.json)
shows source SMD Configures are 2,096 bytes larger than their bigdata counterparts. Preserve their existing stream-specific
schemas rather than replacing either Configure with the other. The SMD
preflight alone is not the bigdata check. Preserve nonzero source damage rather
than silently clearing it. Record the template
source offset/hash and an allowed-byte-change check for each generated record.
The [source preflight](../../../../../notes/synthetic_mfx_stage1_20261103/review_20261104/source_preflight.json)
records the frozen SMD template headers, offsets and hashes for all ten streams.

Measured pilot gaps after events 19, 39, and 60 are all 8,388,611 ns, so 1,000 ns
spacing works. For the full fixture, the smallest gap to the next source record
is 1,814 ns due to a nearby SlowUpdate; adaptive spacing drops to 453 ns there.
The header/state-machine audit yields the intended counts and strict timestamp
ordering for both schedules. No inserted timestamp crosses a second boundary
in this input; Stage 3 arithmetic checks must also cover a constructed rollover
case so normalized nanosecond conversion is exercised. This is not yet an
EventBuilder execution test.

The writer must assert `BeginStep < Enable < each L1 < Disable < EndStep`,
strict per-stream ordering, transition completeness and service agreement across
streams, and original event timestamp preservation. EventBuilder groups exact
timestamps and rejects incomplete transitions; it does not repair or sort a
malformed input timeline. Serial and MPI DataSource tests in Stage 4 must still
verify the generated files, including boundaries split across SMD/event batches.

Inspect using the standard experiment/run path, with calibration enabled.
Launch serial and MPI checks with `PS_FAKESTEP_FLAG=0` (set it before importing
psana, and record it); fail acceptance setup if a nonzero value is inherited.
The SMD reader otherwise injects artificial step pairs. Use explicit `dir=` and
`live=False`: the last verified logbook file list for run 1 is empty. Recheck
run/file collisions immediately before publication. Disk discovery still makes
a service query; a network failure is not an empty-file result.

```python
import json
import os
assert int(os.environ.get("PS_FAKESTEP_FLAG", "0")) == 0
from psana import DataSource

ds = DataSource(
    exp="prjdatpsana", run=1,
    dir="/sdf/data/lcls/ds/prj/prjdatpsana/xtc", live=False,
)
for run in ds.runs():
    step_value = run.Detector("step_value")
    step_docstring = run.Detector("step_docstring")
    jf = run.Detector("jungfrau")
    counts = []
    for index, step in enumerate(run.steps()):
        assert step_value(step) == index
        assert json.loads(step_docstring(step))["step"] == index
        count = 0
        for evt in step.events():
            assert step_value(evt) == index
            assert json.loads(step_docstring(evt))["step"] == index
            raw = jf.raw.raw(evt)
            calibrated = jf.raw.calib(evt)
            # Compare against the matching source event/reference.
            count += 1
        counts.append(count)
    assert counts == [19, 20, 21]  # Serial inspection.
```

Acceptance before scaling up:

- Run identity is `prjdatpsana`/1; detector identities and source segment mapping
  match. Calibration readback resolves the pinned destination records.
- Serial `run.steps()` yields three steps with the specified counts and scan
  values; a separate `run.events()` pass yields exactly the same 60 timestamps.
- All copied detector payloads agree with the source, including missing/damaged
  data semantics. Compare sampled calibrated JF arrays using identical options,
  dtype, and NaN handling; inspect masks/status per gain and representative
  images. Finite output alone does not establish correct calibration.
- Read supporting detectors and EPICS through their normal interfaces and
  compare values at matching events with the source.
- Run CPU MPI validation and aggregate distributed counts/timestamps: total
  counts must match, with no duplicates, omissions, hangs, or misplaced steps.
  Do not apply the serial per-rank count assertion to MPI. Vary SMD/event batch
  sizes around 20 to exercise boundaries inside batches and final tails.
- Record results, loaded psana version and `PS_FAKESTEP_FLAG`. Check final-file
  ACLs and ancestor traversal, then verify listing/opening XTC and SMD plus a
  small calibrated DataSource read under the collaborator's own login. Group
  membership and Mona's successful reads alone do not complete this check.
  Offline `dir=` discovery can be used before logbook registration;
  it still queries the service, and `live=True` requires a registered file list.

GPU transition tests follow this CPU/reference acceptance and use the current
GPU event API; public GPU `run.steps()` support is not assumed. GPU failures
must be distinguished from malformed fixtures. This first pilot establishes
data/calibration correctness, not throughput or complete transition coverage.
The latest requested full fixture is 30 steps of 1,000 events (30,000 total),
superseding the earlier 36,000-event target. It needs its own verified run
identity. Before generation, extend calibration validity to 1–2 and verify that
both runs select the existing experiment document IDs and frozen payload hashes.
Shared fallback can otherwise select other experiments' newer constants.
See the [validity extension records](../../../../../notes/synthetic_mfx_calib_range_20261104/README.md).

### Stage 1 execution record — November 3, 2026

Stage 1 is complete. See the [verification summary](../../../../../notes/synthetic_mfx_stage1_20261103/verification.json)
and [source/calibration manifest](../../../../../notes/synthetic_mfx_stage1_20261103/manifest.json).
The snapshot, preparation script, selection source, and SHA256SUMS are stored in:

```text
/sdf/data/lcls/ds/prj/prjdatpsana/results/synthetic_mfx/r0001/stage1-20261103/
```

- All ten streams have the same 36,037 unique ordered event timestamps. The
  first 60 timestamps and nearby transition headers are recorded for the pilot.
  Full SMD files are copied and hashed. Bigdata files are inventoried by
  size/mtime/inode and Configure hashes, not full-file hashes or copies.
- Authenticated logbook run listing and run-1 file listing are empty. Run 1 is
  unused at inspection, not reserved. Create/read/fsync/remove probes succeeded
  in `xtc`, `xtc/smalldata`, and `results`; all probe files were removed.
- The read-only calibration privilege endpoint returned `status=true` for
  `cdb_prjdatpsana`. No constants were deployed and no synthetic XTC files were
  created. Stage 2 remains the next action.
- Thirteen unique calibration payloads (1,420,380,186 bytes) are frozen and
  verified: eight JF types and five PV types. All 19 configuration identities
  were inventoried. `inline_alvium`, `qadc_ch0`, and `qadc_ch1` share source
  identity `pv_`/collection `pv_000001`; their selection references the same
  five payloads. Stage 2 must deduplicate insertions by collection/type and
  source document, rather than inserting once per user-facing detector name.
  Preserve this source behavior and inspect the consumers in Stage 4.
  These five PV payloads originate from `rix102353327` run 38 (`alvium3rix1`),
  not a detector-specific MFX calibration. The shared `pv_` identity caused the
  source lookup. The destination `detname="inline_alvium"` labels the first
  deduplicated consumer, not all three consumers or physical calibration
  provenance. Current `pv_raw_1_0_0` exposes raw fields and does not apply these
  constants. Keep the frozen provenance and existing run-1 records unchanged;
  do not describe these as scientifically valid camera/QADC calibration.
- JF `pixel_status` has 16,732 / 11,917 / 12,768 nonzero pixels in its three gain
  planes, each containing 16,777,216 pixels, with no nonfinite values. The
  all-pixels-flagged problem is absent in this snapshot. Calibrated-image
  validation remains Stage 4 work.
- The active runtime's four calibration selection functions match the current
  checkout. Public read transport was adapted for the installed client without
  changing selection; HTTP operations other than GET were prohibited.
- `results` reports a 4 TiB capacity through `statvfs`; `xtc` reports global
  filesystem capacity. Do not treat the results capacity as confirmation of an
  XTC project quota. Pilot write access is verified; collaborator access and
  large-run storage allocation remain later checks.

### Stage 2 execution record — November 3, 2026

Stage 2 is complete. The frozen Stage 1 payloads were deployed only into
`cdb_prjdatpsana`, with `run=1` and `run_end=1`: eight records in
`jungfrau_000009` and five in `pv_000001`. Shared PV selections were deduplicated
before insertion. No shared-detector DB or detector-name mapping was modified.

The destination was empty and write privilege was reconfirmed before deployment.
All 13 payloads (1,420,380,186 bytes) were rehashed before upload. The operation
used `add_data_and_doc()` with an HTTP guard permitting writes only to the
destination GridFS and those two experiment collections. All 26 POST operations
succeeded; new data/document IDs and source provenance were recorded.

Normal `calib_constants_all_types(..., exp="prjdatpsana", run=1)` readback passed
for `jungfrau`, `inline_alvium`, `qadc_ch0`, and `qadc_ch1`: all 23 detector/type
selections resolved to the new destination document IDs, with exact payload
hashes, shapes, and dtypes matching Stage 1. At Stage 2 completion these records
were out of range for run 2. The subsequent authorized validity extension to
1–2 supersedes that restriction; the Stage 2 reports remain historical evidence.

See the [readback verification](../../../../../notes/synthetic_mfx_stage1_20261103/stage2_readback_verification.json)
and [deployment journal](../../../../../notes/synthetic_mfx_stage1_20261103/stage2_deployment_journal.json).
The full deployment plan, script, POST journal, readback report, and SHA256SUMS
are stored under:

```text
/sdf/data/lcls/ds/prj/prjdatpsana/results/synthetic_mfx/r0001/stage2-20261103/
```

No synthetic XTC files were generated. Stage 3 (the 60-event pilot) is next;
event-level DataSource inspection and calibrated-image comparison remain
Stage 4 work.

### Review handoff — November 4, 2026

The review branch is `work/synthetic-mfx-scan` in `monarin/lcls2`, based on
checkout `87bce00cb`. It contains this proposal, the preparation/deployment
scripts, source manifest, deployment journals, readback reports, and timestamp
schedule audit under `notes/synthetic_mfx_stage1_20261103/`. Large calibration
payloads and SMD snapshots stay in the project-storage paths recorded above;
they are not Git artifacts. A clone alone is therefore sufficient for code and
metadata review, but payload verification requires access to those paths.

Stages 1 and 2 are complete. Stage 3 has not generated any synthetic files.
The timestamp audit validates a proposed schedule, not EventBuilder execution.
Do not rerun deployment scripts during review: run-1 records already exist.

For parallel review, use separate worktrees at an explicitly recorded commit.
Record findings by stable ID, affected file/line, evidence, severity, and proposed
fix. Responses should identify the disposition, corrective commit, and validation
evidence. Save audit and response files beside the existing reports and commit
them so each reviewer sees the same findings and resolutions. Avoid concurrent
edits in this shared worktree. This handoff does not authorize additional
database writes or generation of the larger dataset during an audit.

### Response to independent audit — November 4, 2026

The [finding responses and evidence](../../../../../notes/synthetic_mfx_stage1_20261103/review_20261104/responses.md)
close the construction-specification gaps; implementation and output checks
remain Stage 3/4 work. Seema is currently listed in `prjdatpsana`, and the
experiment root, XTC and SMD directory ACLs grant that group read/traverse.
Actual collaborator-session readback remains pending. Stage 1/2 artifacts and
this plan were already committed and pushed at `e118b3c7e`; the report that they
were untracked predates that checkpoint. This audit response does not generate
XTC, redeploy constants, change permissions or establish performance results.


### Stage 3 pilot — November 4, 2026

The [Stage 3 implementation and evidence](../../../../../notes/synthetic_mfx_stage3_20261104/README.md)
now contain a published 60-event pilot: ten bigdata/SMD pairs,
76 records per stream, steps `[19, 20, 21]`, 2,040,571,066 total bytes. Every
record passed source-byte and structural validation, and an uncalibrated
serial DataSource smoke test passed step/event counts, scan values and original
timestamps. Claude passed two review rounds, closing all three initial
nonblocking findings at `3c77c8b189fcdbc99a10760325f0727069e4f249`.
Final-name publication completed without overwrites at
`2026-11-04T18:17:09Z` under `/sdf/data/lcls/ds/prj/prjdatpsana/xtc/`;
SMD files are in `smalldata/`. A second serial smoke from that final directory
passed. Publication and review records are saved beside the implementation,
with durable artifacts and SHA256SUMS in
`/sdf/data/lcls/ds/prj/prjdatpsana/results/synthetic_mfx/r0001/stage3-20261104/`.
No calibration records or psana core were changed. Full Stage 4 acceptance
remains pending. Direct source ShapesData inspection corrected the earlier
experiment-field-width assumption from 32 bytes to 16; see the Stage 3 report.

### Stage 4 local acceptance — November 4, 2026

The [Stage 4 drivers and evidence](../../../../../notes/synthetic_mfx_stage4_20261104/README.md)
passed normal calibrated source/pilot comparison: 23 calibration selections and
payloads, all 60 event records, 79 detector field interfaces, 267 EPICS aliases,
six calibrated JF samples, per-gain status masks and two assembled images.
Serial step/flat-event iteration and nine CPU MPI cases (18 distributed passes)
passed with SMD/event batches straddling the step boundaries, one/two EBs and
multiple BDs. One MPI case checked all six calibrated samples. This is recorded
runtime/pilot acceptance, not a performance or GPU result.

Encoder and user-DIO data are absent in this source interval; matching absence
and the existing encoder descriptions accessor exception are recorded explicitly.
The drivers made no changes to XTC, constants, permissions or psana core.
Claude passed two implementation review rounds, with final reviewed commit
`0c9e57ef20bbe1d0cc4701d0325e232a11290b4d`. Seema's subsequently supplied own-login
report passed the archived script/reference hashes, all 20 file names, eight
Jungfrau calibration IDs and the exact first calibrated image. This closes
Stage 4; see the [collaborator evidence](../../../../../notes/synthetic_mfx_stage4_20261104/collaborator/verification.json).
MPI shutdown
UCX warnings are retained and documented; all nine jobs exited zero after passing
their assertions. The final collaborator script was synchronized to the reviewed
Git blob and its hash recorded. A standalone collaborator
check and full serial reports are available under
`/sdf/data/lcls/ds/prj/prjdatpsana/results/synthetic_mfx/r0001/stage4-20261104/`.
