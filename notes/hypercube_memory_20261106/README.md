# JF hypercube memory projection and proposed policies

A 32 MiB uint16 Jungfrau event has 16,777,216 pixels. One float32 full-image sum
costs **64 MiB per logical bin**. Two independent stream banks cost **128 MiB per
bin** before other memory. Input sample10 compression does not shrink these sums.

This is an analytic projection, not a GPU OOM experiment or performance result.
It extrapolates beyond the current prototype's explicit 30-bin limit; raising
that limit or adding these memory policies is not part of this change.
[Figure](hypercube_memory.png), [PDF](hypercube_memory.pdf),
[numeric assumptions/results](projection.json), [examples CSV](examples.csv),
and [reproduction script](projection.py) accompany this note.

## Model

For user bin dimensions `(n1, n2, ..., nd)` and L independent labels:

```
B = L * product(ni)
P = 32 * 512 * 1024
A = P * bytes_per_sum + 8                # scalar uint64 count per bin
M_cube = R * B * A
M_device = M_other + M_cube
B_max = floor((device_budget - M_other) / (R * A))
```

R counts independent cube banks **on one physical GPU**, not the number of GPUs.
For the current prototype, one BD process has two CUDA streams and R=2. If four
BD processes each keep two full banks on the same GPU, R=8; their other memory
also adds up. CUDA streams do not inherently require separate banks; that is our
current accumulation policy. A task batch/ProducerContext is temporary and must
not own an ever-growing bank. Persistent ownership is per run/worker/stream.

Plots assume float32 sums, one scalar count/bin, and a fixed **8 GiB reserve**
for all non-cube allocations and safety margin. 40 and 80 GiB are illustrative
per-device budgets, not an assertion about advertised GPU memory or free bytes.
Query actual device memory and sum all process reservations before admitting
work. The reserve includes constants, compressed/raw input slots, expanded raw,
calibrated batches, kernel temporaries, contexts, allocator/pool behavior, and
export/merge staging; it is not guaranteed sufficient. It generally grows with
BD count, stream count and batch size. Lines changing R hold the reserve fixed
only to isolate the cost of cube replication.

For the accepted prototype, persistent state is **calculated** from Cube's
allocation formula: two banks × 30 × (64 MiB + 8 B) = 4,026,532,320 bytes.
The **measured** CuPy pool peak was 6,064,654,848 bytes. The
~1.90 GiB difference is only pool-accounted non-cube storage for that small
configuration; it is not total-device overhead and does not validate the 8 GiB
reserve for larger configurations. Sources: [Cube allocation formula](../mfx_expand10_stage3a_20261105/cube.py)
and [measured pool report](../mfx_expand10_prototype_20261106/smoke-report.json).

| Independent full banks/GPU | Max bins: 40 GiB, 8 GiB reserve | Max bins: 80 GiB, 8 GiB reserve |
|---:|---:|---:|
| 1 | 511 | 1,151 |
| 2 | 255 | 575 |
| 4 | 127 | 287 |
| 8 | 63 | 143 |

These are exact integer limits of the stated sum+count model. Rounded sum-only
limits for two banks are 256 and 576; their extra scalar counts put those boundary
cases just over budget. Real admission needs further allocator safety and cannot
rely on a few KiB of remaining space. Adding GPUs does not change a per-device
limit if every GPU still stores a complete copy of every bin.

| Bin dimensions | Bins | Float32 state, two banks (GiB, rounded) | With reserve (GiB, rounded) |
|---|---:|---:|---:|
| 30 | 30 | 3.75 | 11.75 |
| 10 × 10 | 100 | 12.5 | 20.5 |
| 20 × 20 | 400 | 50 | 58 |
| 30 × 30 | 900 | 112.5 | 120.5 |
| 10 × 10 × 10 | 1,000 | 125 | 133 |
| 100 × 100 | 10,000 | 1,250 | 1,258 |
| 100 × 100 × 100 | 1,000,000 | 125,000 | 125,008 |

Float64 sums double the dominant sum term: two banks allow 127/287 bins in these
40/80 GiB scenarios. Float32 sum plus float32 sum-of-squares has the same dominant
storage as a float64 sum. Per-pixel valid counts add another image/bin; a uint64
count image costs 128 MiB/bin/replica. Pump-on/off labels multiply B. Output means
can be computed from sums/counts later; do not allocate another full resident
cube unless explicitly budgeted. Choice of float32/64 and summation method needs
scientific error requirements: memory pressure alone is not a reason to reduce
precision. Long runs can need higher precision or bounded partial sums with a
more accurate reducer. Different reduction order changes floating-point results.

## Policies to consider

1. **Keep dense banks only when a preflight proves they fit.** Accept user bin
   edges/selectors and output statistics first, calculate the product (including
   labels), and charge persistent state separately from the framework input
   quota. Establish one physical-GPU budget across all BDs. There must also be a
   bounded host budget and output path. A lower batch size cannot fix a cube that
   by itself exceeds the device budget.

2. **Use a bounded active-bin cache as the first general fallback.** Allocate at
   most K full-image bin slots per stream/worker. Empty bins need no device state.
   On eviction or periodic flush, emit mergeable partial sums/counts with run,
   bin tuple, label, tile/segment and unique source/flush identity. A later event
   for the same bin starts a new partial; a flush does not close a fly-scan bin.
   Reduce partials by key, add each exactly once, and spill to chunked storage
   when host memory is full. Keep small bounded queues and apply backpressure.
   Finish only when all contributors have declared completion. Prefer bin reuse
   within a batch when safe; highly interleaved uniform bins may thrash.

   A slot may be cleared/reused only after its accumulation and export have
   finished. Overlapped export needs explicitly budgeted extra slots/staging and
   event dependencies. Event leases must cover every downstream consumer.
   Naive sparse Python/GPU dictionaries without a byte cap do not solve this.
   Sparse allocation helps only if occupied/live bins are substantially fewer
   than the Cartesian product.

3. **Remove stream replication within one process if it is the limiting factor.**
   Keep read/expand/calibrate on two streams, then hand each calibrated batch to
   a single accumulation stream owning one cube. Record producer completion;
   the accumulator waits for it. Input/output leases must survive until the
   accumulation consumer completes. This removes one full cube copy, at the
   cost of serializing accumulation submissions; throughput must be measured.
   Existing independent Cube kernels cannot simply write a shared bank from
   both streams. Atomics or bin-level ownership are alternatives, with contention
   and numerical-order costs. Sharing banks across BD processes is a separate
   IPC/lifecycle problem, not achieved by assigning the same Python object.
   CUDA stream/event dependencies are described in
   [NVIDIA's asynchronous execution guide](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html).

4. **Partition bins across GPUs when resident state or spill traffic justifies
   routing changes.** Give each bin an owner GPU and send its events there before
   the large detector payload is read, if the small-detector bin fields are
   available at that routing point. For dense known bins, balanced contiguous or
   cyclic bin-ID partitioning gives a predictable capacity; hashing is useful
   for sparse keys but needs imbalance headroom and a bounded fallback.
   With two banks per GPU, per-device state is approximately
   `2 * ceil(B/G) * A`, plus each device's own non-cube reserve. The exact minimum
   GPU count is `ceil(B / B_max)` for balanced whole-bin assignment; ceil(B/G)
   rounding matters. For 1,000 bins this means **4 × 40 GiB or 2 × 80 GiB** in
   the illustrative model. These are capacity minima, not performance estimates.

   An explicit bin tuple maps to a stable global ID using a frozen bin-edge
   schema, e.g. `(i,j,k) -> (i*n2+j)*n3+k`; define missing/out-of-range handling.
   All workers must use identical schema/labels. This is a proposed scheduler
   change; ordinary BD event distribution does not automatically colocate bins.
   If bin fields are only available after reading/calibration, owner routing
   requires forwarding images/partials or a metadata pass, and its traffic must
   be budgeted. At the 35 kHz aggregate target, forwarding every float32 image
   would move **2.349 TB/s** before protocol overhead. Prefer metadata routing
   before the payload read when possible. This does not imply all small fields
   are already available in SMD or that routing is implemented today.

   Balanced bin count is not balanced event rate: one hot bin can overload its
   owner. Split hot bins into a bounded number of owner shards, then reduce their
   partials, or distribute pixel tiles for those bins. Both increase complexity
   and/or replication. MPI/CUDA IPC memory is not automatically shared state.

5. **Consider panel/pixel ownership or smaller outputs.** The source has five JF
   DRP streams and 32 panels. Workers could own detector panels and accumulate
   all bins only for those pixels, preserving file locality and full-image
   coverage through distributed output. This avoids routing on each event's bin
   but requires aligned small-detector metadata on all relevant workers, correct
   panel-specific calibration, transition/completion coordination, and final
   assembly. Five DRP files need not balance pixel counts or map to five GPUs.
   Hybrid bin×panel partitioning may be useful. If science permits ROI images,
   spatial rebinning or scalar reductions, reduce P before persistent storage;
   these change the output contract and cannot be silently substituted.

## I/O and host memory do not disappear

A 1,000-bin final float64 cube is **125 GiB**; 10,000 bins is **1,250 GiB**.
Current Cube.finish() allocates a whole float64 result in host RAM, so that
implementation must change for large cubes even if device state is distributed.
Export/reduce/write bounded bins or pixel tiles to chunked storage and publish a
manifest, rather than gathering the entire result into one process. Budget disk
space and reducer memory. Empty unobserved combinations may be represented as
implicit zeros with occupancy metadata if the output format permits.

Let q be accepted events accumulated per exported float32 bin partial, aggregated
across all workers. At rate F, sum-only export traffic is `F * 64 MiB / q`.
Traffic units GB/s and TB/s here and above are decimal (10^9 and 10^12 bytes/s);
memory capacities throughout the note use binary MiB and GiB.
At F=35,000/s this is 2,348.8 GB/s for q=1, 234.9 GB/s for q=10,
23.49 GB/s for q=100, and 2.349 GB/s for q=1,000. Counts are negligible here;
this excludes host merge reads/writes and storage overhead. The cache must achieve
sufficient reuse to meet measured sink bandwidth. These are analytic traffic
requirements, not achieved rates. q depends on ordering, occupancy and cache
capacity; no particular hit rate is assumed.

## Suggested next implementation

Keep the current small dense prototype as a correctness baseline. Next prototype
a configurable byte-bounded active-bin cache with partial reduction, exercised
on the existing fixture with deliberately more logical bins than cache slots.
Verify no lost/double-counted events, counts/sums across repeated evictions,
stream/step interleaving, slow-sink backpressure, aborts and bounded GPU/host
high-water usage. Preserve current CPU comparison and precision criteria.
Measure reuse q and export bandwidth before choosing bin-directed scheduling.
In parallel design work, trace where the actual small-detector parameters become
available to decide whether early bin ownership or panel ownership is practical.
No such runtime changes are implemented by this projection task.

Reproduce with the recorded Stage 2 Python runtime and matplotlib, setting
MPLCONFIGDIR to scratch:

```
MPLCONFIGDIR=/sdf/scratch/users/m/monarin/gpu-validation/hypercube-memory/mpl-cache \
bash notes/mfx_expand10_stage2_20261105/runtime.sh \
  notes/hypercube_memory_20261106/projection.py --output NEW_OUTPUT_DIRECTORY
```

At runtime query actual free/total bytes and maintain owned reservations; a free
memory snapshot is not an allocation guarantee, especially with other processes.
[NVIDIA memory API documentation](https://docs.nvidia.com/cuda/archive/12.9.2/cuda-driver-api/group__CUDA__MEM.html)
describes this limitation. The arithmetic here uses explicit budgets instead.

Independent Claude review passed model/plot commit `9d555787d` with no blocking
findings. [Review responses](review-01/response.md) record documentation-only
clarifications and the distinction between reviewer and coordinator checks.
