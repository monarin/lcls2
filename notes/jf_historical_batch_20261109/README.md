# Historical psana-GPU staging settings and sim10 batch size

Measured on sdfampere003, job44003240 completed0:0 in13m05s. Code:353df539b89d3ee35dea1b58a07ab6b4a602a023.

| Input | Batch | GiB/s, two repeats | Events/s, two repeats | Peak outstanding input |
|---|---:|---:|---:|---:|
| raw-dense | 20 | 4.6835, 4.7684 | 149.9, 152.6 | 640.1 MiB |
| sim10-field | 20 | 1.9389, 1.9855 | 619.9, 634.8 | 64.1 MiB |
| sim10-field | 100 | 2.1203, 2.1071 | 677.9, 673.7 | 320.3 MiB |

Sim10 batch100 improved mean bandwidth by 7.7% (1.962 to 2.114GiB/s), reaching 42.3% of5GiB/s. The raw result is close to5GiB/s. Larger sim10 batches alone did not recover raw bandwidth.

Every full case delivered exactly10,000 ordered events with50,000 GPU reads. Raw20 and sim10-20 each made500 callbacks of20; sim10-100 made100 callbacks of100. Peak requests were100 versus500. No CPU payload reads or admission drains occurred, and all GPU reader/event-slot/budget resources drained. These counters show more outstanding bytes but unchanged per-read size; they do not establish which component limits throughput.

The complete audit and11 positive/negative validation tests passed (validation.json, tests.log). All initial per-file mincore counts were0. Final resident fractions:
- 00-raw-dense-b20: 94.349%; NIC RX 343.962GiB for 312.526GiB payload.
- 01-sim10-field-b20: 99.997%; NIC RX 33.599GiB for 31.276GiB payload.
- 02-sim10-field-b100: 99.997%; NIC RX 34.096GiB for 31.276GiB payload.
- 03-sim10-field-b100: 99.997%; NIC RX 33.746GiB for 31.276GiB payload.
- 04-sim10-field-b20: 99.997%; NIC RX 33.362GiB for 31.276GiB payload.
- 05-raw-dense-b20: 100.000%; NIC RX 339.171GiB for 312.526GiB payload.

Physical NIC:enp225s0np0; idle RX about0.94–1.01MiB/s. Both data paths were verified drpsrcf/ffb. First raw final residency was94.35%, versus almost100% for sim10 and the second raw run; this is not equal cache-retention proof or evidence of a specific reclaim mechanism.

Review: pending independent Claude review. Temporary sim10 FFB copy removed; original fixtures preserved.


Native artifacts:
`/sdf/data/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx/jf_historical_batch_20261109_r1`.
Slurm logs: same path with `_logs` suffix.

Both inputs run through DataSource/MPI with one BD and one GPU, using public
GpuTask. Raw requests dense uint16 gathering; sim10 requests seed descriptors.
There are no user compute, expansion, calibration, cube, or timed payload D2H
operations. This is staging throughput including framework XTC parsing and the
raw gather, not the full decompress-plus-cube application. The prior exact
historical adapter used no GpuTask callback; this campaign preserves its settings
while using the same public task harness for both datasets.

Settings: one execution stream, 8 KvikIO workers, 1MiB task size, 16MiB bounce,
CPU fallback ON, auto direct I/O OFF, 6GiB framework budget, SMD1000. All MPI ranks
share all available physical cores and use NUMA interleaving. Exclusive node with
700GiB requested memory. Raw bulk target1MiB; sim10 user-requested512KiB.

Each measured case uses 10,000 events, 50,000 GPU read requests. Raw input:
335,571,760,000 bytes (312.526GiB); sim10:33,582,000,000 (31.276GiB). Nominal batch
payload: raw20=640.052MiB; sim10-20=64.053MiB; sim10-100=320.263MiB. Actual GPU
subbatch sizes and outstanding reads/bytes are checked and reported separately.

Order: raw20, sim10-20, sim10-100, sim10-100, sim10-20, raw20. First run two
200-event preflights, including three CPU/GPU raw pixel hashes; these warm the
compiler cache. Lazy per-process allocation remains inside the measured loop.
Setup, SHA checks, copying, eviction, idle NIC sample, and final mincore scans
are outside the loop timer. Timing spans the MPI event iteration and final GPU
drain. No scientific correctness claim for the lossy sim10 seed representation.

Both inputs are read directly from Weka FFB. Raw reuses the owned run387 prefix;
sim10 copies the five JF streams and SMD files from the accepted 10k fixture to
a unique FFB directory removed on completion. Before every case, owner-scoped
FADV_DONTNEED plus mincore must show zero resident pages for every file after
DataSource setup. The server cache is uncontrolled and may be warmed by prior
reads/copying. Physical NIC counters bracket the loop, with separate idle samples;
NIC bytes include protocol/background traffic and are not an exact cache fraction.

Raw and sim10 originate from different runs and use different task input forms.
The sim10 batch20-versus100 comparison isolates the batch setting within this
matrix; cross-dataset comparisons also include datagram size, gather, and source
layout differences. Node/storage contention remains uncontrolled.
