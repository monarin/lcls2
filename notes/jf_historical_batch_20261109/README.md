# Historical psana-GPU staging settings and sim10 batch size

Uncompressed staging reached 4.68–4.77 GiB/s. Increasing sim10's batch size from
20 to 100 raised mean throughput by 7.7%, to 2.11 GiB/s, still well below 5 GiB/s.

| Input | Batch | GiB/s, two repeats | Events/s, two repeats | Peak outstanding input |
|---|---:|---:|---:|---:|
| Uncompressed, dense raw | 20 | 4.6835, 4.7684 | 149.9, 152.6 | 640.1 MiB |
| Sim10, seed descriptors | 20 | 1.9389, 1.9855 | 619.9, 634.8 | 64.1 MiB |
| Sim10, seed descriptors | 100 | 2.1203, 2.1071 | 677.9, 673.7 | 320.3 MiB |

Job **44003240** completed with exit 0:0 in 13m05s on **sdfampere003**, using one
BD and one A100-SXM4-40GB. Implementation: `353df539b89d3ee35dea1b58a07ab6b4a602a023`.
The harness and evidence were committed for review at
`67502407a3b0ba2395a2684f03b650d280e20762`.

Every full case delivered exactly 10,000 ordered events with 50,000 GPU reads.
Raw batch 20 and sim10 batch 20 each made 500 callbacks of 20 events; sim10 batch
100 made 100 callbacks of 100 events. Peak outstanding requests rose from 100 to
500. No CPU payload reads or admission drains occurred, and all GPU reader,
event-slot, and budget resources drained. The complete audit and all 11
positive/negative validation tests passed: [validation.json](validation.json),
[tests.log](tests.log).

Larger sim10 batches increased outstanding bytes without increasing individual
read sizes or reducing total read requests. The mean rose from 1.962 to
2.114 GiB/s, reaching 42.3% of the 5 GiB/s target. This is a modest observed gain
across two repeats, not a bottleneck diagnosis or a statistical confidence claim.
The current input-group path keeps these roughly 0.5–0.7 MiB datagrams as separate
reads at the chosen 512 KiB bulk target. Larger batches alone did not recover raw
bandwidth; request granularity is a candidate for a separate controlled test.

Both inputs ran through DataSource/MPI with the public GpuTask interface. Raw
requested dense uint16 gathering; sim10 requested seed descriptors. There were
no user compute, expansion, calibration, cube, or timed payload D2H operations.
This measures staging, including framework XTC parsing and the raw gather. The
prior exact historical benchmark used a private input adapter with no GpuTask
callback; this campaign retained its settings while using the same public task
harness for both datasets. The different source runs and task input forms limit
cross-dataset causal claims. Within sim10, only the batch setting changed.

Settings: one execution stream, eight KvikIO workers, 1 MiB task size, 16 MiB
bounce buffer, CPU fallback ON, automatic direct I/O OFF, 6 GiB framework budget,
and SMD batch size 1000. All three MPI ranks shared the same 56 physical cores
and used NUMA interleaving. CPU NUMA node 4 was excluded by the available
CPU set (logical CPU IDs selected: 0–31 and 40–63). The exclusive node allocation requested 700 GiB of
memory. Raw used a 1 MiB bulk target; sim10 retained the user-requested 512 KiB.
The selected GPU UUID was `GPU-4ad8ca36-f654-96d2-ae9b-827a636aba74`, PCI
`0000:01:00.0`. Runtime: frozen B install, CuPy 14.0.1, KvikIO 26.10.00; actual
module paths and SHA hashes are in each native result.

Each measured case read 335,571,760,000 bytes (312.526 GiB) for raw, or
33,582,000,000 bytes (31.276 GiB) for sim10. Order: raw20, sim10-20, sim10-100,
sim10-100, sim10-20, raw20. Two 200-event preflights ran first, including three
CPU/GPU raw pixel hashes; these warmed the compiler cache. Lazy per-process
allocations remained inside the measured loop. Setup, SHA checks, copying,
eviction, idle NIC samples, and final mincore scans were outside the loop timer.
Timing covered MPI event iteration through final GPU drain. The sim10 seed
representation is intentionally lossy; no scientific accuracy claim is made.

Both inputs were read directly from **Weka FFB** (`drpsrcf/ffb`). Raw reused the
owned run-387 prefix; sim10 copied the five JF streams and SMD files from the
accepted 10k fixture into a unique FFB directory, removed after the campaign.
No input data was staged on NVMe. Node-local storage held temporary and compiler
files only. The runtime's generic warning mentions NVMe, but it does not identify
the actual input mount.

Before every case, owner-scoped FADV_DONTNEED plus mincore verified zero resident
pages for all five bigdata payload files after DataSource setup. SMD files were
not evicted or measured; their small traffic is included in the NIC counters.
Server cache was uncontrolled and
may have been warmed by prior reads or copying. Physical NIC counters bracketed
the loop, with separate idle samples. NIC bytes include protocol/background
traffic and a slightly wider measurement window; they are not an exact remote
read or cache-hit fraction.

| Case | Final client residency | NIC RX GiB | Payload GiB |
|---|---:|---:|---:|
| First raw20 | 94.349% | 343.962 | 312.526 |
| First sim10-20 | 99.997% | 33.599 | 31.276 |
| First sim10-100 | 99.997% | 34.096 | 31.276 |
| Second sim10-100 | 99.997% | 33.746 | 31.276 |
| Second sim10-20 | 99.997% | 33.362 | 31.276 |
| Second raw20 | 100.000% (rounded) | 339.171 | 312.526 |

Physical NIC: `enp225s0np0`; measured-case idle RX was approximately
0.91–1.00 MiB/s. Final residency differed, so this does not establish equal cache
retention or a particular reclaim mechanism. Shared storage activity remained
uncontrolled.

Native artifacts:
`/sdf/data/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx/jf_historical_batch_20261109_r1`.
Slurm logs use the same path with `_logs` appended. Review copies and their
SHA-verified index are in
`/lscratch/monarin/tmp/jf-history-review-20261109-r1/build/history-review-evidence`.
Original fixtures were preserved; this campaign's temporary sim10 copy was removed.

Independent Claude review: **pass**, no blocking findings, on commit
`67502407a3b0ba2395a2684f03b650d280e20762`. Five low-priority documentation
findings were addressed in the final evidence-only update; the reviewed benchmark
code is unchanged. See [review](review-01/review.json) and
[responses](review-01/response.md). The reviewer could inspect files but could not
execute tests, independently hash artifacts, or access Git metadata outside its
read-only checkout. The coordinator performed those checks.

Validation commands (Python 3.11 from the xpp_drp_gpu_311 environment):

```bash
python notes/jf_historical_batch_20261109/audit.py /sdf/data/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx/jf_historical_batch_20261109_r1
python notes/jf_historical_batch_20261109/test_audit.py /sdf/data/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx/jf_historical_batch_20261109_r1
```
