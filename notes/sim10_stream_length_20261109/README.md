# Controlled one-BD cold Weka comparison

**Moved to Perlmutter at the user's request.** SDF job **44043617** was cancelled
while pending, before any allocation or data generation. No new performance
result exists. Read [TRANSFER.md](TRANSFER.md) for the receiving agent's task,
verified local dataset, runtime candidates, controls and required adaptations.

The question is whether CUDA execution stream count or measured event count can
explain the earlier one-BD results of about 2.15 versus 1.19 GiB/s. All cases use
one BD and the same physical GPU/node within this allocation.

| CUDA streams | Events | Requested JF bytes | Repeats |
|---:|---:|---:|---:|
| 1 | 10,000 | 31.275675 GiB | 2 |
| 2 | 10,000 | 31.275675 GiB | 2 |
| 1 | 100,000 | 312.756747 GiB | 2 |
| 2 | 100,000 | 312.756747 GiB | 2 |

Measured order is the table order followed by its reverse, after two 1000-event
pilots (one per stream count). The 10k cases read a prefix of the same accepted
100k files. This holds file layout fixed and tests measured event count/read
duration; it is not a comparison of physically separate 10k versus 100k files.

Fixed controls: batch100, SMD1000, 512 KiB bulk target, 6 GiB per-BD GPU budget,
KvikIO fallback ON with eight workers, 1 MiB tasks, 16 MiB bounce buffers, automatic
direct I/O OFF. Shared physical CPU affinity and NUMA interleave; existing frozen
B runtime. Same descriptor staging task as the previous BD sweep, with no
expansion, calibration, cube or payload D2H.

Cold means fsync/FADV_DONTNEED followed by zero mincore-resident input pages after
DataSource setup before each case. Physical NIC counters and idle RX are recorded.
Backend and SMD caches are uncontrolled. Input is copied and SHA-verified on Weka
FFB; node-local scratch holds only temporary/compiler files. Native checks require
exact event/byte/read identities, requested execution streams, proper drain, no
CPU payload reads or pressure drains. Independent audit and 24 native-receipt
tests must still run after completion. No benchmark rate is a test threshold.

Implementation commit: `21f56de8884dc386d2d0ae65d159a795ed448614`.
Implementation review passed (code only); the SDF native checks did not run.
Perlmutter porting, native acceptance and results review remain pending. Do not interpret successful submission or code review as measurement
acceptance. See [scope](scope.md) and [status](status.json).

Cancelled SDF native output path (not created):

```
/sdf/data/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx/sim10_stream_length_20261109_r1
```

Scheduler logs are in the sibling directory ending `_logs`, named `44043617.out`
and `44043617.err`. The unique temporary FFB copy is removed on normal completion;
missing complete/cleanup receipts do not pass. Existing optional cancellation/
timeout hardening remains deferred, so failed jobs require process/stage inspection.

Do not resubmit the cancelled SDF job. Continue from TRANSFER.md on Perlmutter
and adapt audit.py/test_audit.py to its native outputs. Record exact
commands and immutable evidence, compare all individual repeats/paired ratios,
and complete the independent results review before acceptance. If only stream
count and event count are ruled out, node or storage-time effects remain possible;
this single-node experiment cannot distinguish those effects.
