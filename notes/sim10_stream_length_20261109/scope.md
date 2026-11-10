# Controlled one-BD stream/count comparison

User asks to resolve whether stream count or dataset length accounts for the
earlier ~2.15 versus ~1.19 GiB/s single-BD cold staging results.

Run one BD on one physical A100, same node and GPU throughout: streams 1/2
crossed with 10,000/100,000 events. Two repetitions in reverse order, two short
stream pilots. Use the same accepted 100k Sim10 files; 10k is a prefix, not a
separate file layout. This tests read-window duration, not physical file size.
Batch100, SMD1000, bulk512KiB, budget6GiB, KvikIO compatibility ON,8workers,
1MiB tasks,16MiB bounce, automatic direct I/O OFF; shared physical CPU affinity
and NUMA interleave identical throughout. Keep the previously frozen runtime.

Cold Weka FFB input copy only; no NVMe payload staging. SHA-verified owner-only
copy, zero local file residency after DataSource setup before every run; NIC
physical counters and idle window, backend/SMD caches uncontrolled.
Staging/descriptor task only: no expansion/calibration/cube/payload D2H.

Acceptance: exact event identities, 5 reads/event with exact bytes/histogram,
requested CUDA stream counts exercised, same host/GPU/runtime/placement, correct
BD peer discovery, all resources drained, no CPU payload reads or pressure
drains; independently audit native records and negative-check audit mutations.
Preserve failure evidence; no pass without complete and cleanup receipts.
Report individual repeats, paired ratios and NIC measurements, not statistical
significance or an attributed node effect. Node effects require another node.

Use implement-review-loop, up to three rounds, separate detached read-only
Claude review. Branch work/synthetic-mfx-scan; no production framework changes.
Outputs under project scratch synthetic_mfx/sim10_stream_length_20261109_r1;
temporary FFB copy removed only after work ends.
