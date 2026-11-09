Investigate why sample10 psana throughput is below the standalone four-worker
cold-read baseline, and whether multiple BDs on one GPU help. No psana core or
kernel changes. Use the accepted 10k fixture and frozen runtime. All GPU BDs
must select one physical GPU UUID; four distinct physical CPU cores per BD,
one per helper; record placement. Force KvikIO fallback, four workers, 4 MiB
tasks, 16 MiB bounce, no auto-direct; batch7, SMD20, bulk512KiB, framework4GiB.
Full task: unchanged expansion/calibration/cube kernels, 100 bins/99 active.
MPI: SMD0+EB+1/2/4 BDs. Compare 1/2 BDs with 1/2 streams; 4 BDs only one stream
because 100-bin duplicated banks otherwise exceed A100 40GiB. Sum bank budget
must leave >=8GiB non-bank reserve. Read/parse-only control keeps the descriptor
path but omits expansion, calibration and cube computation; not scientific output.
Whole-loop timing includes start barrier, all event delivery and GPU drains;
setup/cache conditioning/NIC endpoints and validation excluded. No claimed warm
GPU timing window. Both cold-data and explicitly preloaded warm-data cases use
fresh processes; exclude preload time. Cold gate covers ten bigdata files;
SMDs may have been primed before the gate. Keep NIC bytes separate from exact
JF requested/completed bytes and expected ancillary payloads. No exact cache-hit
percentage or backend/node ceiling claim. File-specific eviction only; no global
cache flush. No new dataset or calibration writes. Results on project scratch.
Bounded pilot (400 events) before 10k runs. Exact event-set partition across BDs,
no duplicates/missing identities, task/delivery equality, per-bank exact counts,
empty control bin, reader byte accounting and zero pressure drains required.
Unchanged kernel pixel correctness inherited; no full cube merge/export timed.
Diagnostic CPU exclusive wall/thread time and CUDA-event intervals are separate
from clean rates; nested/overlapping intervals must not be summed as wall time.
Single-node exclusive allocation <=35min; up to three independent Claude review
rounds, preserve failures, commit/push only work/synthetic-mfx-scan.
