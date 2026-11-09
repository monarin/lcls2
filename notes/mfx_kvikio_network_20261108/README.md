# Four-worker Weka NIC/cache profile

Measurement pending. This extends the [read benchmark](../mfx_kvikio_read_20261108/README.md)
with the September 18 Weka methodology, using the same five synthetic JF files.

Three fresh processes each run a full 10k-event cache-cold pass followed by two
warm passes. Each pass completes 50,000 requests / 33,582,000,000 bytes to GPU.
Four KvikIO workers, four logical CPUs, 35 outstanding buffers, 4 MiB tasks,
16 MiB bounce buffers, compatibility ON and auto_direct_io_read=False.
Slurm requests an exclusive node (one GPU, 16 allocated CPUs), while the reader
restricts itself and its workers to four allowed logical CPUs.

The five files are owner-checked and confined to the accepted fixture directory.
After KvikIO initialization, fsync/FADV_DONTNEED evicts only these files; mincore
must show zero resident pages before each cold pass. No payload is changed and
no global cache flush occurs. There is no staging or backend/server cache reset.
These fixtures can be read by collaborators: eviction changes their local-node
cache residency, not contents; a concurrent reader may refill cache and fail the
cold gate. No such failure is silently relabeled as cold.

Each NIC's physical rx_bytes_phy/tx_bytes_phy counters bracket each timed pass.
Counters must exist and must not decrease; physical ports are reported separately,
without adding netdev or IB duplicates. A one-second idle interval is recorded
before conditioning in each process. This is node-wide traffic, including possible
OS/services traffic even with an exclusive Slurm allocation. Idle subtraction is
not used to claim exact file traffic. NIC intervals include endpoint sampling
overhead; application timers only cover request issue through completion/device
synchronization. Cache queries, report writes and GPU/CPU witnesses are outside
both read timing and the NIC endpoints. Near-zero NIC bytes on a warm read with
resident input pages support local-cache service. Read-ahead, protocol overhead,
and other traffic prevent exact cache-hit fractions from subtraction. The warm
label means repeated reads; actual residency is separately recorded.

Run artifacts: `/sdf/data/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx/kvikio_network_20261108_r1`.
Job 43878284. Submission overrides: `--exclusive --time=00:10:00`, with stdout/stderr
under the adjacent `kvikio_network_20261108_r1_logs` directory.

Validation uses the original byte-count/effective-setting/GPU witness audit plus
independent NIC-delta/rate, cache gate and source-hash checks in `summarize.py`.
Historical baseline artifacts remain measurements of their original code commit;
re-auditing them against current source hashes requires that original checkout.
