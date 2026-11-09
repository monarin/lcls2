Review scoped benchmark implementation and measured claims independently. Do not
launch another review or use any shell/MCP/delegation. Read/Glob/Grep only.

Base: 51acd52b1. Reviewed implementation:
353df539b89d3ee35dea1b58a07ab6b4a602a023.
Repository: detached scratch checkout; ignore unrelated historical notes except
explicit reused helpers. Scope: notes/jf_historical_batch_20261109/scope.md;
inspect all seven committed files there. New benchmark preserves historical
settings with same public GpuTask harness for raw and sim10, unlike the private
adapter used in preceding historical control. No framework implementation edits.

User wants cold Weka FFB psana-GPU raw staging, historical configuration for both
uncompressed raw and sim10, and sim10 batch100 (5x20, half raw20 byte size).
User explicitly retains sim10 bulk512KiB. No NVMe data staging or user kernels.
Use actual native case results for all conclusions; preflight rates are not
reported throughput. Verify exactly 10k events/50k reads, one BD/GPU/stream,
8workers/1MiBtasks/fallback ON, batch20 or100, SMD1000, 6GiBbudget, shared full
physical affinity, NUMAinterleave. Cold means zero per-file client mincore after
setup, not server-cold. NIC counters physical and separately sampled idle.

Acceptance includes exact timestamp identity, five panel streams, exact byte
counts, no CPU payload reads or pressure drains, serialized drained resources,
actual callbacks reaching requested batch sizes, pixel-exact raw preflight,
consistent host/GPU/runtime/code hashes and mounted FFB. Different source runs,
raw dense gather vs sim10 descriptors, and shared storage activity limit causal
claims. Batch20 versus100 within sim10 is the focused comparison.

Prior review suggestions addressed here: frozen sim10 timestamp hash checked
by auditor; drain state serialized; TERM cleanup retained from main launcher.
Preflights warm compiler cache but do not remove lazy process allocations from
loop timing. Node-local directories contain only temp/compiler files.

Reused helpers: notes/jf_raw_cache_20261109/{common.py,runtime.sh,reference.json,
expected-sha256.json,pixels.json}; notes/jf_cold_ffb_20261109/run.py cache helper;
notes/mfx_pipeline_scaling_20261109/instrument.py; notes/mfx_kvikio_read_20261108/
{plan_reads.py,network_watch.py}; notes/mfx_expand10_performance_20261106/
fixture_check.py; notes/mfx_expand10_stage2_20261105/adapter.py;
notes/jf_framework_cold_20261109/sim10_identity_check.json. Native runtime is the
frozen B install, not a rebuild of current checkout; paths and hashes captured.

Hardware completed: job44003240, sdfampere003, all8cases (2preflights plus6full),
13m05s exit0:0. audit.py passed;11 positive/negative tests passed. Native JSON/logs
and Slurm logs are byte/SHA-verified copies at build/history-review-evidence in
this detached checkout. Copy manifest there is copy-index.json. Read native case
JSON selectively (timestamps/callback lists are large); validation.json provides
compact per-case rates, NIC totals, residency, IO and callback histograms.
The README/validation/tests/status records are committed alongside the scripts.
Please check measured claims and limitations against this native evidence.
You cannot run tests or hashes with your allowed tools; acknowledge that limit.
No unresolved previous findings on this new scope. Provide stable IDs for any new
issues and assess latest reviewed SHA, separating blocking from optional advice.
