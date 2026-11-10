Independent implementation review. No delegation, nested review, writes or commands.
Base 1805e1d9522a4e45aa611a59d83a4cf051678104.
Review 95e6c3788e39afdd237bc115481802cd1bd33da8. Round1/3.
Scope: remove repeated input pool scans and slot scans, batch submission, retirement
candidate notification. Preserve exact reads, scheduling, ownership, deferred CUDA
completion, small-stream credit, failure cleanup. No coalescing or reader changes.

Inspect runtime diff in gpu_events.py, gpu_input_group.py, gpu_input_window.py;
new tests in test_gpu_input_group.py and fixture changes test_gpu_slot_selection.py.
Inspect new notes/sim10_input_bookkeeping_20261110 acceptance harness.
Use scope.md and status.json. Prior diagnosis: notes/sim10_trace_20261110/README.md.
Read necessary dependencies but avoid raw prior timing receipt arrays.

Validation supplied: 968 CPU tests pass in8.15s with checkout Python and existing
SDF native modules; PS_PARALLEL=mpi. Log is cpu-tests.log; cpu_tests.py records
bootstrap. Initial targeted42 passed, full run4 failures from slot fixtures that
directly populated _groups without new busy set and1 MPI test with PS_PARALLELnone.
Fixtures now set busy set when simulating state; MPI enabled, full suite passes.
AST/bash syntax/diffcheck/git hooks pass. Native GPU tests and performance PENDING.
Do not infer GPU acceptance from CPU mocks.

Important invariants:
- InputWindow notifies once when closed with no remaining references, before
  completion check. Callback only SimpleQueue.put(state); no pool mutation.
- Pool only drains queue/polls eligible windows on BD thread; candidates survive
  not-ready/errors; identity check rejects stale queued states after key reuse.
- _try_retire actual behavior/CUDA dependencies unchanged; notify is bookkeeping.
- Busy set added/removed with live group ownership; release_raw failure preserves
  group, candidate, slot. close() drains queued stale states and remains retryable.
- issue_groups validates whole plan before I/O; successfully submitted groups on
  partial failure stay in pool for close. Controller pending list can be empty
  on that exception; group pool owns/drains partial submissions. Check this.
- Standalone issue() preserved; plan_slots still polls once on production path.
- Performance does NOT change KvikIO read sizes/counts or callback sizes.

Native acceptance design: source runner copied from accepted stream-length runner
with ONLY explicit old/new overlay setup and runtime validation changes. Frozen
install-B unchanged. Overlay symlinks all unchanged files, copies exactly3 changed
Python files. Runtime identity compares resolved paths+hashes to frozen manifest,
with explicit exceptions for these3. GPU tests ensure import overlay and reject
skips; 6 ownership/transition/task testfiles. Then8 clean old/new x1/2 cases in
forward/reverse order +4timing cases. All10k, same node/GPU/affinity/NUMA, same
512KiB/batch100/6GiB/KvikIOON8workers1MiBtask16MiBbounce/directOFF. Previous exact
cold gate and invariants reused via accepted checks.validate. 20min premiumlcls_g
allocation; percase180s timeout, GPUtests240s, failure record onSIGTERM. All heavy
artifacts on scratch. Prior dataset hashes reused with before/after metadata.

Review for concurrency races, ownership/error regressions, unsupported imports,
benchmark fidelity, and whether tests cover the new behavior. Return structured
verdict with stable finding IDs. Reviewer cannot access Perlmutter native runtime
from SDF; record that limitation. Runtime source snapshots and prior receipts
are in notes/sim10_trace_20261110/frozen if needed.
