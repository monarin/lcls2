# Responses to review of 36b48dbabaeb00514fcfa091f827650d28b76bad

The reviewer requested changes with six non-blocking findings. All are accepted;
none is disputed or deferred. Repairs require re-review on a fresh snapshot.

| ID | Response | Verification |
|---|---|---|
| HOLD-IMPL-001 | Cooperative parent-signal release during acknowledged holds; cleanup grace before kill. A worker signalled in a hold records cancellation, not a new camera failure. | Assert parent/child reason is parent_signal, successful peer has no failure.json, original target remains failed; direct successful-worker signal records cancellation. |
| HOLD-IMPL-002 | Preserve released/cleanup_failed state and original release reason on later cleanup errors; hold.json and summary inspection data agree. | Operator and timeout release followed by cleanup hang retain cleanup_failed and the original reason. |
| HOLD-IMPL-003 | Normalize connection reset/broken pipe on receive to EOFError. | Real AF_UNIX peer close with unread data is reported as EOF. |
| HOLD-IMPL-004 | Add subprocess lifecycle coverage using the real InspectionRecorder and endpoint against software-only fake nodes. | Real rotating files and cached updates appear, original SHA stays fixed, camera calls do not change during hold; release adds only shutdown/stop. |
| HOLD-IMPL-005 | Document wildcard server's advertised loopback versus physical identity host and same-host/three-port-tunnel requirements. | Operator README updated. |
| HOLD-IMPL-006 | Record implementation/review SHAs and preserve the distinction between code and later evidence commits. | Status and round contexts identify exact snapshots; post-commit status is saved in a separate evidence commit, avoiding self-referential SHAs. |

The reviewer's pinned-library source access was restricted to the detached
worktree; its source semantics assessment therefore relied on supplied runtime
proof and prior plan evidence. This is retained as a limitation, not described as
independent execution or direct pinned-source inspection. Hardware/GUI/SSH
rehearsal remains outside the local implementation scope.
