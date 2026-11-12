# Round 1 response

H1 accepted/fixed: signal all survivors first; bound post-kill waiting to a shared deadline of at most two seconds; record forced/unreaped PIDs and errors; preserve final failed summary and restore handlers in finally. Normal group teardown is not repeated by abort cleanup. Added mocked unkillable peers test checking every peer is signaled, parent finishes promptly, summary identifies survivors and signal handlers are restored.

H2 accepted/fixed: save uart-failure.json before hardware/barrier cleanup; record snapshot-write error in failure.json; retain final snapshot separately. Successful-trial capture is also saved before deallocation. Added GCP failure plus hanging root.stop case verifying pre-cleanup raw RX survives forced termination.

Also made production child output unbuffered for logs that survive abrupt termination. README describes forced/unreaped outcomes and operator intervention. No hardware run. Tests rerun after changes.
