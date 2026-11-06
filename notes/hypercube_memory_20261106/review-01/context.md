# Hypercube projection independent review

Review exact commit supplied by helper; base 3c025ce18. Scope: analytic memory
plots and proposed handling of user-selected Cartesian bin dimensions for the
32 MiB uint16 (32,512,1024) JF. Read scope.md, projection.py, projection.json,
README.md, validation.json, examples.csv, status.json in
notes/hypercube_memory_20261106. SVG contains the generated plot labels as text
if needed; no external scratch access necessary. All artifacts have local copies.
User requests memory projection and policy options, not implementation of those
policies. No runtime/GPU/dataset/DB changes. Current per-stream prototype and
persistent_task_state_xpp_cube.md are contextual dependencies.

Check units, float32/64 size distinction, multiplicative bin axes/labels and
per-physical-GPU bank replicas, exact count-inclusive integer thresholds,
assumption vs measurement separation (8 GiB reserve, 40/80 GiB illustrative
budgets), balanced-bin sharding minimum GPU counts, host output and export-rate
math, and practical routing/stream/buffer/partial-reducer hazards. Advice must
not claim current BD scheduling routes bins or that MPS/streams pool memory.
Kernel precision, repeated flush exactly-once addition, completion, live slot
ownership, backpressure, occupancy/cache thrashing, hot-bin skew and bounded host
output must be treated candidly. No performance or actual hardware OOM claim.

Coordinator executed script using recorded Stage2 Python runtime (matplotlib
3.11.2), verified hand values and integer boundaries, checked 1000/10000-bin GPU
counts, visually inspected PNG, and byte-compared scratch output to committed
artifacts. Reviewer can inspect source/records but should not claim it ran code,
hashes or viewed pixels if its tools cannot. All new policies remain proposed.
References to NVIDIA docs supply stream/event semantics and memory API caution;
no external read required for arithmetic review. Prior prototype deferred F-008
is unrelated and unchanged; start new finding IDs with HC-001.

Maximum three rounds. No delegation or recursive review loop. Return verdict,
exact reviewed SHA and stable findings. No unrelated implementation work.
