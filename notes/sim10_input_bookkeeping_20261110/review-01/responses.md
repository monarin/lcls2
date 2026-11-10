Round1 on95e6c3788: changes_requested because native results were pending; no
blocking implementation defect. Responding to all suggestions:

- IBK-R1-01: pending. Attempt01 stopped correctly at GPU gate:30passed,1failed.
  Failure expects BeginStep constant refresh, explicitly unsupported by frozen
  _dispatch_transition. Preserve original assertion as baseline_test, require
  reproduction on unchanged install-B before accepting corrected new GPU tests.
  Standalone diagnostic61098227 failed before tests because its launcher erased
  srun's libfabric library setup; use the already working campaign launcher.
- IBK-R1-02: added raw release callback failure test; group, candidate and busy
  slot survive and a later poll reclaims exactly once.
- IBK-R1-03: documented partial batch ownership and outstanding-read guard at
  controller call site; existing controller partial I/O tests remain passing.
- IBK-R1-04: register notification before assigning state.window; added failed
  duplicate-registration test proving group can bind a valid window afterward.
- IBK-R1-05: inventory now records only wrapped methods and their actual labels.

Full CPU suite after fixes:970passed in8.77s. No calibration runtime behavior
changed. Corrected GPU transition assertion preserves consumer order and pixel
checks, asserts stable constant identity/value and step generation increment.
Native baseline reproduction and new acceptance remain required, never inferred.
