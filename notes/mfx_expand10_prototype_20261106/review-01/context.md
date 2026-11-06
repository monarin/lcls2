# Standalone GpuTask prototype review

Review the exact commit supplied by the helper; base 5153ab21c. Scope is
notes/mfx_expand10_prototype_20261106/scope.md. The user corrected sequencing:
a small runnable GpuTask expansion/calibration/cube prototype comes before large
dataset generation. Review prototype.py, run.sbatch, README.md, scope/status,
smoke-report.json, evidence.json, and the proposal's sequencing update.

Cube and sample10 kernels are reused unchanged from reviewed Stage 3A/Stage 2.
Inspect those dependencies as needed, but do not expand this entrypoint review
into a framework redesign. Current GpuTask is a callable plus inputs/calibconst;
managed factory/lifecycle remains proposed. Application explicitly calls finish
only after full iteration and close on success/abort. One process, two CUDA
streams, one A100, batch size 7. Banks are distinct per stream and count every
accepted event once; deterministic ordinal%29 bin selection with one empty bin.
Frozen constants loader is scoped to serial Run construction, same as acceptance.
No new XTC, calibration DB writes, public deployment, core API or kernel changes.

Check executable wiring, runtime/fixture/code provenance, failure cleanup,
claims against smoke evidence, and whether README provides a runnable recipe.
New smoke acceptance: all 1,000 native events ordered, seed errors zero, exact bin
counts, finite final sums and an empty bin, one export, saved final arrays.
Prior Stage 3A full CPU pixelwise checks are inherited evidence, not rerun by this
smoke entrypoint. No MPI, actual GDS, native multi-step or 35kHz claim. Runtime
KvikIO compatibility ON; framework input budget excludes user persistent arrays.
Stage 3A low nonblocking F-008 (third distinct stream rejection test) remains
deferred; this fixed two-stream entrypoint does not broaden supported settings.

Coordinator-generated full report and artifact/code hashes are committed in the
prototype notes; original scratch paths are provenance only outside the read
sandbox. Inspect files from this checkout. Do not claim to have executed tests
or hashes, do not delegate, and do not launch another review loop. Return exact
reviewed SHA and stable findings. Maximum three rounds.
