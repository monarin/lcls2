# Real MFX codec feasibility gate

Use the unchanged SLEEK float32 compressor wrapper from lcls2 master
`9f7391c091d21afe585165d5ee72ca057ed98e4a` and CPU/GPU decoders from its
SLEEK wrap revision `00ff130d8d3182204b6b63fc22d5c85ad6b5f2e1`.
Start with one complete MFX run-10 event, grouped into the five original DRPs.
Scale to approximately 1,000 events only after the round-trip gate passes.

Acceptance: frozen calibration bytes verified; GPU calibration matches CPU v3
per pixel; both decoders recover the expected float32 shape and stay within
absolute error 3.0 of the compressor input; CPU and GPU decoded bits agree.
Use offset, disable masks and common mode explicitly for this arithmetic test.
This is not a scientific validation of those settings or of error bound 3.0.
Retain raw references and original timestamps; compressed calibrated values
must never be calibrated a second time.

Keep generated artifacts in private scratch. No public XTC publication, new
calibration deployment, or changes to existing runs. Run assignment is pending.
No production DRP or codec fixes are included in this initial feasibility gate.
If the MFX sample gate fails, retain the reproducer and stop scaling. Synthetic XTC schema/reader integration follows a
successful codec gate and is not claimed by a standalone codec test.

An additional boundary-value probe failed after the MFX gate passed and the
1,000-event trial started. Complete that already-running, bounded sample test
and retain the separate failure. This does not waive the MFX pixel checks or
qualify the decoder for general use; XTC publication remains excluded. Diagnose
the failure without patching the upstream codec in this scope.

Use the existing work/synthetic-mfx-scan branch. Codex implements and verifies;
Claude independently reviews the exact commit, with up to three review rounds.
