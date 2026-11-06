# Hypercube projection review closure

Claude passed model/plot commit `9d555787df1f632d647019b7cc95346babfe6c17`
with no blocking findings. No model code, figures or numerical outputs changed
after review; this closure records evidence and documentation clarifications.

| Finding | Response |
|---|---|
| HC-001 | Accepted and addressed. README explicitly labels persistent state as calculated from Cube's allocation formula, and the pool peak as measured. It links both sources. |
| HC-002 | Accepted and addressed. Reproduction command explicitly sets MPLCONFIGDIR to scratch. Traffic GB/TB are explicitly decimal while capacity MiB/GiB are binary. |

The note also explicitly states that projections exceed the current prototype's
30-bin limit and do not implement a raised limit or new allocation policy.

Coordinator verified model arithmetic and integer bounds, inspected PNG pixels,
and compared artifact/code hashes against Git blobs. Reviewer independently
checked the source/numbers and SVG labels but did not run code or verify hashes
or image pixels; limitations remain in review.json.

One incidental arithmetic statement in the review summary/checks needs correction:
the first-over-budget excess is 4,096 bytes for the 40 GiB cases and **9,216 bytes
for the 80 GiB cases**, not 4,096 in every case. The model's capacities and results
are unchanged and correct. Coordinator computed the excess for all eight cases
in commit-provenance.json. This is a correction to review prose, not a disputed
finding or a change to the approved model.
