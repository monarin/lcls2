# Round 1 responses

Reviewed commit: 57495428124d135e67d3651a1351d3a0e807b74e. Verdict: pass.
Attempt 1 timed out without a verdict; attempt 2 reviewed the same commit.

| Finding | Response |
| --- | --- |
| XPP-S2-01 | Fixed: binary uploads set application/octet-stream explicitly. |
| XPP-S2-02 | Fixed: persist exact body privately and status/safe headers before checking HTTP status or parsing JSON; reject redirects. |
| XPP-S2-03 | Fixed: regenerate Kerberos headers for each POST. |
| XPP-S2-04 | Fixed: derive completion counts from checked reports; verify request evidence counts against journal. |
| XPP-S2-05 | Accepted without code change: fresh destination time/metadata are intentional; original documents and source IDs are frozen and linked in extpars. Pinned runtime Run._get_calibconst supplies run/exp without time_sec, so run 3–4 selection is the relevant constraint. Calibrated DataSource remains a required post-deployment check. |
| XPP-S2-06 | Fixed: mocked actual deploy loop covers success, data-ID substitution, durable intent-before-POST, HTTP/JSON/redirect failure after a blob, stop and refusal to retry. README documents read-only orphan reconciliation. |
| XPP-S2-07 | Fixed: hash baseline snapshot against the pinned Stage 1 admin-check digest. |

A fresh run-slot check now runs immediately before deployment as well as during
prepare. Reprepared plan supersedes the old plan retained here. No deployment
has occurred at this response point. Writer/byte-verifier/acceptance code and
existing private data remain unchanged.
