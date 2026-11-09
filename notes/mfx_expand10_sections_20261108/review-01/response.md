# Response to independent review

Verdict: pass on executable SHA 27975c173a08bd69ba6b03c87c70cc31fcc8914f.
All four findings are non-blocking. Closure changes are documentation/evidence
only; executable scripts, runtime and kernels are unchanged, so no new GPU run
or executable re-review is warranted.

| ID | Response |
|---|---|
| SEC-01 | Fixed: README now uses 0.024 s consistently with cpu-table.md. |
| SEC-02 | Accepted the reviewer's documentation alternative: README requires normal Python with assertions enabled, prohibiting -O/-OO/PYTHONOPTIMIZE. Recorded analysis used no optimization. Explicit require() gates remain a potential robustness improvement if this scoped analysis tool is generalized. |
| SEC-03 | Fixed: documented per-call overhead inside measured spans, with observed NVTX-minus-perf_counter differences (~0.04/0.07 s for one-stream wait/submit). Profiler overhead is not confined to the uninstrumented remainder. |
| SEC-04 | Fixed: explicitly state no kernel-kernel concurrency in either measured steady envelope, including two streams. Limit claim to the envelopes, because the analysis does not test all setup/warmup kernels. This does not rule out copy/compute overlap or framework concurrency. |

Reviewer source/evidence limitations are preserved verbatim in review.json.
Coordinator generated native timing exports, ran five interval tests, verified
source hashes and native acceptance, and queried scheduler completion; reviewer
could not independently execute those operations with its restricted file tools.
