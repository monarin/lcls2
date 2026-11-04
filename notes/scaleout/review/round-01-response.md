# Round 1 response

All six findings accepted and addressed in the revised plan; closure awaits
independent re-review.

| ID | Response | Changed design |
| --- | --- | --- |
| S1 | Fixed | Active stripe/OST capacity grows with N: five streams, N*s0 stripes per stream, N*P1 distinct OSTs; inventory all OST/OSS mappings, reject accidental overlaps, verify active-range coverage, same files for replay/application. Explicit capacity and chunking gates. |
| S2 | Fixed | B_pattern now requires its own exact-replay concurrency plateau; coalescing is diagnostic. Lower-bound references cannot support saturation claims. |
| S3 | Fixed | Require OSC/RPC or validated wire-level counters; llite is application-level only. Expected bytes include separately recorded bigdata, SMD and preflight-fixed readahead/page allowances. |
| S4 | Fixed | Large D_node-sized data is a prerequisite for sustained B-E, not only weak scaling. Valid XTC needed for C-E; short pilots cannot pass saturation gates. |
| S5 | Fixed | Replace small-sample percentile bootstrap with predeclared paired log-ratio Student-t intervals, independent balanced blocks, assumptions/drift checks and leave-one-block-out sensitivity. Define weak-scaling pairing explicitly. |
| S6 | Fixed | Add separately accounted PS_EB_NODE_LOCAL topology and require cache checks on every range each reader node may access under dynamic scheduling. |
