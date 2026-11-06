# Hypercube memory projection

User requests projection plots for full-image JF hypercubes with user-selected
bin dimensions and discussion of handling GPU memory exhaustion. Produce a
reproducible analytic model/plots and a design note, not runtime modifications.
Raw detector is 32*512*1024 uint16 = 32 MiB. Model float32 sums plus scalar uint64
counts, product of bin dimensions, independent replica count per physical GPU,
optional float64 sums, and ideal balanced bin ownership across GPUs. State a
fixed illustrative 8 GiB non-cube reserve for 40/80 GiB device budgets; not a
measurement or guaranteed hardware capacity. Separate input compression from
output accumulator memory. Explain sparse/cache/partial export, single-device
owner accumulation, routing, skew, output memory and precision tradeoffs.

Use existing psana2-gpu memory/stream contracts; all new policies proposed.
No new dataset, DB, GPU jobs or core code changes. Plot with standard Python
matplotlib, inspect generated figures, verify numeric examples. Independently
review the model/design/claims with Claude (max three rounds), commit/push only
this scope on work/synthetic-mfx-scan. Existing deferred findings are unrelated
and remain deferred. Base 3c025ce18.
