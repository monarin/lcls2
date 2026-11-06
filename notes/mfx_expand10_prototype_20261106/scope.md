# Runnable GpuTask prototype

User corrected the next stage: provide a simple runnable blow-up + calibration +
cube GpuTask prototype before preparing a large dataset. Reuse the reviewed Stage
3A Cube and Stage 2 expansion/calibration unchanged. Add a short entrypoint that
uses the existing 1,000-event fixture, consumes native run.events(), checks scalar
input errors, then explicitly exports the final cube/counts. Remove the expensive
CPU-reference pass from normal prototype usage; existing full numerical validation
remains Stage 3A evidence. Do not claim this smoke run revalidates every cube pixel.

No new kernels/core API/data/DB changes, no sustained-throughput claim, no large
fixture generation. The earlier read-only capacity probe and repetition draft are
deferred; the storage question is withdrawn. Use one A100, existing runtime and
private scratch; source commit base 5153ab21c. Review the entrypoint and full smoke
report with implement-review-loop, maximum three rounds. Preserve old results.
