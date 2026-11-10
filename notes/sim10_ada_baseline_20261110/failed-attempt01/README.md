# sdfada011 CUDA initialization failure

Benchmark44185349 failed in5seconds, before any GPU tests executed or input
staging, at CuPy getDeviceCount():cudaErrorUnknown. Empty stage removed.
Diagnostic44185635 uses standalone CuPy (no psana) and reproduces the failure
with no CUDA mask and separately for all nine nvidia-smi-visible GPU UUIDs.
Slurm assigned/advertised10 devices but nvidia-smi reports9. Driver575.57.08,
CuPy14.0.1, runtime/driverAPI12090. This points to a node CUDA problem; exact
hardware/driver root cause is not established. No performance result accepted.
Unchanged reviewed benchmark resubmitted as44185752, excluding sdfada011.
