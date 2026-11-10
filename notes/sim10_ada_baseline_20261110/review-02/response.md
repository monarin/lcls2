# Response to review 02

Pre-launch code readiness passed on58894c331; no blocking findings.

- ADA-08: actual psana/meson.build diff saved in build-config-diff.json. Only
  addition is gpu/notes in the Python install_subdir exclude_directories list;
  no extension source lists, flags, includes or links change. Before sbatch,
  submission helper asserted the PR file equals frozen text plus exactly that
  one insertion. Config mismatch is therefore explicitly checked for this
  pinned attempt. Moving that allow-list into reusable prepare_runtime/audit
  code is deferred; acceptance review must inspect the saved diff/launch receipt.
- ADA-09: actual receipt source path was asserted equal to resolve() before
  submission, so this attempt does not exercise the symlink-path limitation.
  Generic symlink support is deferred; no measured-runtime changes made.

Review pass is code readiness only. GPU tests/performance and evidence review
remain pending. Submission helper first failed before sbatch under system
Python3.6 (text= unsupported); retried with Python3.11, producing exactly one
job,44185349. No benchmark attempt ran in that helper failure.
