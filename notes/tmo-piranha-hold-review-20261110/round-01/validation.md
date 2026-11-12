# Coordinator validation

- Reviewed documentation commit: a3d454c63a4616f6d1b3e8cbac3a6251fbf6ffa8.
- `git diff --check BASE REVIEW_SHA` passed. Changes are only the plan and scope.
- Commit hooks passed (whitespace, end-of-file, conflict markers, mixed endings,
  added-file size; language checks skipped because no code changed).
- Pilot unchanged: SHA256 d7c7e249753f29dccf5d713365b79758f8fb60da42fa4e336b5728d2ea10b804.
- Local and deployed piranha4_config.py both hash to
  9efaf0c9b206e9823e6ba3751c7b0ee3cdc68b8fb3849889fabc5cf404adcca6.
- Read actual child exception/finally path, parent wait/cleanup path, successful
  child shutdown-before-result path, and barrier shutdown implementation.
- Read deployed root ZMQ startup, devGui arguments/new-root construction,
  PyRogue GUI CLI, SimpleClient value/get and VirtualClient implementation.
- Read `_ZmqServer.py`: base port is `port()`, three ports are logged, wildcard
  binding's `.address` resolves to loopback and needs host-aware instructions.
- No runtime tests, hardware opens, network clients, or production changes.
  This supports only source-level plan claims, not implementation behavior.
