# Pilot validation

No hardware execution. No root constructed using real dependencies. No production process/job control, firmware changes or installed-source edits.

- Initial 13 offline tests passed on SDF Python 3.6.
- Expanded 15 offline tests passed on the pinned environment's Python 3.9.21 (3.855 s). Includes cold pair PIDs, warm reuse/deallocation, pair abort, init failure, watchdog, stuck cleanup, shutdown errors, Ctrl-C reaping, default dry-run, explicit flags, wrong host, output collision, bounded capture/drain, and instrumented versus original receiver parsing.
- Default CLI preview for paired warm trials returned dry_run=true without hardware imports.
- Import-only check using the pinned setup_env_daq.sh succeeded on SDF. It imported psdaq.configdb.piranha4_config, Rogue and dependency classes without constructing root/stream/device objects. Rogue version v6.1.3. Config, gateway, UART and TX module hashes matched constants in pilot.py; installed config path was release/install/lib/python3.9/site-packages/psdaq/configdb/piranha4_config.py.
- The script uses absolute-path invocation, so the development checkout does not replace the installed psdaq module. Only stdlib is imported by the parent.

Limitations: no real camera pilot yet; tests use fake hardware/barriers. Parser fidelity is checked with the original receiver definition but callback timing is not. Preflight cannot establish absence of invisible /proc holders. Operator attestation and full board exclusivity are mandatory. Capture instrumentation changes timing; warm mode does not Configure/acquire between Alloc calls. Real cleanup and endpoint release still need operator readback.

After round-1 review: 17 offline tests passed on Python 3.9.21 (see tests-round-02.log), including unkillable mocked peers and target timeout combined with hanging cleanup. These verify bounded signaling/reaping, survivor reporting, restored signal handlers and pre-cleanup raw capture preservation. Production child logging is now unbuffered. No hardware was run for these changes.
