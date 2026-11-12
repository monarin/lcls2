# Verification of the proposed direct Python path — 2026-11-06

**Result: exact software failure path demonstrated offline; spontaneous hardware reproduction remains unverified.** The standalone hardware harness remains proposed, not implemented or run. This supplemental verification does not change the reviewed plan or claim hardware validation.

Executed `python3 notes/tmo-piranha-review-20261106/verification/offline_path_check.py` on SDF. The script compiles selected unmodified AST definitions from the checkout's `piranha4_config.py`, retaining source line numbers: receiver, init, init_feb, connectionInfo, connectionShutdown, setup_timing and dumpTiming. It also uses the actual supervisor_info function. All gateway construction/startup, register access, UART transport, stream connection and barriers are explicit fakes. There are no Rogue/PyRogue imports, sockets or hardware opens. Logging output is suppressed; parser and real five-second timeout are unchanged. Results, tracebacks, fake operation traces and injected frame bytes are in `offline_results.json`; its SHA256 identifies the source used.

| Scenario | Observed outcome |
|---|---|
| Valid startup and allocation replies, single-process supervisor | Returned model/serial/BiST after GCP, VT, VV |
| Valid startup, no allocation GCP reply, supervisor | Original exception at connectionInfo line 286, receiver line 61 |
| Valid startup, allocation reply contains only USER> | Same timeout; `_resp` empty despite received bytes |
| Valid startup, full allocation reply with NUL-prefixed prompt | Same timeout, nonempty response list |
| Valid startup/allocation replies, worker role | Returned success |
| Valid startup, no allocation reply, worker role | Same exception at line 286 |

All six assertions passed. Failure cases took approximately five seconds in the original `_await`; full supervisor cases include another roughly 1.3 seconds of the original setup sleeps. Worker role election was checked against the real helper, but synchronization was faked: this is not a test of two-process barriers. The first script attempt failed before execution because its repository-root path was one directory too high; that local path was corrected before these results were produced.

The causal claim is deliberately limited: withholding a fake reply proves where the real function fails **if** a reply is absent. It does not demonstrate why the camera/transport becomes silent, whether standalone startup induces that state, or the real failure frequency. Fake startup supplies a complete GCP response synchronously; actual Rogue callback threading and timing have not been tested.

Additional source verification supports standalone feasibility:

- `psdaq/drp/BEBDetector.cc:119–125,153–163` shows the C++ DRP calling these same Python functions; no C++ UART-send wrapper is necessary.
- Preserved `evidence/UartPiranha4.py:23–28,111–114` connects RX/TX directly to the serial stream and implements GCP by sending `gcp` on that stream.
- Preserved `evidence/ClinkDevRoot.py:54,128–172` opens VC0/VC2/VC3 by default (`0xD`); UART uses VC2 independently of the C++ image channel.
- `psdaq/drp/BEBDetector.cc:90` assigns C++ virtChan=1; `PGPDetector.cc:311` configures that channel's DMA mask. Its image-reader thread is launched in `PGPDrp::startup()` (`400–425`), not by the Python allocation call. C++ device open/mapping still precedes allocation (`DrpBase.cc:359–384`), so removing the C++ process is not a proven equivalent board/driver state.

Conclusion: there is no source-level requirement to start the DAQ controller/TEB/MEB to issue the same UART handshake and encounter its timeout. Whether the real issue is triggered without those processes remains an empirical question. It requires an explicitly authorized standalone hardware run with exclusive board access, fresh processes and phase-specific UART evidence. No production state was changed by this verification.
