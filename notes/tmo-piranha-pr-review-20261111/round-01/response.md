# Response to round 1

PR-001 accepted and fixed. Worker exit/EOF observed after the shared deadline
now leads to timeout release and existing bounded cleanup verification. Earlier
exit/EOF still fails inspection. A valid release read after expiry is classified
as timeout. Missing exit records, failed cleanup, cancellation and unacceptable
return codes remain failures through cleanup_succeeded.

The regression pauses the parent after the loop time check, either before poll
or inside receive, until real fake workers have independently exited. It covers
successful rehearsal and target-error holds, plus watchdog-killed cleanup hangs.
All six cases failed on the old implementation (expiry-before-tests.txt.gz).
The updated full-suite result will be recorded in round-02/tests.txt.
