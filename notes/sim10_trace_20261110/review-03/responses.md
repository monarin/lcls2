Results review PASS on 3b1e9b3ac2d4e0d45c660be8097b1c9c3111c76b; no blocking findings.

- R1 accepted: README and status record the completed results review and exact SHA.
- R2 accepted: README states that 99.6% is a net share, with other phases netting +0.04 s despite offsetting changes of a few tenths of a second.
- R3 accepted: worker-thread attribution is explicitly inferred from per-read call counts; the API sum is across all recorded host threads and is not main-thread critical-path time.
- R4 accepted: retirement-path interpretation is marked as source inference, visited_states includes window-less entries, and equal 150,000 event-query totals are cited as supporting evidence.

Only documentation and review records changed after the reviewed results commit.
The harness remains ec0941a31; offline analysis remains identical to reviewed 3b1e9b3ac.
No additional runtime tests are warranted for these wording changes.
Prior nonblocking T3/T4 remain explicitly deferred. No unresolved blocking findings.
