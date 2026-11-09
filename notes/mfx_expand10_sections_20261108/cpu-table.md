| Host wall-time section | 1 stream | 2 streams |
|---|---:|---:|
| Wait for GPU input completion | 6.299 s (18.5%) | 10.028 s (29.2%) |
| Wait to reuse execution slot | 5.945 s (17.5%) | 0.024 s (0.1%) |
| CPU ancillary reads + event construction | 5.788 s (17.0%) | 6.402 s (18.6%) |
| SMD/EventBuilder | 1.232 s (3.6%) | 1.275 s (3.7%) |
| Submit KvikIO reads | 2.020 s (5.9%) | 2.066 s (6.0%) |
| XTC parser preparation/enqueue | 1.993 s (5.9%) | 2.395 s (7.0%) |
| Field descriptors: build/upload call | 1.987 s (5.8%) | 2.113 s (6.1%) |
| Other instrumented pipeline work | 2.153 s (6.3%) | 2.623 s (7.6%) |
| Outside instrumented sections | 6.595 s (19.4%) | 7.433 s (21.6%) |
