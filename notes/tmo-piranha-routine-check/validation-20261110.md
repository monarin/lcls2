# Branch consolidation validation — November 10, 2026

- Base: fetched origin/master `25e6f4d6535b7b5b24e9cf0470ada5e9273b2ed0`.
  Fork master already matched this SHA. The new branch contains that base.
- Eight Piranha commits transplanted cleanly; unrelated old-base work excluded.
  The original-to-new mapping is in `commit-map.json`.
- Pilot source remains exactly the reviewed SHA256
  `d7c7e249753f29dccf5d713365b79758f8fb60da42fa4e336b5728d2ea10b804`.
- Hold-plan text exactly matches the independently reviewed original snapshot.
  Historical review records retain their original SHAs; no new implementation
  review is claimed for the consolidation.
- All 17 offline lifecycle/capture tests passed using the deployed Python 3.9.21
  interpreter against this worktree, without Rogue imports or hardware access.
  Output: `offline-tests-20261110.txt`.
- Verified 284 copied files against original and stored SHA256 values. The 285th
  imported file is the pilot README, intentionally updated to point to the
  current hardware results and hold plan. The manifest describes the import
  snapshot, not the subsequently updated README.
- Original process logs round-trip byte-for-byte through gzip. Other artifacts
  remain uncompressed. Recorded summaries contain 31 successful camera calls.
- Relative links in the overview and pilot README resolve.
- Reviewed the scoped source/evidence for credential patterns before publication;
  no matches were found. Historical host paths and camera serials are intentional
  experiment provenance.
- Main checkout branch and its tracked files were not changed. No hardware run,
  GUI attachment, production update, or hold-on-error implementation occurred.
