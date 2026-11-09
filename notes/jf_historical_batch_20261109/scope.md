# Historical configuration and larger sim10 batches

User-authorized cold Weka FFB psana-GPU read/staging benchmark. One node,
one BD, one A100, 10,000 events per measured case. Compare raw dense batch20,
sim10 descriptor batch20, and sim10 descriptor batch100; repeat in reverse order.
Historical settings: one GPU execution stream, eight KvikIO workers, 1MiB tasks,
16MiB bounce buffer, compatibility ON, automatic direct I/O OFF, 6GiB framework
budget, SMD1000, full available physical-core affinity shared by all three ranks.
Raw bulk target1MiB; sim10 retains user-requested512KiB.

Use the same public GpuTask harness for both inputs. Raw requests production dense
gather; sim10 requests seed field descriptors. This retains the historical
settings but differs from the older raw benchmark's private input adapter.
No user compute, expansion, calibration, cube, or timed payload D2H.
One 200-event pixel-exact raw preflight and one sim10 batch100 preflight warm the
compiler cache. All cases evict owned FFB input pages AFTER setup and require
zero per-file mincore residency. Server cache is uncontrolled. No input NVMe stage.

Verify exact timestamp identity (independently frozen for both fixtures), five
reads/event, exact bytes, no CPU payload reads/admission drains, resources drained,
actual callback batch sizes, physical NIC deltas and idle traffic, runtime/code
hashes, device/host/affinity, FFB mounts. Record final cache occupancy; avoid claiming
an isolated input-size effect when raw dense gathering and sources differ.

Keep existing benchmark files unchanged. Store native logs/results under project
scratch; remove only this campaign's temporary sim10 FFB copy. Review with Claude
in a detached commit worktree, max three rounds. No production framework edits.
