# UED detector connectivity — 2026-09-28 16:14 PDT

> Historical September 2026 record. For current repeat-test entry points and archived evidence, see [the repeat-test guide](../epixquad/README.md). Process IDs, running-state statements and launch paths below describe that session.

Read-only probe on `drp-ued-cmp003`, `/dev/datadev_0`, lane 0 succeeded.
Used the private test submodule bundle and DAQ Python/Rogue 6.1.3.

- PCIe: XilinxKcu1500, `Lcls2XilinxKcu1500Pgp4_10Gbps`, firmware v4.2.0.
- PGP4: RX, TX and remote ready; RX/TX clocks 156.251 MHz; reported link-down
  and link-error counters both zero at the time of the read.
- Camera firmware and all four carrier IDs read successfully.
- ADC done=true, failed=false; ASIC mask `0xffff`.
- Existing trigger state: TrigEn=true, AutoTrigEn=true, TrigSrcSel=3.

Before probing, inspected the running `epixquad_monitor_ioc` process (PID
238794, uedopr, started September 25). Its arguments select this device/lane
and `--no-microblaze`; its default monitoring stream is VC3. The probe opened
only camera register VC1, which succeeded. The IOC was not stopped or changed.
The host reports Rocky Linux 9.6 and DMA driver 7.4.0/API 6.

The probe uses generic PyRogue roots with polling and initialization reads/
writes disabled. It reads PCIe identity and PGP link status before opening
SRPv3 VC1, and never starts DevRoot or ePixQuad.Top. No YAML load, register
write, reset, training, PROM operation, timing configuration, or capture was
performed. This proves register connectivity, not image acquisition or the
production profile's remaining timing/configuration choices.

Full output: [probe.log](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-connectivity-20260928/probe.log). Structured readings and script checksum:
[result.json](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-connectivity-20260928/result.json). Reusable script:
[probe_connection.py](../../psdaq/psdaq/debugtools/epixquad1kfps/probe_connection.py).

Executed on the detector host with the selected test environment:

```bash
python psdaq/psdaq/debugtools/epixquad1kfps/probe_connection.py \
  --dev /dev/datadev_0 --lane 0
```

Reviewed source and startup paths, checked CLI parsing, and completed the live
read-only probe successfully. The UED capture profile remains unvalidated.
