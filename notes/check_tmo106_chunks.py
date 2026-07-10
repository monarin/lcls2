#!/usr/bin/env python3
import argparse
import glob
import os
import struct
import sys
import time


XTC_DIR = "/sdf/data/lcls/ds/tmo/tmol1030922/xtc"
PY_SITE = "/sdf/home/m/monarin/lcls2/install/lib/python3.9/site-packages"


def count_smd_file(path):
    counts = {}
    n_dgrams = 0
    with open(path, "rb", buffering=1024 * 1024) as f:
        while True:
            hdr = f.read(24)
            if not hdr:
                break
            if len(hdr) != 24:
                raise RuntimeError(f"short dgram header in {path}: {len(hdr)} bytes")
            _ts_low, _ts_high, env, _src, _damage, _typeid, extent = struct.unpack(
                "<III I H H I", hdr
            )
            if extent < 12:
                raise RuntimeError(f"bad xtc extent {extent} in {path}")
            service = (env >> 24) & 0xF
            counts[service] = counts.get(service, 0) + 1
            n_dgrams += 1
            f.seek(extent - 12, os.SEEK_CUR)
    return n_dgrams, counts


def count_smd(args):
    pattern = os.path.join(
        args.xtc_dir,
        "smalldata",
        f"tmol1030922-r{args.run:04d}-s*-c000.smd.xtc2",
    )
    files = sorted(glob.glob(pattern))
    if args.streams:
        wanted = {int(s) for s in args.streams.split(",")}
        files = [
            path
            for path in files
            if int(os.path.basename(path).split("-s")[1].split("-")[0]) in wanted
        ]
    if not files:
        raise RuntimeError(f"no smd files matched {pattern}")

    rows = []
    for path in files:
        n_dgrams, counts = count_smd_file(path)
        l1 = counts.get(12, 0)
        eob = counts.get(11, 0)
        rows.append((path, n_dgrams, l1, eob, counts))

    print("SMD_COUNTS_BEGIN", flush=True)
    for path, n_dgrams, l1, eob, counts in rows:
        print(
            f"SMD_FILE {path} dgrams={n_dgrams} l1accept={l1} "
            f"l1_endofbatch={eob} event_total={l1 + eob} services={counts}",
            flush=True,
        )
    event_totals = [l1 + eob for _path, _n, l1, eob, _counts in rows]
    print(
        f"SMD_SUM files={len(rows)} min={min(event_totals)} max={max(event_totals)} "
        f"all_equal={len(set(event_totals)) == 1} expected={event_totals[0]}",
        flush=True,
    )
    print("SMD_COUNTS_END", flush=True)


def count_datasource(args):
    sys.path.insert(0, PY_SITE)
    from mpi4py import MPI
    from psana import DataSource

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    host = os.uname().nodename

    kwargs = {
        "exp": "tmol1030922",
        "run": args.run,
        "dir": args.xtc_dir,
        "batch_size": args.batch_size,
    }
    if args.max_events:
        kwargs["max_events"] = args.max_events
    if args.detectors:
        kwargs["detectors"] = args.detectors.split(",")
    if args.small_xtc:
        kwargs["small_xtc"] = args.small_xtc.split(",")

    t0 = time.time()
    local = 0
    ds = DataSource(**kwargs)
    for run in ds.runs():
        for _evt in run.events():
            local += 1
            if args.progress and local % args.progress == 0:
                print(
                    f"RANK_PROGRESS rank={rank} host={host} local={local} "
                    f"elapsed={time.time() - t0:.1f}",
                    flush=True,
                )
    total = comm.reduce(local, op=MPI.SUM, root=0)
    all_counts = comm.gather(local, root=0)
    if rank == 0:
        print(
            f"DATASOURCE_TOTAL ranks={size} total={total} per_rank={all_counts} "
            f"elapsed={time.time() - t0:.1f} kwargs={kwargs}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smd", "datasource"), required=True)
    parser.add_argument("--run", type=int, default=106)
    parser.add_argument("--xtc-dir", default=XTC_DIR)
    parser.add_argument("--streams", default="")
    parser.add_argument("--detectors", default="")
    parser.add_argument("--small-xtc", default="")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--progress", type=int, default=0)
    args = parser.parse_args()

    if args.mode == "smd":
        count_smd(args)
    else:
        count_datasource(args)


if __name__ == "__main__":
    main()
