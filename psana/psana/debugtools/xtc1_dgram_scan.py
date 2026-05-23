#!/usr/bin/env python
"""Print a compact listing of XTC1 datagram headers."""

import argparse
import os
import struct
import sys
from pathlib import Path
from typing import Optional, Sequence

try:
    from psana.xtc1dgramlite import Xtc1DgramLite
except ImportError:
    # Useful for phase-1 development after building only this extension:
    #   PYTHONPATH=$PWD/builddir/psana python psana/psana/debugtools/xtc1_dgram_scan.py ...
    from xtc1dgramlite import Xtc1DgramLite

XTC1_DGRAM_HEADER_SIZE = 40

XTC1_SERVICE_NAMES = {
    0: "Unknown",
    1: "Reset",
    2: "Map",
    3: "Unmap",
    4: "Configure",
    5: "Unconfigure",
    6: "BeginRun",
    7: "EndRun",
    8: "BeginCalibCycle",
    9: "EndCalibCycle",
    10: "Enable",
    11: "Disable",
    12: "L1Accept",
}

PSANA2_SERVICE_NAMES = {
    0: "ClearReadout",
    1: "Reset",
    2: "Configure",
    3: "Unconfigure",
    4: "BeginRun",
    5: "EndRun",
    6: "BeginStep",
    7: "EndStep",
    8: "Enable",
    9: "Disable",
    10: "SlowUpdate",
    11: "L1Accept_EndOfBatch",
    12: "L1Accept",
    13: "NumberOf",
}


def _positive_int(value: str) -> int:
    intval = int(value)
    if intval <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer.")
    return intval


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan XTC1 datagram headers and print offsets, sizes, services, and root XTC metadata."
    )
    parser.add_argument("xtc_file", help="Path to an XTC1 .xtc file.")
    parser.add_argument("-n", "--limit", type=_positive_int, default=20, help="Number of datagrams to print.")
    parser.add_argument(
        "--raw-words",
        action="store_true",
        help="Also print the first 10 little-endian uint32 words from each datagram header.",
    )
    return parser.parse_args(argv)


def _format_timestamp(timestamp: int) -> str:
    sec = (timestamp >> 32) & 0xFFFFFFFF
    nsec = timestamp & 0xFFFFFFFF
    return f"{sec}.{nsec:09d}"


def _scan(path: Path, limit: int, raw_words: bool) -> int:
    offset = 0
    count = 0
    file_size = path.stat().st_size

    with path.open("rb") as f:
        while count < limit:
            f.seek(offset)
            header = f.read(XTC1_DGRAM_HEADER_SIZE)
            if not header:
                break
            if len(header) < XTC1_DGRAM_HEADER_SIZE:
                print(f"truncated header at offset={offset}: got {len(header)} bytes", file=sys.stderr)
                return 1

            dgram = Xtc1DgramLite(header)
            if dgram.size <= 0:
                print(f"invalid datagram size at offset={offset}: {dgram.size}", file=sys.stderr)
                return 1
            if offset + dgram.size > file_size:
                print(
                    f"datagram at offset={offset} overruns file: size={dgram.size} file_size={file_size}",
                    file=sys.stderr,
                )
                return 1

            raw_name = XTC1_SERVICE_NAMES.get(dgram.raw_service, f"Unknown({dgram.raw_service})")
            mapped_name = PSANA2_SERVICE_NAMES.get(dgram.service, f"Unknown({dgram.service})")
            print(
                "idx={idx:6d} offset={offset:12d} size={size:9d} "
                "ts={ts} raw_service={raw:2d}:{raw_name:<16s} "
                "service={svc:2d}:{mapped_name:<16s} "
                "env=0x{env:08x} damage=0x{damage:08x} "
                "src=(0x{src_log:08x},0x{src_phy:08x}) "
                "contains=0x{contains:08x} extent={extent}".format(
                    idx=count,
                    offset=offset,
                    size=dgram.size,
                    ts=_format_timestamp(dgram.timestamp),
                    raw=dgram.raw_service,
                    raw_name=raw_name,
                    svc=dgram.service,
                    mapped_name=mapped_name,
                    env=dgram.env,
                    damage=dgram.damage,
                    src_log=dgram.src_log,
                    src_phy=dgram.src_phy,
                    contains=dgram.contains,
                    extent=dgram.extent,
                )
            )
            if raw_words:
                words = struct.unpack("<10I", header)
                print("  words=" + " ".join(f"0x{word:08x}" for word in words))

            offset += dgram.size
            count += 1

    print(f"scanned={count} next_offset={offset} file_size={file_size}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    path = Path(args.xtc_file)
    if not path.is_file():
        print(f"not a file: {path}", file=sys.stderr)
        return 1
    return _scan(path, args.limit, args.raw_words)


if __name__ == "__main__":
    raise SystemExit(main())
