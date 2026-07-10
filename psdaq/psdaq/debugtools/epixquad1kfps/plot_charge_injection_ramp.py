#!/usr/bin/env python3

"""Plot ePixQuad charge-injection pixel response versus decoded frame."""

import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from psdaq.configdb.epixquad_layout import (
    DAQ_RAW_FRAME_BYTES,
    RAW_SHAPE,
    rogue_payload_to_daq_raw,
)


def _parse_vc(value):
    if str(value).lower() in ("any", "all", "*"):
        return None
    return int(value, 0)


def _parse_pixel(value):
    parts = [part.strip() for part in str(value).split(",")]
    if len(parts) == 3:
        label = None
        coord_parts = parts
    elif len(parts) == 4:
        label = parts[0]
        coord_parts = parts[1:]
    else:
        raise argparse.ArgumentTypeError(
            "--pixel expects SEG,ROW,COL or LABEL,SEG,ROW,COL"
        )

    try:
        segment, row, col = (int(part, 0) for part in coord_parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--pixel contains a non-integer coordinate: {value!r}"
        ) from exc

    limits = RAW_SHAPE
    for name, coordinate, limit in zip(
        ("segment", "row", "col"), (segment, row, col), limits
    ):
        if not (0 <= coordinate < limit):
            raise argparse.ArgumentTypeError(
                f"pixel {name} out of range 0-{limit - 1}: {coordinate}"
            )

    if not label:
        label = f"s{segment}_r{row}_c{col}"
    return label, segment, row, col


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Decode an ePixQuad Rogue StreamWriter file and plot selected raw14 "
            "pixel values versus decoded frame for charge-injection testing."
        )
    )
    parser.add_argument("data_file", type=Path, help="Rogue StreamWriter .dat/.data file")
    parser.add_argument(
        "--pixel",
        action="append",
        type=_parse_pixel,
        default=[],
        metavar="[LABEL,]SEG,ROW,COL",
        help="raw DAQ pixel to plot; repeatable",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=500,
        help="maximum decoded frames; default 500",
    )
    parser.add_argument(
        "--fit-start",
        type=int,
        default=1,
        help="first decoded frame included in the fit, one-based; default 1",
    )
    parser.add_argument(
        "--fit-stop",
        type=int,
        default=None,
        help="last decoded frame included in the fit, inclusive; default last frame",
    )
    parser.add_argument(
        "--data-channel",
        type=int,
        default=1,
        help="Rogue file channel for ePixQuad image data; default 1",
    )
    parser.add_argument(
        "--vc",
        type=_parse_vc,
        default=None,
        help="optional payload byte0 low-nibble VC filter; default any",
    )
    parser.add_argument(
        "--bit-mask",
        type=lambda value: int(value, 0),
        default=0x7FFF,
        help="decoder bit mask; default 0x7fff",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output plot path; default DATA_FILE_charge_injection_ramp.png",
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        type=Path,
        default=None,
        help="optional CSV path for frame-by-frame raw14 and gainbit values",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="save the plot without opening an interactive window",
    )
    args = parser.parse_args()

    args.pixel = list(dict.fromkeys(args.pixel))
    if not args.pixel:
        parser.error("at least one --pixel is required")
    if args.max_frames <= 1:
        parser.error("--max-frames must be greater than 1")
    if args.fit_start < 1:
        parser.error("--fit-start must be at least 1")
    if args.fit_stop is not None and args.fit_stop < args.fit_start:
        parser.error("--fit-stop must be greater than or equal to --fit-start")
    if args.output is None:
        args.output = args.data_file.with_name(
            f"{args.data_file.stem}_charge_injection_ramp.png"
        )
    return args


def _load_file_reader():
    try:
        from psdaq.utils import enable_epix_quad1kfps  # noqa: F401
        from pyrogue.utilities.fileio import FileReader
    except Exception as exc:
        print("Failed to import Rogue/ePixQuad reader modules.", file=sys.stderr)
        print("Source setup_env.sh first.", file=sys.stderr)
        print(f"Import error: {exc!r}", file=sys.stderr)
        raise SystemExit(1) from exc
    return FileReader


def _payload_bytes(data):
    return np.asarray(data).view(np.uint8).tobytes()


def _decode_traces(args):
    FileReader = _load_file_reader()
    traces = {pixel: {"raw14": [], "gainbit": []} for pixel in args.pixel}
    decoded = 0
    decode_errors = 0

    reader = FileReader(str(args.data_file))
    for header, data in reader.records():
        if int(header.channel) != args.data_channel:
            continue
        payload = _payload_bytes(data)
        if len(payload) < DAQ_RAW_FRAME_BYTES:
            continue
        if args.vc is not None and (payload[0] & 0xF) != args.vc:
            continue

        try:
            raw, _ = rogue_payload_to_daq_raw(payload, bit_mask=args.bit_mask)
        except Exception as exc:
            decode_errors += 1
            if decode_errors <= 5:
                print(f"Decode error: {exc}", file=sys.stderr)
            continue

        for pixel in args.pixel:
            _, segment, row, col = pixel
            word = int(raw[segment, row, col])
            traces[pixel]["raw14"].append(word & 0x3FFF)
            traces[pixel]["gainbit"].append((word >> 14) & 0x1)
        decoded += 1
        if decoded >= args.max_frames:
            break

    if decoded < 2:
        raise RuntimeError(f"only {decoded} full image frame(s) decoded")
    return traces, decoded, decode_errors


def _fit_trace(frame, values, start, stop):
    fit_frame = frame[start:stop]
    fit_values = values[start:stop]
    if fit_frame.size < 2:
        raise ValueError("fit range must contain at least two decoded frames")

    slope, intercept = np.polyfit(fit_frame, fit_values, 1)
    prediction = slope * fit_frame + intercept
    residual = fit_values - prediction
    ss_res = float(np.sum(residual * residual))
    centered = fit_values - np.mean(fit_values)
    ss_tot = float(np.sum(centered * centered))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": r_squared,
        "prediction": prediction,
        "fit_frame": fit_frame,
    }


def _write_csv(path, frame, traces):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.writer(output)
        header = ["frame"]
        for label, _, _, _ in traces:
            header.extend((f"{label}_raw14", f"{label}_gainbit"))
        writer.writerow(header)
        for index, frame_number in enumerate(frame):
            row = [int(frame_number)]
            for values in traces.values():
                row.extend((values["raw14"][index], values["gainbit"][index]))
            writer.writerow(row)


def _plot(args, frame, traces, fits, fit_start, fit_stop):
    import matplotlib

    if args.no_show or not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    colors = plt.get_cmap("tab10").colors
    fig, (raw_axis, delta_axis) = plt.subplots(
        2, 1, figsize=(11, 8), sharex=True, constrained_layout=True
    )

    for index, (pixel, values) in enumerate(traces.items()):
        label, segment, row, col = pixel
        color = colors[index % len(colors)]
        raw14 = np.asarray(values["raw14"], dtype=np.float64)
        fit = fits[pixel]
        legend = (
            f"{label} raw=({segment},{row},{col}) "
            f"slope={fit['slope']:+.4f} ADU/frame R2={fit['r_squared']:.3f}"
        )
        raw_axis.plot(frame, raw14, color=color, linewidth=1.0, label=legend)
        raw_axis.plot(
            fit["fit_frame"],
            fit["prediction"],
            color=color,
            linewidth=2.0,
            linestyle="--",
        )
        delta_axis.plot(
            frame,
            raw14 - raw14[0],
            color=color,
            linewidth=1.0,
            label=label,
        )

    raw_axis.axvspan(fit_start + 1, fit_stop, color="#dddddd", alpha=0.25)
    raw_axis.set_ylabel("raw14 [ADU]")
    raw_axis.set_title("Charge-injection response and linear fits")
    raw_axis.grid(True, alpha=0.25)
    raw_axis.legend(loc="best", fontsize=8)

    delta_axis.set_xlabel("Decoded image frame / pulser step")
    delta_axis.set_ylabel("raw14 - first raw14 [ADU]")
    delta_axis.grid(True, alpha=0.25)
    delta_axis.legend(loc="best", fontsize=8)

    fig.suptitle(args.data_file.name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    print(f"saved_plot: {args.output}")
    if not args.no_show and os.environ.get("DISPLAY"):
        plt.show()
    plt.close(fig)


def main():
    args = _parse_args()
    if not args.data_file.exists():
        print(f"Data file does not exist: {args.data_file}", file=sys.stderr)
        return 1

    traces, decoded, decode_errors = _decode_traces(args)
    frame = np.arange(1, decoded + 1, dtype=np.float64)
    fit_start = args.fit_start - 1
    fit_stop = decoded if args.fit_stop is None else min(args.fit_stop, decoded)
    if fit_start >= fit_stop - 1:
        raise ValueError(
            f"fit range {args.fit_start}:{fit_stop} contains fewer than two decoded frames"
        )

    fits = {}
    print(f"decoded_frames: {decoded}")
    print(f"decode_errors: {decode_errors}")
    print(f"fit_frames: {fit_start + 1}:{fit_stop}")
    for pixel, values in traces.items():
        label, segment, row, col = pixel
        raw14 = np.asarray(values["raw14"], dtype=np.float64)
        gainbit_counts = Counter(int(value) for value in values["gainbit"])
        fits[pixel] = _fit_trace(frame, raw14, fit_start, fit_stop)
        fit = fits[pixel]
        print(
            f"{label}: raw=({segment},{row},{col}) "
            f"raw14_first={raw14[0]:.0f} raw14_last={raw14[-1]:.0f} "
            f"raw14_range={np.min(raw14):.0f}:{np.max(raw14):.0f} "
            f"slope={fit['slope']:+.6f} ADU/frame r_squared={fit['r_squared']:.6f} "
            f"gainbit_counts={dict(sorted(gainbit_counts.items()))}"
        )

    if args.csv_path is not None:
        _write_csv(args.csv_path, frame, traces)
        print(f"saved_csv: {args.csv_path}")
    _plot(args, frame, traces, fits, fit_start, fit_stop)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
