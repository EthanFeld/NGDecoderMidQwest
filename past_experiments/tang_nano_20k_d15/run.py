"""Generate, test, decode, and score distance-15 memory shots."""

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from geometry import ROOT, circuit
from transport import (ACCEPT_0, ACCEPT_1, BAD_PACKET, DEFER, GEOMETRY, PROJECTED_BYTES,
                       REPLY_BYTES, parse_reply, request, telescope)


class UartFront:
    """Keep one packet in flight and fall back after a link failure."""

    def __init__(self, connection):
        self.connection = connection
        self.tag = 0
        self.fault = None

    def send(self, projected, *, corrupt=False):
        if self.fault is not None:
            return BAD_PACKET
        tag = self.tag
        self.tag = (tag + 1) & 255
        packet = request(projected, tag)
        if corrupt:
            packet = packet[:-1] + bytes([packet[-1] ^ 1])
        try:
            if self.connection.write(packet) != len(packet):
                raise ValueError("Incomplete request.")
            return parse_reply(self.connection.read(REPLY_BYTES), tag).code
        except (OSError, ValueError) as failure:
            self.fault = str(failure)
            return BAD_PACKET

    def __call__(self, rows):
        return np.fromiter((self.send(row) for row in rows), dtype=np.uint8, count=len(rows))


def open_port(device, baud=115_200):
    import serial

    port = serial.Serial(device, baud, timeout=.35, write_timeout=.35)
    try:
        port.reset_input_buffer()
    except BaseException:
        port.close()
        raise
    return port


def smoke(front):
    zeros = np.zeros(PROJECTED_BYTES, dtype=np.uint8)
    boundary = zeros.copy()
    site = next(site for site, logical, _ in GEOMETRY["singles"] if logical == 1)
    boundary[site // 8] |= 1 << (site % 8)
    dense = np.full(PROJECTED_BYTES, 255, dtype=np.uint8)
    tests = [
        ("clean", zeros, False, ACCEPT_0),
        ("single boundary fault", boundary, False, ACCEPT_1),
        ("dense", dense, False, DEFER),
        ("damaged CRC", zeros, True, BAD_PACKET),
        ("clean after damaged CRC", zeros, False, ACCEPT_0),
    ]
    for name, row, corrupt, expected in tests:
        actual = front.send(row, corrupt=corrupt)
        if actual != expected or front.fault is not None:
            hint = f" Link error was {front.fault}." if front.fault else ""
            raise ValueError(f"{name} returned status {actual}, expected {expected}.{hint}")
        print(f"{name} passed")


def make_sample(args):
    args.directory.mkdir(parents=True, exist_ok=True)
    code = circuit(args.p)
    packed, labels = code.compile_detector_sampler(seed=args.seed).sample(
        args.shots, bit_packed=True, separate_observables=True,
    )
    np.save(args.directory / "shots.npy", packed, allow_pickle=False)
    np.save(args.directory / "observables.npy", labels[:, 0], allow_pickle=False)
    description = {"shots": args.shots, "seed": args.seed, "p": args.p,
                   "layout": "Stim 1.15.0 rotated Z memory, distance 15, 15 rounds"}
    (args.directory / "sample.json").write_text(json.dumps(description, indent=2) + "\n")
    print(json.dumps(description))


def decode_file(args):
    import pymatching

    raw = np.load(args.input, allow_pickle=False)
    base = circuit(args.p)
    matching = pymatching.Matching.from_detector_error_model(
        base.detector_error_model(decompose_errors=True), enable_correlations=True,
    )

    def fallback(rows, _ids):
        return matching.decode_batch(rows, bit_packed_shots=True, enable_correlations=True)[:, 0]

    front = None
    mode = args.command
    context = open_port(args.port, args.baud) if mode == "decode" else nullcontext()
    with context as port:
        if mode == "decode":
            front = UartFront(port)
            decide = front
            logical, accepted = telescope(raw, decide, fallback)
        elif mode == "reference":
            from reference import Reference, build_reference

            with TemporaryDirectory() as temp, Reference(build_reference(temp)) as engine:
                logical, accepted = telescope(raw, lambda data: engine.decode(data)[0], fallback)
        else:
            logical, accepted = telescope(raw, lambda data: np.zeros(len(data), dtype=np.uint8), fallback)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    on_fpga = accepted if mode == "decode" else np.zeros_like(accepted)
    np.savez_compressed(args.output, logical=logical, accepted_on_fpga=on_fpga,
                        accepted_by_front_end=accepted)
    print(json.dumps({"shots": len(logical), "accepted_by_front_end": int(accepted.sum()),
                      "accepted_on_fpga": int(on_fpga.sum()), "decoded_by_cpu": int((~accepted).sum()),
                      "link_error": front.fault if front is not None else None}))


def score(args):
    labels = np.load(args.observables, allow_pickle=False)
    with np.load(args.predictions, allow_pickle=False) as data:
        logical = data["logical"]
        accepted = data["accepted_by_front_end"]
    if logical.ndim != 1 or labels.shape != logical.shape or accepted.shape != logical.shape:
        raise ValueError("Predictions, accept mask, and labels must have the same one-dimensional shape.")
    bad = logical != labels
    print(json.dumps({"shots": len(labels), "logical_errors": int(bad.sum()),
                      "front_end_accepts": int(accepted.sum()),
                      "wrong_front_end_accepts": int((bad & accepted).sum())}))


def verify():
    for line in (ROOT / "results/SHA256SUMS").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        file = ROOT / "bitstreams" / name
        if hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Checksum mismatch for {file.name}.")
        print(f"{file.name} OK")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    commands.add_parser("verify", help="Check the shipped bitstream SHA-256 hashes")
    commands.add_parser("ports", help="List available serial ports")
    usb = commands.add_parser("smoke", help="Send five checks to the board over USB-UART")
    usb.add_argument("--port", required=True)
    usb.add_argument("--baud", type=int, choices=(115_200, 1_000_000), default=115_200)
    sample = commands.add_parser("make-sample", help="Generate correctly ordered Stim input and labels")
    sample.add_argument("--shots", type=int, default=64)
    sample.add_argument("--seed", type=int, default=42857)
    sample.add_argument("--p", type=float, default=.005, help="Circuit noise probability, e.g. .005 for 0.5 percent")
    sample.add_argument("--directory", type=Path, required=True)
    for name, help_text in (("cpu", "Decode on CPU with correlated PyMatching"),
                            ("reference", "Test the FPGA policy in software with CPU fallback"),
                            ("decode", "Decode via USB-UART with CPU fallback")):
        job = commands.add_parser(name, help=help_text)
        job.add_argument("--input", type=Path, required=True)
        job.add_argument("--output", type=Path, required=True)
        job.add_argument("--p", type=float, default=.005, help="Noise probability for the fallback graph")
        if name == "decode":
            job.add_argument("--port", required=True)
            job.add_argument("--baud", type=int, choices=(115_200, 1_000_000), default=115_200)
    compare = commands.add_parser("score", help="Compare decoded flips to known Stim observables")
    compare.add_argument("--predictions", type=Path, required=True)
    compare.add_argument("--observables", type=Path, required=True)
    args = cli.parse_args()
    if hasattr(args, "p") and not 0 < args.p < .1:
        cli.error("Use an error probability between zero and 0.1.")
    if args.command == "make-sample" and args.shots < 1:
        cli.error("Use a positive sample count.")
    if hasattr(args, "output") and args.output.suffix.lower() != ".npz":
        cli.error("Use a .npz output path.")
    try:
        if args.command == "verify":
            verify()
        elif args.command == "ports":
            from serial.tools.list_ports import comports

            ports = sorted(comports(), key=lambda port: port.device)
            if not ports:
                print("No serial ports found.")
            for port in ports:
                print(f"{port.device}  {port.description}")
        elif args.command == "smoke":
            with open_port(args.port, args.baud) as port:
                smoke(UartFront(port))
        elif args.command == "make-sample":
            make_sample(args)
        elif args.command == "score":
            score(args)
        else:
            decode_file(args)
    except (OSError, ValueError) as failure:
        cli.exit(1, f"{failure}\n")


if __name__ == "__main__":
    main()
