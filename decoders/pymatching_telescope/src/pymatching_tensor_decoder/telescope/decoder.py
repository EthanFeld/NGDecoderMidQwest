"""Distance-15 prior tilt with the repository's full-syndrome tensor tail."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pymatching_tensor_decoder import TensorDecoder
from pymatching_tensor_decoder.baseline import CorrelatedMatching
from pymatching_tensor_decoder.model import validate_input

from .growth_reference import GrowthModel
from .host import Link, unpack
from .reference import Predecoder
from .rescue import compile_patch

PRIOR_TILT, RESIDUAL_CHECK, TENSOR = range(3)
POLICIES = ("plain", "guarded")


@dataclass
class FrontBatch:
    """One row per shot. A negative cycle count marks an unavailable reply."""

    accepted: np.ndarray
    residual: np.ndarray
    logical: np.ndarray
    cycles: np.ndarray


class ReferenceFront:
    def __init__(self, image, growth=None):
        self.decoder = Predecoder(image)
        self.budget = image.get("growth_cycles", 0)
        self.growth = growth
        if self.budget and (growth is None or growth.vertices != image["checks"]):
            raise ValueError("growth simulation requires its matching topology and cycle model")

    def decode(self, memory):
        result = self.decoder.decode(memory)
        if self.budget:
            self.growth.recover(result, memory, self.budget)
        return FrontBatch(result.accepted, result.residual, result.logical, result.clocks)


class SerialFront:
    def __init__(self, port, image):
        plan = Predecoder(image).plan
        budget = image.get("growth_cycles", 0)
        self.link = Link(port, image, plan.max_clocks + (budget + 1 if budget else 0))

    def decode(self, memory):
        n = len(memory)
        front = FrontBatch(
            np.zeros(n, dtype=bool), memory.copy(),
            np.zeros(n, dtype=np.uint8), np.full(n, -1, dtype=np.int32),
        )
        for i, syndrome in enumerate(memory):
            try:
                reply = self.link.run(syndrome, "tilt")
                residual = None if reply.residual is None else unpack(
                    reply.residual, self.link.width, self.link.rows,
                )
                if residual is not None and np.any(residual & ~syndrome):
                    continue
            except (OSError, ValueError):
                continue
            front.cycles[i] = reply.cycles
            front.logical[i] = reply.logical
            front.accepted[i] = residual is None
            front.residual[i] = False if residual is None else residual
        return front


class SyndromeCuts:
    """Count the odd boundary-to-boundary cuts implied by each logical class."""

    def __init__(self, image):
        strips = compile_patch(image).strips
        self.cols = np.concatenate(strips)
        self.starts = np.cumsum([0, *map(len, strips[:-1])])
        self.width = image["width"]
        self.rows = image["rounds"] + 1
        self.distance = image["distance"]

    def zero_class_crossings(self, memory):
        site = np.bitwise_xor.reduce(memory.reshape(-1, self.rows, self.width), axis=1)
        strips = np.bitwise_xor.reduceat(site[:, self.cols], self.starts, axis=1)
        prefix = np.bitwise_xor.accumulate(strips, axis=1)
        return prefix.sum(axis=1, dtype=np.int16)


class TelescopeDecoder:
    def __init__(self, context, image, memory_indices, *, front=None, backend=None, device="cpu", policy="plain"):
        self.num_detectors = context.num_detectors
        self.memory_indices = np.asarray(memory_indices, dtype=np.intp)
        if (
            context.num_observables != 1
            or self.memory_indices.shape != (image["checks"],)
            or np.unique(self.memory_indices).size != image["checks"]
            or self.memory_indices.min() < 0
            or self.memory_indices.max() >= self.num_detectors
        ):
            raise ValueError("the FPGA profile and joint detector map are incompatible")
        growth = image["frontend"] == "cluster" and image.get("growth_cycles", 0) > 0
        if image["distance"] != 15 or not (image["frontend"] == "rescue" or growth):
            raise ValueError("the telescope requires a d15 prior-tilt rescue or growth FPGA profile")
        if policy not in POLICIES:
            raise ValueError(f"unknown telescope policy {policy!r}")
        self.front = front if front is not None else ReferenceFront(image)
        self.backend = backend if backend is not None else TensorDecoder(context, device=device)
        self.cuts = SyndromeCuts(image) if policy == "guarded" else None
        self.correlated = CorrelatedMatching(context) if policy == "guarded" else None

    def memory(self, events):
        validate_input(events, self.num_detectors)
        bits = np.unpackbits(events, axis=1, count=self.num_detectors, bitorder="little")
        return np.ascontiguousarray(bits[:, self.memory_indices], dtype=bool)

    def mixed_events(self, events, residual):
        """Replace memory checks for a residual experiment and retain the other basis."""
        validate_input(events, self.num_detectors)
        if residual.shape != (len(events), len(self.memory_indices)):
            raise ValueError("the residual does not match the FPGA detector layout")
        bits = np.unpackbits(events, axis=1, count=self.num_detectors, bitorder="little")
        bits[:, self.memory_indices] = residual
        return np.packbits(bits, axis=1, bitorder="little")

    def route_front(self, events, front, memory=None):
        """Finalize valid FPGA accepts and route only unresolved shots."""
        validate_input(events, self.num_detectors)
        if any(data.shape != (len(events),) for data in (front.accepted, front.logical, front.cycles)):
            raise ValueError("the front-end metadata does not match the batch")
        predictions = np.array(front.logical, dtype=np.uint8, copy=True).reshape(-1, 1)
        accepted = front.accepted & (front.cycles > 0)
        stages = np.full(len(events), TENSOR, dtype=np.uint8)
        stages[accepted] = PRIOR_TILT
        if (
            self.cuts is not None and not accepted.all()
            and np.shape(front.residual) == (len(events), len(self.memory_indices))
        ):
            self.apply_residual_guard(events, front, memory, stages, predictions)
        return predictions, stages

    def complete_with_stages(self, events, front, memory=None):
        """Return predictions and the stage that supplied each logical bit."""
        predictions, stages = self.route_front(events, front, memory)
        forwarded = stages == TENSOR
        if forwarded.any():
            tail_events = events[forwarded]
            tail_events.flags.writeable = False
            residual = np.asarray(front.residual)
            residual = (
                residual[forwarded]
                if residual.shape == (len(events), len(self.memory_indices))
                else None
            )
            predictions[forwarded] = self._decode_tail(
                tail_events, residual, front.logical[forwarded],
                (front.cycles[forwarded] > 0) & ~front.accepted[forwarded],
            )
        return predictions, stages

    def _decode_tail(self, events, residual, logical, valid):
        """Pass optional FPGA hints while always decoding the original syndrome."""
        hinted = getattr(self.backend, "decode_with_front", None)
        if hinted is None or residual is None:
            return self.backend.decode_batch(events)
        residual = np.asarray(residual)
        logical = np.asarray(logical)
        valid = np.asarray(valid, dtype=bool)
        rows = len(events)
        if (
            residual.shape != (rows, len(self.memory_indices))
            or logical.shape != (rows,)
            or valid.shape != (rows,)
            or residual.dtype.kind not in "biuf"
            or logical.dtype.kind not in "biuf"
        ):
            return self.backend.decode_batch(events)
        valid = valid.copy()
        valid &= ((logical == 0) | (logical == 1))
        if residual.dtype != np.bool_:
            valid &= ((residual == 0) | (residual == 1)).all(axis=1)
        return hinted(
            events, np.asarray(residual != 0, dtype=bool),
            np.where(valid, logical, 0).astype(np.uint8), valid,
            self.memory_indices,
        )

    def apply_residual_guard(self, events, front, memory, stages, predictions):
        residual_rows = np.asarray(front.residual)
        if residual_rows.dtype.kind not in "biuf":
            return
        valid_hint = ((front.logical == 0) | (front.logical == 1))
        if residual_rows.dtype != np.bool_:
            valid_hint &= ((residual_rows == 0) | (residual_rows == 1)).all(axis=1)
        pending = np.flatnonzero(~front.accepted & (front.cycles > 0) & valid_hint)
        if not pending.size:
            return
        weights = residual_rows[pending].sum(axis=1)
        candidates = pending[(weights > 0) & (weights <= 32)]
        if not candidates.size:
            return
        raw = self.memory(events[candidates]) if memory is None else memory[candidates]
        residual = np.asarray(residual_rows[candidates], dtype=bool)
        distance = self.cuts.distance
        raw_gap = distance - 2 * self.cuts.zero_class_crossings(raw)
        residual_gap = distance - 2 * self.cuts.zero_class_crossings(residual)
        residual_gap = np.where(front.logical[candidates], -residual_gap, residual_gap)
        hint = (
            (np.abs(raw_gap) >= 5) & (np.abs(residual_gap) >= 5)
            & ((raw_gap < 0) == (residual_gap < 0))
            & ~(residual & ~raw).any(axis=1)
        )
        proposed = candidates[hint]
        if proposed.size:
            baseline = self.correlated_labels(events[proposed])
            agrees = baseline == (raw_gap[hint] < 0)
            kept = proposed[agrees]
            predictions[kept, 0] = baseline[agrees]
            stages[kept] = RESIDUAL_CHECK

    def correlated_labels(self, events):
        events.flags.writeable = False
        return self.correlated.decode_batch(events)[:, 0]

    def complete(self, events, front):
        return self.complete_with_stages(events, front)[0]

    def decode_batch(self, events):
        memory = self.memory(events)
        return self.complete_with_stages(events, self.front.decode(memory), memory)[0]

    def decode_stream(self, events, defer_batch):
        """Yield ready (shot, bit, stage, cycles) records, including the final partial batch."""
        if type(defer_batch) is not int or defer_batch < 1:
            raise ValueError("the deferred batch size must be positive")
        memory = self.memory(events)
        pending = []

        def finish():
            indices, cycles, residuals, logicals, valid = zip(*pending, strict=True)
            tail = events[np.asarray(indices, dtype=np.intp)]
            tail.flags.writeable = False
            labels = self._decode_tail(
                tail, np.stack(residuals), np.asarray(logicals), np.asarray(valid),
            )
            for i, bit, count in zip(indices, labels[:, 0], cycles, strict=True):
                yield i, int(bit), TENSOR, count
            pending.clear()

        for i in range(len(events)):
            selected = slice(i, i + 1)
            row = memory[selected]
            reply = self.front.decode(row)
            bits, stages = self.route_front(events[selected], reply, row)
            stage, clocks = int(stages[0]), int(reply.cycles[0])
            if stage != TENSOR:
                yield i, int(bits[0, 0]), stage, clocks
                continue
            residual = np.asarray(reply.residual)
            has_hint = residual.shape == (1, len(self.memory_indices))
            pending.append((
                i, clocks,
                residual[0].copy() if has_hint else np.zeros(len(self.memory_indices), dtype=bool),
                reply.logical[0], bool(clocks > 0 and not reply.accepted[0] and has_hint),
            ))
            if len(pending) == defer_batch:
                yield from finish()
        if pending:
            yield from finish()


def main():
    from .telescope_circuit import build_context

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=Path(__file__).parent / "profiles/d15z-gap3-81mhz.json")
    parser.add_argument("--noise", type=float, default=.005)
    parser.add_argument("--device", choices=("cpu", "cuda", "cuda-batch"), default="cpu")
    parser.add_argument("--policy", choices=POLICIES, default="plain", help="host handling of FPGA defers")
    parser.add_argument("--defer-batch", type=int, help="stream accepted shots and queue this many tensor defers")
    parser.add_argument("--tensor-workers", type=int, default=1, help="CPU workers for batched conditioned solves")
    parser.add_argument("--fast-routing", action="store_true", help="use the exact threshold path accelerator")
    parser.add_argument("--cuda-correlated", action="store_true", help="run correlated matching on CUDA")
    parser.add_argument("--port", help="serial port for the FPGA prior-tilt image")
    parser.add_argument("--baud", type=int, default=1_000_000, help="UART baud rate for the loaded FPGA image")
    parser.add_argument("--growth-topology", type=Path, help="growth topology for software replay")
    parser.add_argument("--growth-library", type=Path, help="matching growth model DLL or shared library")
    parser.add_argument("--events", type=Path, help="packed full-basis detector rows in a NumPy .npy file")
    parser.add_argument("--shots", type=int, default=8, help="sample count for the demonstration")
    parser.add_argument("--seed", type=int, default=51918031)
    args = parser.parse_args()
    if min(args.shots, args.baud, args.tensor_workers) < 1 or args.defer_batch is not None and args.defer_batch < 1:
        parser.error("shot count, baud rate, worker count, and deferred batch size must be positive")

    image = json.loads(args.profile.read_bytes())
    context, mapping = build_context(image, args.noise)
    events = (
        np.load(args.events, allow_pickle=False)
        if args.events
        else context.circuit.compile_detector_sampler(seed=args.seed).sample(args.shots, bit_packed=True)
    )
    events.flags.writeable = False
    names = ("fpga" if image.get("growth_cycles") else "prior_tilt", "residual_check", "tensor")

    def write(i, bit, stage, cycles):
        print(json.dumps({
            "shot": i, "logical": bit, "fpga_cycles": cycles if cycles > 0 else None,
            "stage": names[stage], "link_fault": cycles < 0,
        }), flush=True)

    def run(frontend):
        telescope = TelescopeDecoder(context, image, mapping, front=frontend, backend=backend, policy=args.policy)
        if args.defer_batch is not None:
            for result in telescope.decode_stream(events, args.defer_batch):
                write(*result)
            return
        memory = telescope.memory(events)
        front = telescope.front.decode(memory)
        labels, stages = telescope.complete_with_stages(events, front, memory)
        for i, (label, stage, cycles) in enumerate(zip(labels[:, 0], stages, front.cycles, strict=True)):
            write(i, int(label), int(stage), int(cycles))

    backend = TensorDecoder(
        context, device=args.device, solve_workers=args.tensor_workers, path_bucket=args.fast_routing,
        cuda_correlated=args.cuda_correlated,
    )
    try:
        if args.port:
            import serial

            with serial.Serial(args.port, args.baud, timeout=.025, write_timeout=.4) as port:
                port.reset_input_buffer()
                run(SerialFront(port, image))
        else:
            growth = None
            if image.get("growth_cycles"):
                if args.growth_topology is None or args.growth_library is None:
                    parser.error("growth software replay requires --growth-topology and --growth-library")
                growth = GrowthModel(args.growth_topology, args.growth_library)
            run(ReferenceFront(image, growth))
    finally:
        backend.close()


if __name__ == "__main__":
    main()
