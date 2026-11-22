"""Sparse sum-product contraction across a bounded detector frontier.

Compilation stores a schedule, not a table over every possible full syndrome.
Completed checks are projected immediately and their bit lanes are reused.
Finite beam truncation is an explicit approximation, not an ML certificate.
"""

from __future__ import annotations

import ctypes as ct
import heapq
import sys
from dataclasses import dataclass

import numpy as np

from ..model import FaultModel, validate_input
from ..native import BytePointer, DoublePointer, support_library
from .certificates import CertificateCache, compile_certificates


class FrontierCapacityError(ValueError):
    """A configured model or frontier resource limit was exceeded."""


@dataclass(frozen=True)
class FrontierConfig:
    beam: int = 8
    refine_beam: int = 8
    refine_margin: float = 0.8
    lookahead_strength: float = 1.0
    certificate_bits: int = 0
    certificate_order: int = 2
    certificate_projections: int = 2
    max_frontier_bits: int = 63
    max_mechanisms: int = 10000
    max_detectors: int = 4096
    reverse_order: bool = False

    def __post_init__(self):
        if type(self.reverse_order) is not bool:
            raise ValueError("reverse_order must be boolean")
        if type(self.certificate_bits) is not int or not 0 <= self.certificate_bits <= 16:
            raise ValueError("certificate_bits must be an integer in [0,16]")
        if type(self.certificate_order) is not int or self.certificate_order not in (1, 2):
            raise ValueError("certificate_order must be 1 or 2")
        if (
            type(self.certificate_projections) is not int
            or not 1 <= self.certificate_projections <= 4
        ):
            raise ValueError("certificate_projections must be in [1,4]")
        for name in ("beam", "refine_beam"):
            if type(getattr(self, name)) is not int or not 1 <= getattr(self, name) <= 1024:
                raise ValueError(f"{name} must be an integer in [1, 1024]")
        if self.refine_beam < self.beam:
            raise ValueError("refine_beam must be >= beam")
        if not np.isfinite(self.refine_margin) or not 0 <= self.refine_margin <= 1:
            raise ValueError("refine_margin must be finite and in [0, 1]")
        if not np.isfinite(self.lookahead_strength) or not 0 <= self.lookahead_strength <= 1:
            raise ValueError("lookahead_strength must be finite and in [0, 1]")
        if type(self.max_frontier_bits) is not int or not 1 <= self.max_frontier_bits <= 127:
            raise ValueError("max_frontier_bits must be an integer in [1, 127]")
        for name in ("max_mechanisms", "max_detectors"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass
class FrontierPlan:
    probabilities: np.ndarray
    toggle: np.ndarray
    close: np.ndarray
    new_start: np.ndarray
    new_detector: np.ndarray
    new_lane: np.ndarray
    future_odds: np.ndarray
    detector_offset: np.ndarray
    unused_detectors: np.ndarray
    logical_offset: int
    max_active: int
    order: tuple[int, ...]
    zero_margin: float
    certificates: CertificateCache

    @property
    def nbytes(self):
        return (
            sum(value.nbytes for value in vars(self).values() if isinstance(value, np.ndarray))
            + self.certificates.nbytes
        )


def _lane_bit(lane):
    # Keep the logical sector at bit 63; additional detector lanes use word two.
    return 1 << (lane + (lane >= 63))


def _two_words(masks):
    return np.asarray([(x & ((1 << 64) - 1), x >> 64) for x in masks], dtype=np.uint64)


def _order_faults(model, reverse=False):
    # Incremental scores replace the original quadratic scan over all faults.
    # A score only changes when one of its adjacent detectors first opens or
    # becomes incident to the final unprocessed fault.
    adjacency = [[] for _ in model.detector_offset]
    for i, ds in enumerate(model.detectors):
        for d in ds:
            adjacency[d].append(i)
    remaining = [len(row) for row in adjacency]
    new = [len(ds) for ds in model.detectors]
    closes = [sum(remaining[d] == 1 for d in ds) for ds in model.detectors]
    maxima = [
        max((len(remaining) - 1 - d for d in ds), default=-1) if reverse else max(ds, default=-1)
        for ds in model.detectors
    ]
    pending = [True] * len(new)
    active = [False] * len(remaining)
    versions = [0] * len(new)
    heap = [(new[i] - closes[i], new[i], maxima[i], i, 0) for i in range(len(new))]
    heapq.heapify(heap)
    order = []
    while heap:
        _, _, _, index, version = heapq.heappop(heap)
        if not pending[index] or version != versions[index]:
            continue
        pending[index] = False
        order.append(index)
        changed = set()
        for d in model.detectors[index]:
            if not active[d]:
                active[d] = True
                for other in adjacency[d]:
                    if pending[other]:
                        new[other] -= 1
                        changed.add(other)
            remaining[d] -= 1
            if remaining[d] == 1:
                other = next(i for i in adjacency[d] if pending[i])
                closes[other] += 1
                changed.add(other)
            if remaining[d] == 0:
                active[d] = False
        for i in changed:
            versions[i] += 1
            heapq.heappush(heap, (new[i] - closes[i], new[i], maxima[i], i, versions[i]))
    return order


def _merge_identical_faults(model):
    # Independent faults with identical joint detector/logical action compose by
    # XOR, not by adding their probabilities. expm1 retains very rare priors.
    logs = {}
    for p, ds, ls in zip(model.probabilities, model.detectors, model.observables, strict=True):
        if not ds and not ls:
            continue
        key = ds, ls
        term = -np.inf if p == 0.5 else np.log1p(-2 * p)
        logs[key] = logs.get(key, 0.0) + term
    probabilities = np.array([-np.expm1(log) / 2 for log in logs.values()])
    return FaultModel(
        probabilities,
        tuple(ds for ds, _ in logs),
        tuple(ls for _, ls in logs),
        model.detector_offset,
        model.observable_offset,
    )


def _zero_syndrome_bound(model):
    # If no one- or two-fault pattern can be an undetected logical, all competing
    # logical mass is bounded by the >=3-fault count tail. This proof concerns the
    # independent DEM, including its deterministic offsets, not circuit truth.
    footprints = {}
    minimum = 3
    for ds, ls in zip(model.detectors, model.observables, strict=True):
        label = int(bool(ls))
        if not ds and label:
            minimum = 1
        if ds in footprints and footprints[ds] != label:
            minimum = min(minimum, 2)
        footprints[ds] = label
    count = np.zeros(minimum)
    count[0] = 1
    for p in model.probabilities:
        previous = count.copy()
        count *= 1 - p
        count[1:] += previous[:-1] * p
    guard = 64 * np.finfo(float).eps * (len(model.probabilities) + 1)
    lower = max(0, count[0] - guard)
    upper = min(1, max(0, 1 - count.sum()) + guard)
    return (lower - upper) / (lower + upper) if lower > upper + guard else 0.0


def compile_frontier(context, config: FrontierConfig | None = None):
    config = config or FrontierConfig()
    if context.num_observables != 1:
        raise ValueError("frontier prototype requires exactly one logical observable")
    if (
        context.num_detectors > config.max_detectors
        or context.dem.num_errors > config.max_mechanisms
    ):
        raise FrontierCapacityError("frontier compiler model resource cap exceeded")
    model = _merge_identical_faults(FaultModel.from_context(context))
    certificates = (
        compile_certificates(
            model,
            context.num_detectors,
            config.certificate_bits,
            config.certificate_order,
            config.certificate_projections,
        )
        if config.certificate_bits
        else CertificateCache(
            np.zeros(0, dtype=np.uint64), np.zeros(0, dtype=np.uint8), np.zeros(0), 0, 0
        )
    )
    order = _order_faults(model, config.reverse_order)
    last = {}
    for step, i in enumerate(order):
        for d in model.detectors[i]:
            last[d] = step
    available = list(range(config.max_frontier_bits - 1, -1, -1))
    lanes = {}
    toggle, close, start, detectors, new_lanes = [], [], [0], [], []
    peak = 0
    log_products = np.zeros(context.num_detectors)
    zeros = np.zeros(context.num_detectors, dtype=np.int32)
    for p, ds in zip(model.probabilities, model.detectors, strict=True):
        for d in ds:
            if p == 0.5:
                zeros[d] += 1
            else:
                log_products[d] += np.log1p(-2 * p)
    future_odds = np.ones((len(order), config.max_frontier_bits))
    for step, i in enumerate(order):
        mask = int(bool(model.observables[i])) << 63
        for d in model.detectors[i]:
            if d not in lanes:
                if not available:
                    raise FrontierCapacityError("frontier width exceeds configured bit-lane limit")
                lanes[d] = available.pop()
                detectors.append(d)
                new_lanes.append(lanes[d])
            mask ^= _lane_bit(lanes[d])
        peak = max(peak, len(lanes))
        ends = 0
        for d in model.detectors[i]:
            if last[d] == step:
                lane = lanes.pop(d)
                ends |= _lane_bit(lane)
                available.append(lane)
        toggle.append(mask)
        close.append(ends)
        start.append(len(detectors))
        p = model.probabilities[i]
        for d in model.detectors[i]:
            if p == 0.5:
                zeros[d] -= 1
            else:
                log_products[d] -= np.log1p(-2 * p)
        for d, lane in lanes.items():
            q = 0.5 if zeros[d] else -np.expm1(min(0, log_products[d])) / 2
            future_odds[step, lane] = (q / (1 - q)) ** config.lookahead_strength
    used = set(last)
    return FrontierPlan(
        model.probabilities[order].copy(),
        _two_words(toggle).reshape(-1, 2),
        _two_words(close).reshape(-1, 2),
        np.asarray(start, dtype=np.int32),
        np.asarray(detectors, dtype=np.int32),
        np.asarray(new_lanes, dtype=np.uint8),
        np.ascontiguousarray(future_odds[:, : max(1, peak)]),
        np.packbits(model.detector_offset, bitorder="little"),
        np.asarray([d for d in range(context.num_detectors) if d not in used], dtype=np.int32),
        int(model.observable_offset[0]),
        peak,
        tuple(order),
        _zero_syndrome_bound(model),
        certificates,
    )


def reference_contract(plan: FrontierPlan, events: np.ndarray, beam: int):
    """Independent Python sum-product reference, including sector-protected truncation."""
    if len(plan.certificates.keys):
        syndrome = int.from_bytes(bytes(events ^ plan.detector_offset), "little")
        index = np.searchsorted(plan.certificates.keys, np.uint64(syndrome))
        if index < len(plan.certificates.keys) and plan.certificates.keys[index] == syndrome:
            return (
                int(plan.certificates.labels[index]) ^ plan.logical_offset,
                float(plan.certificates.margins[index]),
                0,
                True,
            )
    for d in plan.unused_detectors:
        if ((int(events[d // 8]) ^ int(plan.detector_offset[d // 8])) >> (d % 8)) & 1:
            return 0, 0.0, 0, False
    if plan.zero_margin and not np.any(events ^ plan.detector_offset):
        return plan.logical_offset, plan.zero_margin, 0, True
    state = {0: 1.0}
    pruned = 0
    for step, p in enumerate(plan.probabilities):
        initial = 0
        for j in range(plan.new_start[step], plan.new_start[step + 1]):
            d = int(plan.new_detector[j])
            if ((int(events[d // 8]) ^ int(plan.detector_offset[d // 8])) >> (d % 8)) & 1:
                initial ^= _lane_bit(int(plan.new_lane[j]))
        output = {}
        close = int(plan.close[step, 0]) | (int(plan.close[step, 1]) << 64)
        toggle = int(plan.toggle[step, 0]) | (int(plan.toggle[step, 1]) << 64)
        for key, mass in state.items():
            key ^= initial
            for candidate, weight in ((key, mass * (1 - p)), (key ^ toggle, mass * p)):
                if not candidate & close:
                    output[candidate] = output.get(candidate, 0.0) + weight
        if not output:
            return 0, 0.0, pruned, False
        kept = {}

        def priority(item, step=step):
            k, mass = item
            residual = k & ~(1 << 63)
            score = mass
            while residual:
                bit = residual & -residual
                index = bit.bit_length() - 1
                score *= plan.future_odds[step, index - (index > 63)]
                residual ^= bit
            return -score, k

        for logical in range(2):
            candidates = sorted(
                ((k, v) for k, v in output.items() if ((k >> 63) & 1) == logical),
                key=priority,
            )
            pruned += max(0, len(candidates) - beam)
            kept.update(candidates[:beam])
        normalizer = max(kept.values())
        state = {k: v / normalizer for k, v in kept.items()}
    p0, p1 = state.get(0, 0), state.get(1 << 63, 0)
    if p0 + p1 == 0:
        return 0, 0.0, pruned, False
    prediction = int(p1 > p0) if p0 != p1 else plan.logical_offset
    return prediction ^ plan.logical_offset, abs(p1 - p0) / (p1 + p0), pruned, True


class _NativePlan(ct.Structure):
    _fields_ = (
        [(name, ct.c_int) for name in ("m", "n", "logical_offset", "unused_count", "lanes")]
        + [("p", DoublePointer)]
        + [(name, ct.POINTER(ct.c_uint64)) for name in ("toggle", "close")]
        + [(name, ct.POINTER(ct.c_int32)) for name in ("new_start", "new_detector")]
        + [
            ("new_lane", BytePointer),
            ("future_odds", DoublePointer),
            ("detector_offset", BytePointer),
            ("unused", ct.POINTER(ct.c_int32)),
        ]
        + [("zero_margin", ct.c_double)]
        + [
            ("certificate_count", ct.c_int),
            ("certificate_keys", ct.POINTER(ct.c_uint64)),
            ("certificate_labels", BytePointer),
            ("certificate_margins", DoublePointer),
        ]
    )


class FrontierTNDecoder:
    """Adaptive sparse TN beam contraction, explicitly approximate when truncated.

    The margin is a heuristic from retained masses, NOT a confidence certificate.
    Failed/numerically underflowed contractions explicitly use matching. A larger
    beam is retried on low-margin shots; no sampled answers or cross-shot state.
    """

    def __init__(self, context, config: FrontierConfig | None = None, library=None):
        self.config = config or FrontierConfig()
        self.num_detectors = context.num_detectors
        self.plan = plan = compile_frontier(context, self.config)
        # Static, model-derived decisions only. This dictionary is never learned
        # from decode calls, warmup, sampled syndromes, or sampled logical labels.
        self._certified_predictions = {
            int(key): int(label) ^ plan.logical_offset
            for key, label in zip(plan.certificates.keys, plan.certificates.labels, strict=True)
        }
        self._detector_offset_int = int.from_bytes(plan.detector_offset.tobytes(), "little")
        self.python_certificate_bytes = sys.getsizeof(self._certified_predictions) + sum(
            sys.getsizeof(key) for key in self._certified_predictions
        )
        self.lib = ct.CDLL(str(library or support_library()))
        try:
            version = self.lib.frontier_abi_version
        except AttributeError as exc:
            raise RuntimeError("native library lacks frontier contraction; rebuild it") from exc
        version.argtypes, version.restype = [], ct.c_int
        if version() != 5:
            raise RuntimeError("frontier kernel ABI mismatch; rebuild it")
        self.lib.frontier_decode.argtypes = [
            ct.POINTER(_NativePlan),
            BytePointer,
            ct.c_int,
            ct.c_int,
            ct.c_int,
            ct.c_int,
            ct.c_double,
            BytePointer,
            DoublePointer,
            ct.POINTER(ct.c_int32),
            BytePointer,
            BytePointer,
            BytePointer,
            BytePointer,
        ]
        self.lib.frontier_decode.restype = ct.c_int
        self.native = _NativePlan(
            context.num_detectors,
            len(plan.order),
            plan.logical_offset,
            len(plan.unused_detectors),
            plan.future_odds.shape[1],
            plan.probabilities.ctypes.data_as(DoublePointer),
            plan.toggle.ctypes.data_as(ct.POINTER(ct.c_uint64)),
            plan.close.ctypes.data_as(ct.POINTER(ct.c_uint64)),
            plan.new_start.ctypes.data_as(ct.POINTER(ct.c_int32)),
            plan.new_detector.ctypes.data_as(ct.POINTER(ct.c_int32)),
            plan.new_lane.ctypes.data_as(BytePointer),
            plan.future_odds.ctypes.data_as(DoublePointer),
            plan.detector_offset.ctypes.data_as(BytePointer),
            plan.unused_detectors.ctypes.data_as(ct.POINTER(ct.c_int32)),
            plan.zero_margin,
            len(plan.certificates.keys),
            plan.certificates.keys.ctypes.data_as(ct.POINTER(ct.c_uint64)),
            plan.certificates.labels.ctypes.data_as(BytePointer),
            plan.certificates.margins.ctypes.data_as(DoublePointer),
        )
        import pymatching

        self.matching = pymatching.Matching.from_detector_error_model(context.dem)
        self.matching.ensure_num_fault_ids(1)

    def _execute(self, events, diagnostics, validated=False):
        if not validated:
            validate_input(events, self.num_detectors)
        count = len(events)
        output = np.empty((count, 1), dtype=np.uint8)
        margins = np.empty(count) if diagnostics else None
        pruned = np.empty(count, dtype=np.int32) if diagnostics else None
        refined = np.empty(count, dtype=np.uint8) if diagnostics else None
        certified = np.empty(count, dtype=np.uint8) if diagnostics else None
        failed, numeric_loss = (np.empty(count, dtype=np.uint8) for _ in range(2))
        # Bound transient cache size and avoid an allocation proportional to full run length.
        for start in range(0, count, 1024):
            stop = min(start + 1024, count)
            status = self.lib.frontier_decode(
                ct.byref(self.native),
                events[start:stop].ctypes.data_as(BytePointer),
                stop - start,
                events.shape[1],
                self.config.beam,
                self.config.refine_beam,
                self.config.refine_margin,
                output[start:stop].ctypes.data_as(BytePointer),
                margins[start:stop].ctypes.data_as(DoublePointer) if diagnostics else None,
                pruned[start:stop].ctypes.data_as(ct.POINTER(ct.c_int32)) if diagnostics else None,
                refined[start:stop].ctypes.data_as(BytePointer) if diagnostics else None,
                failed[start:stop].ctypes.data_as(BytePointer),
                numeric_loss[start:stop].ctypes.data_as(BytePointer),
                certified[start:stop].ctypes.data_as(BytePointer) if diagnostics else None,
            )
            if status:
                raise RuntimeError(f"native frontier contraction returned error {status}")
        if not diagnostics and count == 1 and not (failed[0] or numeric_loss[0]):
            return output
        fallback = (failed != 0) | (numeric_loss != 0)
        if np.any(fallback):
            output[fallback] = self.matching.decode_batch(
                np.ascontiguousarray(events[fallback]),
                bit_packed_shots=True,
                bit_packed_predictions=True,
            )
        if not diagnostics:
            return output
        return output, {
            "margin": margins,
            "pruned_states": pruned,
            "refined": refined.astype(bool),
            "fallback": fallback,
            "unpruned": (pruned == 0) & ~fallback,
            "certified": certified.astype(bool),
        }

    def decode_with_diagnostics(self, events):
        return self._execute(events, True)

    def decode_batch(self, events):
        validate_input(events, self.num_detectors)
        if len(events) == 1:
            address = int.from_bytes(events.tobytes(), "little") ^ self._detector_offset_int
            label = self._certified_predictions.get(address)
            if label is not None:
                return np.array([[label]], dtype=np.uint8)
            if address == 0 and self.plan.zero_margin > 0:
                return np.array([[self.plan.logical_offset]], dtype=np.uint8)
        return self._execute(events, False, validated=True)
