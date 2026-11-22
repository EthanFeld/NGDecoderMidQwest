"""Joint detector context for the prior-tilt and tensor-decoder stages."""

import math

import numpy as np
import stim
from surface_model import generate, layout
from tensor_decoder.model import FaultModel, memory_detector_indices

from tensor_decoder import DecoderContext, Experiment


def surface_circuit(distance: int, basis: str, rounds: int, p: float) -> stim.Circuit:
    """Build the Surface Telescope schedule with both check bases observed."""
    plan = layout(distance)
    circuit = stim.Circuit()
    data_qubits = list(range(len(plan.data)))
    first_ancilla = len(data_qubits)
    memory_checks = plan.memory(basis)
    ancilla_groups = [(kind, plan.memory(kind)) for kind in ("x", "z")]
    ticks = [[q for pair in tick for q in pair] for tick in plan.ticks]
    measurement_count = 0

    def prepare(kind, qubits):
        circuit.append("RX" if kind == "x" else "R", qubits)
        circuit.append("Z_ERROR" if kind == "x" else "X_ERROR", qubits, p)

    def measure(kind, qubits):
        nonlocal measurement_count
        start = measurement_count
        circuit.append("MX" if kind == "x" else "M", qubits, p)
        measurement_count += len(qubits)
        return range(start, measurement_count)

    def rec(indices):
        return [stim.target_rec(index - measurement_count) for index in indices]

    prepare(basis, data_qubits)
    previous = {}
    for t in range(rounds):
        circuit.append("DEPOLARIZE1", data_qubits, p)
        for kind, checks in ancilla_groups:
            prepare(kind, [first_ancilla + check for check in checks])
        for qubits in ticks:
            circuit.append("CX", qubits)
            circuit.append("DEPOLARIZE2", qubits, p)
        current = {}
        for kind, checks in ancilla_groups:
            outcomes = measure(kind, [first_ancilla + check for check in checks])
            current.update(zip(checks, outcomes, strict=True))

        for check, (kind, x, y) in enumerate(plan.checks):
            # Conjugate checks become deterministic only as round differences.
            if t:
                outcomes = (current[check], previous[check])
            elif kind == basis:
                outcomes = (current[check],)
            else:
                continue
            circuit.append("DETECTOR", rec(outcomes), [x, y, t])
        previous = current

    final = measure(basis, data_qubits)
    for check in memory_checks:
        _, x, y = plan.checks[check]
        outcomes = (current[check], *(final[q] for q in plan.support[check]))
        circuit.append("DETECTOR", rec(outcomes), [x, y, rounds])
    circuit.append(
        "OBSERVABLE_INCLUDE", rec(final[q] for q in plan.observable(basis)), 0
    )
    return circuit


class SurfaceExperiment(Experiment):
    """Deterministic circuit identity for the tensor decoder's context validation."""

    def __post_init__(self):
        super().__post_init__()
        if self.geometry != "rotated" or self.noise_model != "circuit":
            raise ValueError("Surface Telescope requires rotated circuit-level memory")
        if (
            self.distance not in (3, 7, 15)
            or self.rounds > self.distance
            or self.noise >= 0.1
        ):
            raise ValueError(
                "Surface Telescope requires d3, d7 or d15, up to d rounds, and p < 0.1"
            )

    def circuit(self):
        return surface_circuit(self.distance, self.basis, self.rounds, self.noise)


def _faults(image):
    return {(tuple(ds), logical): p for ds, logical, p in image["faults"]}


def _validate_faults(actual, expected, source):
    if actual.keys() != expected.keys():
        raise ValueError(
            f"{source} fault edges or logical labels differ from the Surface model"
        )
    if any(
        not math.isclose(p, expected[key], rel_tol=1e-10, abs_tol=1e-13)
        for key, p in actual.items()
    ):
        raise ValueError(f"{source} fault probabilities differ from the Surface model")


def build_context(image: dict, noise: float) -> tuple[DecoderContext, np.ndarray]:
    """Return the joint context and its read-only FPGA-order detector indices."""
    distance, basis, rounds = (image[name] for name in ("distance", "basis", "rounds"))
    experiment = SurfaceExperiment(distance, rounds, noise, basis=basis)
    shipped = generate(distance, basis, rounds, image["p"])
    for field in ("coordinates", "width", "checks"):
        if image[field] != shipped[field]:
            raise ValueError(
                f"FPGA profile {field} does not match the Surface schedule"
            )
    _validate_faults(_faults(image), _faults(shipped), "FPGA profile")

    circuit = experiment.circuit()
    context = DecoderContext(
        experiment, circuit, circuit.detector_error_model(decompose_errors=True)
    )
    detector_ids = {
        tuple(coordinate): index
        for index, coordinate in circuit.get_detector_coordinates().items()
    }
    memory_indices = np.array(
        [
            detector_ids[x, y, t]
            for t in range(rounds + 1)
            for x, y in image["coordinates"]
        ],
        dtype=np.intp,
    )
    if not np.array_equal(memory_indices, memory_detector_indices(context)):
        raise ValueError("joint decoder and FPGA disagree on memory detector indices")

    model = (
        FaultModel.from_context(context)
        .project_detectors(memory_indices)
        .merge_equivalent_faults()
    )
    if model.detector_offset.any() or model.observable_offset.any():
        raise ValueError("joint circuit has deterministic projected faults")
    expected = (
        shipped
        if noise == image["p"]
        else (generate(distance, basis, rounds, noise) if noise else {"faults": []})
    )
    actual = {
        (detectors, int(bool(observables))): p
        for detectors, observables, p in zip(
            model.detectors, model.observables, model.probabilities, strict=True
        )
    }
    _validate_faults(actual, _faults(expected), "joint Stim projection")
    memory_indices.flags.writeable = False
    return context, memory_indices
