"""Conservative serialized operation model for one shared execution unit."""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Operations:
    neighbor_requests: int = 0
    neighbor_generation: int = 0
    edge_examinations: int = 0
    memory_reads: int = 0
    memory_writes: int = 0
    queue_operations: int = 0
    cluster_merges: int = 0
    find_steps: int = 0
    ml_lookups: int = 0
    recovery_operations: int = 0
    validation_operations: int = 0

    @property
    def total(self):
        # These are named work units, not CPU instructions. Reads/writes are
        # deliberately included when comparing ML with the conventional path.
        return sum(asdict(self).values())

    def report(self):
        return {**asdict(self), "total": self.total}


@dataclass(frozen=True)
class CycleModel:
    clock_mhz: float = 80.0
    memory_ports: int = 1
    read_latency: int = 2

    def __post_init__(self):
        if self.clock_mhz <= 0 or self.memory_ports != 1 or self.read_latency < 1:
            raise ValueError("model assumes a positive clock and one serialized memory port")

    def cycles(self, ops):
        # No bandwidth overlap assumed. BRAM reads take two cycles, writes one;
        # compute/queue/find/recovery work adds a cycle per modeled operation.
        return (ops.total + (self.read_latency - 1) * ops.memory_reads
                + 20 * ops.neighbor_requests)

    def summarize(self, operations, rounds):
        values = np.array([self.cycles(o) for o in operations], dtype=float)
        if not len(values):
            raise ValueError("no completed shots")
        return {
            "mean_cycles": float(np.mean(values)), "median_cycles": float(np.median(values)),
            "p95_cycles": float(np.percentile(values, 95)),
            "worst_observed_cycles": int(np.max(values)),
            "mean_cycles_per_round": float(np.mean(values) / rounds),
            "estimated_us_per_round": float(np.mean(values) / rounds / self.clock_mhz),
            "assumptions": {**asdict(self), "coordinate_arithmetic_cycles_per_request": 20,
                            "overlap": False},
            "status": "software estimate, not measured FPGA latency or a worst-case bound",
        }
