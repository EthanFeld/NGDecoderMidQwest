"""Explicit bounds on a proposed packed implementation, not Python heap usage."""

import math
from dataclasses import asdict, dataclass, field


class MemoryOverflow(RuntimeError):
    pass


@dataclass(frozen=True)
class Limits:
    working_bytes: int = 64 * 1024
    max_vertices: int = 8192
    max_frontier: int = 8191
    max_clusters: int = 4096
    max_expansions: int = 100000
    max_seconds: float = 10.0

    def __post_init__(self):
        if min(self.working_bytes, self.max_vertices, self.max_frontier,
               self.max_clusters, self.max_expansions) < 1 or not math.isfinite(
                   self.max_seconds) or self.max_seconds <= 0:
            raise ValueError("memory and runtime limits must be positive")
        if self.max_frontier > 8191 or self.max_vertices > 65535 or self.max_clusters > 65535:
            raise ValueError("capacity exceeds packed record index width")


@dataclass
class MemoryTracker:
    num_detectors: int
    limits: Limits
    peaks: dict = field(default_factory=dict)
    peak_bytes: int = 0

    def observe(self, *, active, clusters, vertices, frontier, forest, recovery=0):
        counts = dict(active=active, clusters=clusters, vertices=vertices, frontier=frontier,
                      forest=forest, recovery=recovery)
        for name, cap in [("vertices", self.limits.max_vertices),
                          ("frontier", self.limits.max_frontier),
                          ("clusters", self.limits.max_clusters)]:
            if counts[name] > cap:
                raise MemoryOverflow(f"{name} capacity exceeded: {counts[name]} > {cap}")
        # Fixed packed records: cluster parent16/size16/flags8/pad8/head16/
        # tail16/schedule-link16 = 12 bytes. Frontier vertex15/cursor4/next13
        # = 4 bytes (8191 is the null link). Forest endpoints16+16/mask8/flags8
        # = 6 bytes. A 50%-load map uses key16/owner16 per slot. Recovery
        # uses key16/degree8/xor-edge16 at 50% load plus a 16-bit leaf queue.
        allocations = {
            "detector_bitmaps": 2 * ((self.num_detectors + 7) // 8),
            "active_events": 2 * active,
            "cluster_records": 12 * clusters,
            "expanded_vertex_map": 8 * vertices,
            "frontier_entries": 4 * frontier,
            "forest_edges": 6 * forest,
            "recovery_scratch": 12 * recovery,
            "correction_bitmap": (forest + 7) // 8 if recovery else 0,
            "metadata_buffering": 512,
        }
        total = sum(allocations.values())
        self.peak_bytes = max(self.peak_bytes, total)
        for name, count in counts.items():
            self.peaks[name] = max(self.peaks.get(name, 0), count)
        if total > self.limits.working_bytes:
            raise MemoryOverflow(f"working memory estimate exceeded: {total} > "
                                 f"{self.limits.working_bytes} bytes")
        return allocations

    def report(self):
        return {"peak_working_bytes": self.peak_bytes, "peak_records": dict(self.peaks),
                "limits": asdict(self.limits),
                "layout": "analytical packed records; excludes Python heap and static graph/model",
                "hash_map": "8 bytes/occupied vertex assumes at most 50% load"}
