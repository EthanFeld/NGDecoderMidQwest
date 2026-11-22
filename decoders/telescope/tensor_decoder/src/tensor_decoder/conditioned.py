"""Condition graph marginals on a proposed opposite-basis correction.

Each complete DEM fault must project to at most one edge in each check basis.
Conditioned one-edge marginals are exact for that proposed edge assignment;
matching the resulting marginals ignores their remaining mutual correlations.
"""

from __future__ import annotations

import numpy as np

from .model import FaultModel, _column_matrix, memory_detector_indices


class ConditionedMatching:
    def __init__(
        self, context, joint_check=False, target_other=False, *, joint_matching_factory=None,
    ):
        self.num_detectors = context.num_detectors
        self.indices = memory_detector_indices(context)
        if target_other:
            self.indices = np.setdiff1d(np.arange(self.num_detectors), self.indices)
        self.memory = set(map(int, self.indices))
        mapping = {int(d): i for i, d in enumerate(self.indices)}
        model = FaultModel.from_context(context)
        if model.detector_offset.any() or model.observable_offset.any():
            raise ValueError("conditioned prototype requires zero deterministic offsets")
        if np.any(model.probabilities >= 0.5):
            raise ValueError("conditioned prototype requires fault priors below one half")
        memory_keys, other_keys, logs = {}, {}, {}
        for p, ds, ls in zip(model.probabilities, model.detectors, model.observables, strict=True):
            if target_other:
                ls = ()
            md = tuple(d for d in ds if d in self.memory)
            od = tuple(d for d in ds if d not in self.memory)
            if len(md) > 2 or len(od) > 2 or (not md and ls):
                raise ValueError("each fault must project to one graph edge per basis")
            a = memory_keys.setdefault((md, bool(ls)), len(memory_keys)) if md else -1
            b = other_keys.setdefault(od, len(other_keys)) if od else -1
            if a >= 0 or b >= 0:
                logs[a, b] = logs.get((a, b), 0.0) + np.log1p(-2 * p)
        self.other_keys = other_keys
        self.memory_keys = tuple(memory_keys)
        self.edge_lookup = {ds: i for i, (ds, _) in enumerate(self.memory_keys)}
        if len(self.edge_lookup) != len(self.memory_keys):
            raise ValueError("parallel memory edges have different logical actions")
        columns = [tuple(mapping[d] for d in ds) for ds, _ in self.memory_keys]
        self.check_matrix = _column_matrix(columns, len(self.indices)).tocsc()
        self.observable_matrix = _column_matrix(
            [((0,) if ls else ()) for _, ls in self.memory_keys], 1
        ).tocsc()
        self.sector_matrix = None
        other_logs = np.zeros(len(other_keys))
        prior_logs = np.zeros(len(memory_keys))
        for (a, b), value in logs.items():
            if b >= 0:
                other_logs[b] += value
            if a >= 0:
                prior_logs[a] += value
        self.prior_probability = -np.expm1(prior_logs) / 2
        self.prior_weights = np.log1p(-self.prior_probability) - np.log(self.prior_probability)
        pairs = sorted((a, b, value) for (a, b), value in logs.items() if a >= 0)
        self.pair_other = np.asarray([b for _, b, _ in pairs], dtype=np.int32)
        self.starts = np.flatnonzero(np.r_[True, np.diff([a for a, _, _ in pairs]) != 0])
        q0, q1 = [], []
        for _, b, value in pairs:
            x_odd = -np.expm1(value) / 2
            if b < 0:
                q0.append(x_odd)
                q1.append(x_odd)
            else:
                rest = min(0.0, other_logs[b] - value)
                rest_odd = -np.expm1(rest) / 2
                total_odd = -np.expm1(other_logs[b]) / 2
                q0.append(x_odd * rest_odd / (1 - total_odd))
                q1.append(x_odd * (1 - rest_odd) / total_odd)
        self.q0, self.q1 = np.asarray(q0), np.asarray(q1)
        self.joint_matching = None
        self.joint_loop_edges = None
        if joint_check:
            joint_columns = [
                tuple(([a] if a >= 0 else []) + ([len(memory_keys) + b] if b >= 0 else []))
                for a, b in logs
            ]
            joint_probability = -np.expm1(np.asarray(list(logs.values()))) / 2
            joint_weights = np.log1p(-joint_probability) - np.log(joint_probability)
            joint_matrix = _column_matrix(joint_columns, len(memory_keys) + len(other_keys)).tocsc()
            if joint_matching_factory is None:
                import pymatching

                self.joint_matching = pymatching.Matching.from_check_matrix(
                    joint_matrix,
                    weights=joint_weights,
                    faults_matrix=_column_matrix([()] * len(logs), 1).tocsc(),
                    use_virtual_boundary_node=True,
                )
            else:
                self.joint_matching = joint_matching_factory(joint_matrix, joint_weights)
            self.joint_matching.ensure_num_fault_ids(1)

    def weights(self, correction_edges, strength=1.0):
        if not np.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("conditioning strength must be in [0,1]")
        parity = np.zeros(len(self.other_keys) + 1, dtype=bool)
        for a, b in correction_edges:
            a, b = int(a), int(b)
            if a in self.memory:
                continue
            ds = (a,) if b < 0 else tuple(sorted((a, b)))
            if ds not in self.other_keys:
                raise ValueError("opposite-basis correction edge absent from DEM")
            parity[self.other_keys[ds]] ^= True
        # Index -1 is the final, always-false entry for faults invisible to that basis.
        q = np.where(parity[self.pair_other], self.q1, self.q0)
        bias = np.multiply.reduceat(1 - 2 * q, self.starts)
        probability = np.clip((1 - bias) / 2, 1e-12, 1 - 1e-12)
        conditioned = np.log1p(-probability) - np.log(probability)
        return self.prior_weights + strength * (conditioned - self.prior_weights)

    def decode_with_gain(self, syndrome, correction_edges, strength=1.0, return_edges=False):
        import pymatching

        weights = self.weights(correction_edges, strength)
        matching = pymatching.Matching.from_check_matrix(
            self.check_matrix,
            weights=weights,
            faults_matrix=self.observable_matrix,
            use_virtual_boundary_node=True,
            merge_strategy="disallow",
        )
        matching.ensure_num_fault_ids(1)
        proposal = None
        if return_edges:
            chosen = matching.decode_to_edges_array(np.asarray(syndrome)[self.indices])
            proposal = np.asarray(
                [(int(self.indices[a]), int(self.indices[b]) if b >= 0 else -1) for a, b in chosen],
                dtype=np.int64,
            ).reshape(-1, 2)
            label, weight = 0, 0.0
            for a, b in proposal:
                ds = (int(a),) if b < 0 else tuple(sorted((int(a), int(b))))
                edge = self.edge_lookup[ds]
                label ^= int(self.memory_keys[edge][1])
                weight += weights[edge]
            prediction = np.array([label], dtype=np.uint8)
        else:
            prediction, weight = matching.decode(
                np.asarray(syndrome)[self.indices], return_weight=True
            )
        baseline = 0.0
        for a, b in correction_edges:
            a, b = int(a), int(b)
            if a not in self.memory:
                continue
            ds = (a,) if b < 0 else tuple(sorted((a, b)))
            if ds not in self.edge_lookup:
                raise ValueError("memory correction edge absent from DEM")
            baseline += weights[self.edge_lookup[ds]]
        if return_edges:
            return prediction, baseline - weight, proposal
        return prediction, baseline - weight

    def joint_score(self, correction_edges):
        if self.joint_matching is None:
            raise RuntimeError("joint graph was not compiled")
        events = np.zeros(len(self.memory_keys) + len(self.other_keys), dtype=np.uint8)
        for a, b in correction_edges:
            a, b = int(a), int(b)
            ds = (a,) if b < 0 else tuple(sorted((a, b)))
            if a in self.memory:
                events[self.edge_lookup[ds]] ^= 1
            else:
                events[len(self.memory_keys) + self.other_keys[ds]] ^= 1
        return self.joint_matching.decode(events, return_weight=True)[1]

    def sector_costs(self, syndrome, correction_edges):
        """Minimum conditioned graph weight in each logical sector, not a partition sum."""
        import pymatching
        from scipy.sparse import vstack

        if self.sector_matrix is None:
            logical_columns = np.asarray(self.observable_matrix.sum(axis=0)).ravel() != 0
            degree = np.diff(self.check_matrix.indptr)
            if not logical_columns.any() or np.any(logical_columns & (degree != 1)):
                raise ValueError("sector constraints require logical edges on a boundary")
            self.sector_matrix = vstack((self.check_matrix, self.observable_matrix), format="csc")
        weights = self.weights(correction_edges)
        augmented = np.r_[np.asarray(syndrome)[self.indices], np.uint8(0)]
        costs = []
        for sector in (0, 1):
            matching = pymatching.Matching.from_check_matrix(
                self.sector_matrix,
                weights=weights,
                faults_matrix=self.observable_matrix,
                use_virtual_boundary_node=True,
                merge_strategy="disallow",
            )
            matching.ensure_num_fault_ids(1)
            augmented[-1] = sector
            _, weight = matching.decode(augmented, return_weight=True)
            costs.append(float(weight))
        return np.asarray(costs)

    def joint_gain(self, baseline, proposed_memory):
        other = [(int(a), int(b)) for a, b in baseline if int(a) not in self.memory]
        proposal = np.concatenate(
            (proposed_memory, np.asarray(other, dtype=np.int64).reshape(-1, 2))
        )
        try:
            return self.joint_score(baseline) - self.joint_score(proposal)
        except ValueError:
            # Infeasible edge-event patterns do not justify changing the baseline.
            return float("nan")

    def decode(self, syndrome, correction_edges, strength=1.0):
        return self.decode_with_gain(syndrome, correction_edges, strength)[0]
