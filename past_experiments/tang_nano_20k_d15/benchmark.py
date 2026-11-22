"""Evaluate the shipped acceptance policy on deterministic Stim input."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

from gen_policy import POLICY, ROOT
from geometry import circuit
from reference import REASONS, Reference, build_reference
from transport import GEOMETRY, PROJECTED_BYTES, project


SEED = 1928515701
RELEASE = {0.0005: 500_000, .001: 500_000, .002: 500_000, .003: 500_000, .004: 500_000,
           .005: 3_000_000, .006: 2_000_000, .0075: 1_000_000, .01: 500_000}
_reference = None


def init_worker(library):
    global _reference
    _reference = Reference(library)


def upper95(errors, trials):
    if trials == 0:
        return None
    if errors == 0:
        return float(-np.expm1(np.log(.05) / trials))
    if errors == trials:
        return 1.0
    from scipy.stats import beta

    return float(beta.ppf(.95, errors + 1, trials - errors))


def save_vectors(path, codes, packed):
    path.write_bytes(np.column_stack((codes, packed)).tobytes())


def directed_cases(engine, output):
    neighbors = [set() for _ in range(1792)]
    for a, b, logical, _ in GEOMETRY["edges"]:
        assert logical == 0
        neighbors[a].add(b)
        neighbors[b].add(a)
    site_band = lambda site: site % 112 // 8
    path = [4 * 112 + 2 * 8 + 3]
    for _ in range(7):
        site = path[-1]
        path.append(min((v for v in neighbors[site] if site_band(v) == site_band(site) + 1),
                        key=lambda v: (abs(v // 112 - site // 112), v)))
    free = set(range(1792)) - (neighbors[8] | {8})
    pairs = []
    for a, b, _, _ in GEOMETRY["edges"]:
        if b - a == 112 and a in free and b in free:
            pairs.extend((a, b))
            free.difference_update(neighbors[a] | neighbors[b] | {a, b})
    cap = POLICY["event_limit"]
    raw_cap = POLICY.get("raw_event_limit", cap)
    assert len(pairs) > cap
    fixtures = [
        ([], 1), ([8], 2), ([12 * 8], 1), ([16], 0), ([8, 8 + 8 * 112], 0),
        (pairs[:136] + [8], 2), (pairs[:138] + [8], 0),
        (path, 1), (path[:-1], 0), (pairs[:cap], 1), (pairs[:cap+2], int(raw_cap >= cap+2)),
    ]
    for band, distance, verdict in ((6, 3, 1), (6, 4, 0), (2, 3, 0)):
        fixtures.append(([t * 112 + band * 8 + 4 for t in (1, 1+distance)], verdict))
    faults = GEOMETRY["edges"] + GEOMETRY["singles"]
    suites = {"directed": fixtures, "single_faults": [(f[:-2], f[-2]+1) for f in faults]}
    for name, cases in suites.items():
        packed = np.zeros((len(cases), PROJECTED_BYTES), dtype=np.uint8)
        for i, (sites, _) in enumerate(cases):
            for site in sites:
                packed[i, site // 8] |= 1 << (site % 8)
        codes, _ = engine.decode(packed)
        np.testing.assert_array_equal(codes, [code for _, code in cases])
        if output:
            save_vectors(output / f"{name}.bin", codes, packed)
    return {name: len(cases) for name, cases in suites.items()}


def audit_distances():
    """Compare the arithmetic distance through radius three to a finite-graph BFS."""
    size = 1792
    neighbors = [set() for _ in range(size)]
    for a, b, _, _ in GEOMETRY["edges"]:
        neighbors[a].add(b)
        neighbors[b].add(a)
    t, index = np.divmod(np.arange(size), 112)
    band, x = np.divmod(index, 8)
    u = x + (band + 2) // 2
    v = x + 7 - (band + 1) // 2
    for a in range(size):
        dt, du, dv = t - t[a], u - u[a], v - v[a]
        lo, hi = np.minimum(0, -dt), np.maximum(0, -dt)
        dist = np.abs(dt) + np.maximum(np.maximum(lo-du, du-hi), 0) + np.maximum(np.maximum(lo-dv, dv-hi), 0)
        seen = {a}
        shell = {a}
        for d in (1, 2, 3):
            shell = set().union(*(neighbors[b] for b in shell)) - seen
            seen |= shell
            if set(np.flatnonzero(dist <= d)) != seen:
                raise ValueError(f"Arithmetic geometry disagrees at site {a}, radius {d}.")
    for site, flip, _ in GEOMETRY["singles"]:
        assert (site % 112 // 8, flip) in ((0, 1), (13, 0))


def test(args):
    p, shots, vector_dir = args
    seed = SEED + round(p * 1_000_000)
    sampler = circuit(p).compile_detector_sampler(seed=seed)
    accepted = old_accepted = wrong = wrong_added = 0
    reasons_total = np.zeros(len(REASONS), dtype=np.int64)
    reasons_wrong = np.zeros_like(reasons_total)
    replay = []
    for offset in range(0, shots, 10_000):
        raw, actual = sampler.sample(min(10_000, shots-offset), bit_packed=True, separate_observables=True)
        packed = project(raw)
        got, why = _reference.decode(packed)
        old, _ = _reference.decode(packed, prior=True)
        inherited = old != 0
        np.testing.assert_array_equal(got[inherited], old[inherited])
        keep = got != 0
        bad = keep & (got != actual[:, 0] + 1)
        accepted += int(keep.sum())
        old_accepted += int(inherited.sum())
        wrong += int(bad.sum())
        wrong_added += int((bad & ~inherited).sum())
        reasons_total += np.bincount(why, minlength=len(REASONS))
        reasons_wrong += np.bincount(why[bad], minlength=len(REASONS))
        if vector_dir is not None and offset == 0:
            save_vectors(vector_dir / f"p{p:g}.bin", got[:4096], packed[:4096])
        replay.extend({"index": offset + int(i), "code": int(got[i]), "actual": int(actual[i, 0]),
                       "mode": REASONS[why[i]], "projected_hex": packed[i].tobytes().hex()}
                      for i in np.flatnonzero(bad))
    added = accepted - old_accepted
    return {"p": p, "seed": seed, "shots": shots, "prior_accepted": old_accepted,
            "accepted": accepted, "added": added, "wrong_accepts": wrong, "wrong_added": wrong_added,
            "wrong_per_accepted_upper95": upper95(wrong, accepted),
            "wrong_per_input_upper95": upper95(wrong, shots),
            "wrong_per_added_upper95": upper95(wrong_added, added),
            "reason_counts": dict(zip(REASONS, map(int, reasons_total))),
            "wrong_by_reason": dict(zip(REASONS, map(int, reasons_wrong))), "wrong_replays": replay}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", action="store_true", help="Run the nine-million-shot release sweep")
    parser.add_argument("--rates", type=float, nargs="+", default=[.005])
    parser.add_argument("--shots", type=int, default=50_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--vectors", type=Path)
    parser.add_argument("--out", type=Path, default=Path("benchmark_result.json"))
    args = parser.parse_args()
    if args.shots < 1 or args.workers < 1 or any(not 0 < rate < .1 for rate in args.rates):
        parser.error("Use positive counts and noise probabilities between zero and 0.1.")
    if args.vectors:
        args.vectors.mkdir(parents=True, exist_ok=True)
    build = ROOT / ".build"
    build.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=build) as scratch:
        library = build_reference(scratch)
        with Reference(library) as engine:
            audit_distances()
            audit = directed_cases(engine, args.vectors)
        jobs = RELEASE.items() if args.release else ((p, args.shots) for p in args.rates)
        context = get_context("spawn" if sys.platform == "win32" else "fork")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context,
                                 initializer=init_worker, initargs=(library,)) as pool:
            results = []
            for case in pool.map(test, ((p, shots, args.vectors) for p, shots in jobs)):
                results.append(case)
                print(f"p={case['p']:g} accepted {case['accepted']}/{case['shots']}, "
                      f"added {case['added']}, wrong {case['wrong_accepts']}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"circuit": "Stim 1.15.0 rotated Z memory, d15, 15 rounds",
                                    "policy": POLICY, "distance_audit_sites": 1792,
                                    "audited_vectors": audit, "cases": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
