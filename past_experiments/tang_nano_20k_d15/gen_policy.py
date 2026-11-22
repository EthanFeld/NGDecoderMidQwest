"""Generate shared compile-time constants for a decoder policy."""

import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parent
POLICY = json.loads((ROOT / "data/policy.json").read_text())


def values(policy=None):
    p = POLICY if policy is None else policy
    near, wide, repair = p["boundary_extension"], p["wide_even"], p["repair"]
    order = p["boundary_order"]
    if not 33 <= p["event_limit"] <= 248 or sorted(order) != [0, 1, 2, 3]:
        raise ValueError("Expected 33 to 248 event slots and four distinct boundary masks.")
    raw_limit = p.get("raw_event_limit", p["event_limit"])
    if not p["event_limit"] <= raw_limit <= 1792:
        raise ValueError("Raw event limit must cover the event buffer and fit the projected syndrome.")
    if not 1 <= near["event_limit"] <= p["standard_event_limit"] <= p["event_limit"]:
        raise ValueError("Event thresholds must be in increasing order.")
    return {
        "QEC_EVENT_LIMIT": p["event_limit"], "QEC_STANDARD_EVENT_LIMIT": p["standard_event_limit"],
        "QEC_RAW_EVENT_LIMIT": raw_limit,
        "QEC_FIRST_SPAN": p["first_span"], "QEC_SECOND_SPAN": p["second_span"],
        "QEC_BOUNDARY_DEPTH": p["combined_boundary_depth"], "QEC_EXTENDED_SPAN": p["extended_span"],
        "QEC_CROSSING_LIMIT": p["extended_crossing_limit"], "QEC_DENSE_SPAN": p["dense_span"],
        "QEC_DENSE_DEPTH": p["dense_boundary_depth"], "QEC_DENSE_CROSSINGS": p["dense_crossing_limit"],
        "QEC_NEAR_SPAN": near["max_span"], "QEC_NEAR_GROUPS": near["max_groups"],
        "QEC_NEAR_EVENT_LIMIT": near["event_limit"], "QEC_NEAR_BUDGET": near["budget"],
        "QEC_WIDE_SPAN": wide["max_span"], "QEC_WIDE_LOW": wide["first_band"],
        "QEC_WIDE_HIGH": wide["last_band"], "QEC_WIDE_DEPTH": wide["boundary_depth"],
        "QEC_WIDE_CROSSINGS": wide["crossing_limit"],
        "QEC_REPAIR_LOW": repair["first_band"], "QEC_REPAIR_HIGH": repair["last_band"],
        "QEC_REPAIR_BANDS": repair["band_difference"], "QEC_REPAIR_DEPTH": repair["boundary_depth"],
        "QEC_VARIANTS": len(order), "QEC_VARIANT_BITS": sum(mask << (2 * i) for i, mask in enumerate(order)),
    }


def generate(path, prefix, policy=None):
    data = "".join(f"{prefix}define {k} {v}\n" for k, v in values(policy).items())
    if path.exists() and path.read_text() == data:
        return
    with NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as temp:
        temp.write(data)
    Path(temp.name).replace(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("header", type=Path, nargs="?")
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--rtl-output", type=Path)
    args = parser.parse_args()
    if args.policy and not args.rtl_output:
        parser.error("--policy requires --rtl-output")
    policy = json.loads(args.policy.read_text()) if args.policy else POLICY
    generate(args.rtl_output or ROOT / "rtl/policy.vh", chr(96), policy)
    if args.header:
        generate(args.header, "#", policy)
