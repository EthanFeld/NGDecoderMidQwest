"""Repack the routed d15 experiment for a lower fabric clock."""

import argparse
import json
from pathlib import Path

DIVIDERS = {27.0: (0, 32), 13.5: (1, 64)}


def retime(route_path, target_path, fabric_mhz=27.0):
    if route_path.resolve() == target_path.resolve():
        raise ValueError("Choose a separate output route for the clock change.")

    route = json.loads(route_path.read_bytes())
    module = route["modules"]["top"]
    pll, = (cell for cell in module["cells"].values() if cell["type"] == "rPLL")
    params = pll["parameters"]
    clock = tuple(int(params[name], 2) for name in ("FBDIV_SEL", "ODIV_SEL", "IDIV_SEL"))
    if clock != (1, 16, 0) or params["FCLKIN"] != "27.0":
        raise ValueError("Expected the routed 54 MHz experiment with a 27 MHz reference.")

    ports = pll["connections"]
    nets = module["netnames"]
    expected = {
        "CLKIN": "board_clock_IBUF_I_O",
        "CLKOUT": "fabric",
        "LOCK": "locked",
        "CLKFB": "$PACKER_GND",
        "RESET": "$PACKER_GND",
        "RESET_P": "$PACKER_GND",
    }
    if any(ports[pin] != nets[net]["bits"] for pin, net in expected.items()):
        raise ValueError("Expected the experimental image's PLL and reset wiring.")

    input_div, output_div = DIVIDERS[fabric_mhz]
    # The divider pairs keep the VCO at 864 MHz and retain the routed fabric clock.
    params["IDIV_SEL"] = f"{input_div:032b}"
    params["FBDIV_SEL"] = f"{0:032b}"
    params["ODIV_SEL"] = f"{output_div:032b}"
    target_path.write_text(json.dumps(route, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_route", type=Path)
    parser.add_argument("output_route", type=Path)
    parser.add_argument("--fabric-mhz", type=float, choices=DIVIDERS, default=27.0)
    args = parser.parse_args()
    retime(args.input_route, args.output_route, args.fabric_mhz)
