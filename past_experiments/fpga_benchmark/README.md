# FPGA comparison tools

Packet generation, Web Serial capture and paired software/board replay for the archived Surface FPGA and Tang Nano distance-15 designs. `vectors.py` prepares comparison inputs, `telescope_uart.py` prepares full joint-syndrome telescope captures and `tang_no_wire.py` measures the Tang software reference. Captures and companion fixtures are retained under `.build/`.

From the repository root, use `python past_experiments/fpga_benchmark/vectors.py --help` or `python past_experiments/fpga_benchmark/telescope_uart.py --help`. Python dependencies are supplied by the main telescope and the archived decoder benchmark package. Firmware profiles resolve from `past_experiments/surface_fpga/` and `past_experiments/tang_nano_20k_d15/`.
