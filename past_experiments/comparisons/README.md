# Archived decoder comparisons

Paired Surface memory-Z comparisons across distances 3, 7 and 15, including FPGA-only, tensor-only, telescope, ordinary/correlated PyMatching and IBM Relay-BP results. [model/results.csv](model/results.csv) contains the aggregate model/software measurements; [model/results.json](model/results.json) retains settings and provenance. Physical-board captures and scored comparisons are under `board/`.

`sweep.py` runs the configured comparison, `board_replay.py` verifies captured inputs and replies, and `report.py` renders the recorded figures. Run their command-line help from the repository root after installing the main PyMatching telescope and `"./past_experiments/decoder_testing[ibm-relay]"`. Historical records retain their original source paths and measurement cohorts.
