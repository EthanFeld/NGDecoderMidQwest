ifeq ($(OS),Windows_NT)
PYTHON ?= python
GROWTH_MODEL := build/growth_model.dll
else
PYTHON ?= python3
GROWTH_MODEL := build/growth_model.so
endif
CXX ?= c++
IVERILOG ?= iverilog
VVP ?= vvp
YOSYS ?= yosys
NEXTPNR ?= nextpnr-himbaechel
GOWIN_PACK ?= gowin_pack
DISTANCE ?= 15
BASIS ?= z
ROUNDS ?= $(DISTANCE)
P ?= .001
FRONTEND ?= rescue
RESCUE_GAP3_CUTS ?= 9 9
GROW_P ?= .007
GROW_RES ?= 2
GROW_ENGINE ?= serial
GROW_CUT_GUARD ?= 0
GROW_CUT_LIMIT ?=
GROW_BOUNDARY_BIAS ?= 0
GROW_SKIP_EMPTY ?= 0
GROW_COMPACT ?= 0
GROW_CYCLE_CAP ?=
IMAGE ?= build/d$(DISTANCE)$(BASIS)
CLOCK_MULT ?= 3
CLOCK_DIV ?= 1
CLOCK_MHZ = $(shell $(PYTHON) -c "import sys; m, d = map(int, sys.argv[1:]); print(27 * m // d if d else 0)" $(CLOCK_MULT) $(CLOCK_DIV))
SIM_UART_DIV ?= 8
CORE := rtl/surface.sv rtl/rescue.sv rtl/lookup.sv rtl/growth.sv rtl/cluster_linkage.sv
RTL := $(CORE) rtl/uart.sv rtl/link.sv

.PHONY: all image growth-model test vectors sim sim-map sim-growth check-stim bitstream
all: image

image:
	$(PYTHON) surface_model.py --distance $(DISTANCE) --basis $(BASIS) --rounds $(ROUNDS) --p $(P) --frontend $(FRONTEND) \
	  --growth-p $(GROW_P) --growth-resolution $(GROW_RES) --growth-engine $(GROW_ENGINE) --growth-boundary-bias=$(GROW_BOUNDARY_BIAS) \
	  $(if $(filter 1,$(GROW_CUT_GUARD)),--growth-cut-guard) $(if $(GROW_CUT_LIMIT),--growth-cut-limit $(GROW_CUT_LIMIT)) \
	  $(if $(filter 1,$(GROW_SKIP_EMPTY)),--growth-skip-empty) $(if $(filter 1,$(GROW_COMPACT)),--growth-compact) \
	  $(if $(GROW_CYCLE_CAP),--growth-cycle-cap $(GROW_CYCLE_CAP)) \
	  $(if $(RESCUE_GAP3_CUTS),--rescue-gap3-cuts $(RESCUE_GAP3_CUTS)) \
	  --clock-mhz $(CLOCK_MHZ) --output $(IMAGE)
	$(PYTHON) stencils.py $(IMAGE)

growth-model: $(GROWTH_MODEL)

$(GROWTH_MODEL): tools/growth_model.cpp growth_reference.py
	$(PYTHON) growth_reference.py $(CXX)

ifeq ($(FRONTEND),cluster)
vectors: $(GROWTH_MODEL)
endif

.PHONY: check-clock
check-clock:
	$(PYTHON) -c "import sys; m, d = map(int, sys.argv[1:]); f, remainder = divmod(27*m, d) if d > 0 else (0, 1); sys.exit(None if 1 <= m <= 64 and 1 <= d <= 9 and not remainder and (27 <= f <= 156 or m == d) else 'Use an integer-MHz PLL ratio with CLOCK_DIV at most 9 and an output in the 27-156 MHz range')" $(CLOCK_MULT) $(CLOCK_DIV)

test:
	$(PYTHON) -m unittest discover -s tests -v

check-stim:
	$(PYTHON) tests/stim_check.py

vectors: image
	$(PYTHON) tests/vectors.py $(IMAGE)

sim: vectors
	$(IVERILOG) -g2012 -I$(IMAGE) -s tb_core -o $(IMAGE)/core.vvp $(CORE) tests/tb_core.sv
	$(VVP) $(IMAGE)/core.vvp +vectors=$(IMAGE)/core.txt
	$(IVERILOG) -g2012 -I$(IMAGE) -s tb_link -Ptb_link.BIT_CYCLES=$(SIM_UART_DIV) -o $(IMAGE)/link.vvp $(RTL) tests/tb_link.sv
	$(VVP) $(IMAGE)/link.vvp +vectors=$(IMAGE)/uart.txt

sim-map: vectors
	$(IVERILOG) -g2012 -I$(IMAGE) -s tb_map -o $(IMAGE)/map.vvp rtl/lookup.sv tests/tb_map.sv
	$(VVP) $(IMAGE)/map.vvp +answers=$(IMAGE)/map_expected.hex

sim-growth: image $(GROWTH_MODEL)
	$(PYTHON) tests/growth_vectors.py $(IMAGE)
	$(IVERILOG) -g2012 -I$(IMAGE) -s tb_growth -o $(IMAGE)/growth.vvp rtl/growth.sv tests/tb_growth.sv
	$(VVP) $(IMAGE)/growth.vvp +vectors=$(IMAGE)/growth_vectors.txt

bitstream: check-clock image
	$(YOSYS) -p 'read_verilog -sv -D SC_CLOCK_MULT=$(CLOCK_MULT) -D SC_CLOCK_DIV=$(CLOCK_DIV) -I$(IMAGE) $(RTL); synth_gowin -family gw2a -nowidelut -top tang_nano20k -json $(IMAGE)/netlist.json'
	$(NEXTPNR) --Werror --json $(IMAGE)/netlist.json --write $(IMAGE)/route.json --report $(IMAGE)/timing.json --device GW2AR-LV18QN88C8/I7 --vopt family=GW2A-18C --vopt cst=tang_nano20k.cst --freq $(CLOCK_MHZ) --sdc board_clock.sdc
	$(GOWIN_PACK) -d GW2A-18C -o $(IMAGE)/surface.fs $(IMAGE)/route.json
