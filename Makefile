# Everything runs inside the local virtual env, so you never need to "activate" it.
PYTHON_BOOTSTRAP ?= python3.12
VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(PY) -m pip install --quiet

# Where PX4-Autopilot is checked out, and whether Gazebo runs without a window.
# HEADLESS=1 needs no XQuartz; use `make sim HEADLESS=0` once XQuartz is installed.
PX4_DIR  ?= ../PX4-Autopilot
HEADLESS ?= 1
# The simulated drone. gz_x500_mono_cam has a forward camera; plain gz_x500 has
# none (then record with `make record CAMERA=synthetic`).
SIM_MODEL ?= gz_x500_mono_cam
CAMERA    ?= gazebo
GZ_MODEL  := $(patsubst gz_%,%,$(SIM_MODEL))_0
SHELL    := /bin/bash

# Phase 2A: generated worlds. ROW is an episode id from MATRIX.
MATRIX    ?= configs/pilot.yaml
ROW       ?= pilot-meadow
WORLD_DIR := data/worlds/$(ROW)
SIDE      ?= 30

.PHONY: venv test synthetic sim record record-fake view clean assets world worlds preview sim-world record-world

# ---- installation (each stamp file means "this group is installed") ---------

$(VENV)/bin/python:
	$(PYTHON_BOOTSTRAP) -m venv $(VENV)
	$(PIP) --upgrade pip

$(VENV)/.dev: pyproject.toml | $(VENV)/bin/python
	$(PIP) -e ".[dev]"
	touch $@

$(VENV)/.fly: pyproject.toml | $(VENV)/bin/python
	$(PIP) -e ".[fly]"
	touch $@

# Large download (about 900 MB installed). Only `make view` triggers it.
$(VENV)/.view: pyproject.toml | $(VENV)/bin/python
	$(PIP) -e ".[view]"
	touch $@

venv: $(VENV)/.dev

# ---- everyday targets -------------------------------------------------------

test: $(VENV)/.dev
	$(PY) -m pytest -v

# A 10-second fake episode: no PX4, no MAVSDK.
synthetic: $(VENV)/.dev
	$(PY) -m sardrone.sim.synthetic

# The real mission flown by a fake drone on simulated time (finishes instantly).
record-fake: $(VENV)/.dev
	$(PY) -m sardrone.sim.record_flight --fake

# Start PX4 SITL + Gazebo (leave it running; Ctrl-C to stop). PX4's own venv
# puts Gazebo on the PATH; pkg-config is pointed at the same environment.
sim:
	cd $(PX4_DIR) && ulimit -S -n 2048 && source .venv/bin/activate && \
	  export PKG_CONFIG_PATH="$$CONDA_PREFIX/lib/pkgconfig:$$CONDA_PREFIX/share/pkgconfig" && \
	  HEADLESS=$(HEADLESS) make px4_sitl $(SIM_MODEL)

# The real thing. Run `make sim` in another terminal first.
record: $(VENV)/.dev $(VENV)/.fly
	SARDRONE_GZ_PYTHON=$(PX4_DIR)/Tools/setup/macos/.pixi/envs/default/bin/python \
	  $(PY) -m sardrone.sim.record_flight --camera $(CAMERA) --gz-model $(GZ_MODEL)

# sardrone.fo.load sets VFF_MULTIMODAL=1 itself, before importing fiftyone;
# it is repeated here so the requirement is visible.
# ---- generated worlds (Phase 2A) --------------------------------------------

# Download the people and tree models from Gazebo Fuel (about 45 MB, once).
assets: $(VENV)/.dev
	$(PY) -c "from sardrone.worlds.assets import fetch_all; fetch_all()"

world: $(VENV)/.dev
	$(PY) -m sardrone.worlds.generate --matrix $(MATRIX) --row $(ROW)

worlds: $(VENV)/.dev
	$(PY) -m sardrone.worlds.generate --matrix $(MATRIX) --all

# Pictures of a world from fixed cameras, without flying (Gazebo only).
preview: world
	PX4_DIR=$(PX4_DIR) $(PY) -m sardrone.worlds.preview --world-dir $(WORLD_DIR)

# Gazebo on the generated world + PX4 attached to the drone in it.
sim-world: world
	PX4_DIR=$(PX4_DIR) $(PY) -m sardrone.sim.launch --world-dir $(WORLD_DIR)

# Run `make sim-world ROW=...` in another terminal first.
record-world: $(VENV)/.dev $(VENV)/.fly
	SARDRONE_GZ_PYTHON=$(PX4_DIR)/Tools/setup/macos/.pixi/envs/default/bin/python \
	  $(PY) -m sardrone.sim.record_flight --camera gazebo --world-dir $(WORLD_DIR) --side $(SIDE)

view: $(VENV)/.dev $(VENV)/.view
	VFF_MULTIMODAL=1 $(PY) -m sardrone.fo.load

clean:
	rm -rf $(VENV) .pytest_cache *.egg-info
