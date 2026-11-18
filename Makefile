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
# Who looks for people: `oracle` reads ground truth; a model version (v4) looks at the images.
DETECTOR  ?= oracle

.PHONY: venv test synthetic sim record record-fake view clean assets world worlds preview sim-world record-world search search-fake batch batch-fake frames overlay curate autolabel sard export smoke-train train evaluate gapmap pipeline agent-eval agent-record curves false-alarms real-voc extra-real

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

# PyTorch and friends (about 1 GB). Only the Phase 2E targets trigger it.
$(VENV)/.ml: pyproject.toml | $(VENV)/bin/python
	$(PIP) -e ".[ml]"
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

# ---- search missions (Phase 2B) ----------------------------------------------

# Lawnmower search with confirm / report / return home. Needs `make sim-world`.
search: $(VENV)/.dev $(VENV)/.fly
	SARDRONE_GZ_PYTHON=$(PX4_DIR)/Tools/setup/macos/.pixi/envs/default/bin/python \
	  $(PY) -m sardrone.mission.search --world-dir $(WORLD_DIR) --detector $(DETECTOR)

# The same mission with a fake drone: no simulator, finishes in seconds.
search-fake: world
	$(PY) -m sardrone.mission.search --world-dir $(WORLD_DIR) --fake --overwrite

# ---- batches (Phase 2C) ------------------------------------------------------

# Every row of MATRIX, unattended. Safe to stop and run again: finished
# episodes are skipped. Starts and stops the simulator itself.
batch: $(VENV)/.dev $(VENV)/.fly
	PX4_DIR=$(PX4_DIR) $(PY) -m sardrone.batch --matrix $(MATRIX)

batch-fake: $(VENV)/.dev
	$(PY) -m sardrone.batch --matrix $(MATRIX) --fake

# ---- frames and labels (Phase 2D) --------------------------------------------

# Labelled pictures at 2 per second from every episode -> FiftyOne "sar-frames".
frames: $(VENV)/.dev $(VENV)/.view
	$(PY) -m sardrone.fo.frames

# 20 random frames with their label boxes drawn, for a human check.
overlay: $(VENV)/.dev $(VENV)/.view
	$(PY) -m sardrone.fo.overlay

# ---- curate, embed, auto-label (Phase 2E) ------------------------------------

# Embeddings, 2D map, near-duplicates, train/val split, saved views.
curate: $(VENV)/.dev $(VENV)/.view $(VENV)/.ml
	$(PY) -m sardrone.fo.curate

# Open-vocabulary auto-labels, scored against ground truth (about 35 min).
autolabel: $(VENV)/.dev $(VENV)/.view $(VENV)/.ml
	$(PY) -m sardrone.fo.autolabel

# ---- training and evaluation (Phase 3) ---------------------------------------

# Load the real test set from data/real/sard/SARD into FiftyOne.
sard: $(VENV)/.dev $(VENV)/.view
	$(PY) -m sardrone.fo.real

# Saved views train-v1 / val-v1 -> a YOLO dataset in runs/exports/$(EXPORT).
EXPORT ?= v1
export: $(VENV)/.dev $(VENV)/.view
	$(PY) -m sardrone.train.export --name $(EXPORT)

# Ultralytics is installed without its own OpenCV, because FiftyOne already
# ships one (two OpenCV packages in one environment overwrite each other).
$(VENV)/.yolo: | $(VENV)/.ml
	$(PY) -m pip install --quiet --no-deps ultralytics ultralytics-thop polars polars-runtime-32
	touch $@

# Two minutes: does export -> train -> predict -> score run at all?
smoke-train: $(VENV)/.dev $(VENV)/.view $(VENV)/.yolo
	$(PY) -m sardrone.train.smoke

VERSION ?= v1
CHANGE  ?= baseline: simulation only
train: $(VENV)/.dev $(VENV)/.view $(VENV)/.yolo
	$(PY) -m sardrone.train.train --export $(EXPORT) --version $(VERSION)

evaluate: $(VENV)/.dev $(VENV)/.view $(VENV)/.yolo
	$(PY) -m sardrone.train.evaluate --version $(VERSION) --change "$(CHANGE)"

# Map of simulated vs real frames, and how separable the two are.
gapmap: $(VENV)/.dev $(VENV)/.view $(VENV)/.ml
	$(PY) -m sardrone.fo.gapmap

# ---- the loop with a retraining gate (Phase 3E) -------------------------------

# Dry run by default; `make pipeline RUN=1 VERSION=v5` really runs it.
RUN ?=
pipeline: $(VENV)/.dev $(VENV)/.view $(VENV)/.ml
	$(PY) -m sardrone.pipeline --version $(VERSION) --max-train 6000 $(if $(RUN),--run,)

# ---- analysis after evaluation -------------------------------------------------

# Precision-recall curves for every evaluated version -> docs/images/pr_curves.png
curves: $(VENV)/.dev $(VENV)/.view
	$(PY) -m sardrone.train.curves

# Mark a model's false alarms; contact sheet of its most confident ones on real photos.
false-alarms: $(VENV)/.dev $(VENV)/.view
	$(PY) -m sardrone.fo.false_alarms --version $(VERSION)

# A second real test set from a Pascal VOC folder (HERIDAL: see docs/DETAILS.md).
REAL_NAME ?= heridal
REAL_ROOT ?= data/real/heridal
real-voc: $(VENV)/.dev $(VENV)/.view
	$(PY) -m sardrone.fo.voc --name $(REAL_NAME) --root $(REAL_ROOT)

# Score model versions on that set, at their scorecard cut-offs.
VERSIONS ?= v0 v4
extra-real: $(VENV)/.dev $(VENV)/.view $(VENV)/.yolo
	$(PY) -m sardrone.train.extra_real --dataset $(REAL_NAME) --versions $(VERSIONS)

# ---- mission agent (Phase 4A) -------------------------------------------------

# 20 written missions against the mission simulator. MODEL=rule-based is free;
# MODEL=cohere calls the Cohere API (needs COHERE_API_KEY and COHERE_MODEL in .env).
MODEL ?= rule-based
agent-eval: $(VENV)/.dev
	$(PY) -m sardrone.agent.evaluate --model rule-based $(if $(filter cohere,$(MODEL)),--model cohere,)

# One mission's agent decisions as an episode with /events (MISSION=m02).
MISSION ?= m02
agent-record: $(VENV)/.dev
	$(PY) -m sardrone.agent.record --mission $(MISSION) --model $(MODEL)

view: $(VENV)/.dev $(VENV)/.view
	VFF_MULTIMODAL=1 $(PY) -m sardrone.fo.load

clean:
	rm -rf $(VENV) .pytest_cache *.egg-info
