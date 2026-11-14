# Everything runs inside the local virtual env, so you never need to "activate" it.
PYTHON_BOOTSTRAP ?= python3.12
VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(PY) -m pip install --quiet

.PHONY: venv test synthetic record record-fake view clean

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

# The real thing. Start PX4 first:  cd PX4-Autopilot && make px4_sitl gz_x500
record: $(VENV)/.dev $(VENV)/.fly
	$(PY) -m sardrone.sim.record_flight

# sardrone.fo.load sets VFF_MULTIMODAL=1 itself, before importing fiftyone;
# it is repeated here so the requirement is visible.
view: $(VENV)/.dev $(VENV)/.view
	VFF_MULTIMODAL=1 $(PY) -m sardrone.fo.load

clean:
	rm -rf $(VENV) .pytest_cache *.egg-info
