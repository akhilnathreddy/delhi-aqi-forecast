# Full pipeline: `make all` (≈7 h on a 4-core CPU; downloads ≈1 h of that).
PY ?= python -m aqf.cli

.PHONY: data build train report test lint all

data:
	$(PY) download-stations
	$(PY) download-gfs
	$(PY) download-fires

build:
	$(PY) build

train:
	$(PY) train --fold main
	$(PY) train --fold winter

report:
	$(PY) report

test:
	pytest -q

lint:
	ruff check src tests

all: data build train report
