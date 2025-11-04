# Makefile
SHELL := /bin/bash
PY ?= python
PIP ?= pip

.PHONY: install compile dev-install classify run detect detect-run count count-run test format lint clean

install: requirements.txt
	$(PY) -m pip install -r requirements.txt

compile:
	@command -v pip-compile >/dev/null || { echo "pip-compile not found. Run: pip install pip-tools"; exit 1; }
	pip-compile --upgrade -o requirements.txt requirements.in

dev-install: requirements.txt
	$(PY) -m pip install -r requirements.txt

# ---- Single-image classification (CLIP) ----
# Usage:
#   make classify <basename|path>
# Examples:
#   make classify banana01              # looks in data/samples/banana01.*
#   make classify data/samples/banana01.jpg
classify:
	@if [ -z "$(word 2,$(MAKECMDGOALS))" ]; then \
	  echo "Usage: make classify <basename|path>"; exit 1; \
	fi; \
	name="$(word 2,$(MAKECMDGOALS))"; \
	if [ -f "$$name" ]; then img="$$name"; \
	else img=$$(ls -1 "data/samples/$$name".* 2>/dev/null | head -n1); fi; \
	if [ -z "$$img" ]; then \
	  echo "Image not found for '$$name' (looked in data/samples/)"; exit 2; \
	fi; \
	$(PY) src/classify_food.py "$$img"

# Explicit run with variables:
#   make run IMG=path/to/image.jpg TOPK=3
run:
	@if [ -z "$(IMG)" ]; then echo "Usage: make run IMG=path/to/image.jpg [TOPK=N]"; exit 1; fi; \
	topk=$${TOPK:-1}; \
	$(PY) src/classify_food.py "$(IMG)" --topk $$topk

# ---- Multi-object detection (OWL-ViT or YOLO) ----
# Usage:
#   make detect <basename|path>
detect:
	@if [ -z "$(word 2,$(MAKECMDGOALS))" ]; then \
	  echo "Usage: make detect <basename|path>"; exit 1; \
	fi; \
	name="$(word 2,$(MAKECMDGOALS))"; \
	if [ -f "$$name" ]; then img="$$name"; \
	else img=$$(ls -1 "data/samples/$$name".* 2>/dev/null | head -n1); fi; \
	[ -n "$$img" ] || { echo "Image not found for '$$name'"; exit 2; }; \
	$(PY) src/detect_foods.py "$$img"

# Explicit detect with variables:
#   make detect-run IMG=path/to/image.jpg THRESHOLD=0.3
detect-run:
	@if [ -z "$(IMG)" ]; then echo "Usage: make detect-run IMG=path [THRESHOLD=0.25]"; exit 1; fi; \
	if [ -n "$(THRESHOLD)" ]; then \
	  TH_OPT="--threshold $(THRESHOLD)"; \
	else TH_OPT=""; fi; \
	$(PY) src/detect_foods.py "$(IMG)" $$TH_OPT

# ---- Counting by label (detect + tally) ----
# Usage:
#   make count <basename|path> [FOCUS=banana]
count:
	@if [ -z "$(word 2,$(MAKECMDGOALS))" ]; then \
	  echo "Usage: make count <basename|path> [FOCUS=banana]"; exit 1; \
	fi; \
	name="$(word 2,$(MAKECMDGOALS))"; \
	if [ -f "$$name" ]; then img="$$name"; \
	else img=$$(ls -1 "data/samples/$$name".* 2>/dev/null | head -n1); fi; \
	[ -n "$$img" ] || { echo "Image not found for '$$name'"; exit 2; }; \
	if [ -n "$(FOCUS)" ]; then \
	  $(PY) src/detect_and_count.py "$$img" --focus "$(FOCUS)"; \
	else \
	  $(PY) src/detect_and_count.py "$$img"; \
	fi

# Explicit count with variables:
#   make count-run IMG=path/to/image.jpg FOCUS=banana JSON=1
count-run:
	@if [ -z "$(IMG)" ]; then echo "Usage: make count-run IMG=path [FOCUS=label] [JSON=1]"; exit 1; fi; \
	json_opt=""; [ "$(JSON)" = "1" ] && json_opt="--json"; \
	focus_opt=""; [ -n "$(FOCUS)" ] && focus_opt="--focus '$(FOCUS)'"; \
	eval $(PY) src/detect_and_count.py "$(IMG)" $$focus_opt $$json_opt

# ---- Quality-of-life ----
test:
	$(PY) -m pytest -q

format:
	$(PY) -m black src tests

lint:
	$(PY) -m ruff check src tests

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache

# Swallow extra words like 'banana01' so make doesn't treat them as targets
%::
	@: