SHELL := /bin/sh
PY ?= python3

.PHONY: install classify_pic count_pic count_vid clean

install: requirements.txt
	$(PY) -m pip install -r requirements.txt

count_vid:
	@name="$(word 2,$(MAKECMDGOALS))"; \
	if [ -z "$$name" ]; then echo "Usage: make count_vid <basename|path> [MODE=in|net] [THRESHOLD=0.25] [SHOW=1] [JSON=1] [ZONE=...]"; exit 1; fi; \
	if [ -f "$$name" ]; then RES="$$name"; else RES=$$(ls -1 "data/samples/$$name".* 2>/dev/null | head -n1); fi; \
	[ -n "$$RES" ] || { echo "File not found for '$$name' (looked in data/samples/)"; exit 2; }; \
	mode_opt=""; [ -n "$$MODE" ] && mode_opt="--mode $$MODE"; \
	json_opt=""; [ "$$JSON" = "1" ] && json_opt="--json"; \
	show_opt=""; [ "$$SHOW" = "1" ] && show_opt="--show"; \
	conf_opt=""; [ -n "$$THRESHOLD" ] && conf_opt="--conf $$THRESHOLD"; \
	zone_opt=""; [ -n "$$ZONE" ] && zone_opt="--zone $$ZONE"; \
	lbl_opt=""; [ -n "$$LABELS" ] && lbl_opt="--labels $$LABELS"; \
	echo "Counting (video): $$RES"; \
	$(PY) -u src/count_video.py "$$RES" $$lbl_opt $$mode_opt $$conf_opt $$json_opt $$show_opt $$zone_opt

percep:
	@name="$(word 2,$(MAKECMDGOALS))"; \
	if [ -z "$$name" ]; then \
		echo "Usage: make percep <basename|path> [SHOW=1]"; \
		exit 1; \
	fi; \
	if [ -f "$$name" ]; then \
		RES="$$name"; \
	else \
		RES=$$(ls -1 "data/samples/$$name".* 2>/dev/null | head -n1); \
	fi; \
	[ -n "$$RES" ] || { \
		echo "File not found for '$$name' (looked in data/samples/)"; \
		exit 2; \
	}; \
	show_opt=""; [ "$$SHOW" = "1" ] && show_opt="--show"; \
	echo "Testing perception on: $$RES"; \
	$(PY) -u src/test_perception.py "$$RES" $$show_opt

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache

%::
	@: