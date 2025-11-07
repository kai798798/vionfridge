SHELL := /bin/sh
PY ?= python

.PHONY: install classify_pic count_pic count_vid clean

install: requirements.txt
	$(PY) -m pip install -r requirements.txt

classify_pic:
	@name="$(word 2,$(MAKECMDGOALS))"; \
	if [ -z "$$name" ]; then echo "Usage: make classify_pic <basename|path>"; exit 1; fi; \
	if [ -f "$$name" ]; then RES="$$name"; else RES=$$(ls -1 "data/samples/$$name".* 2>/dev/null | head -n1); fi; \
	[ -n "$$RES" ] || { echo "File not found for '$$name' (looked in data/samples/)"; exit 2; }; \
	echo "Classifying image: $$RES"; \
	$(PY) src/classify_pic.py "$$RES"

count_pic:
	@name="$(word 2,$(MAKECMDGOALS))"; \
	if [ -z "$$name" ]; then echo "Usage: make count_pic <basename|path> [FOCUS=label] [THRESHOLD=0.25] [JSON=1]"; exit 1; fi; \
	if [ -f "$$name" ]; then RES="$$name"; else RES=$$(ls -1 "data/samples/$$name".* 2>/dev/null | head -n1); fi; \
	[ -n "$$RES" ] || { echo "File not found for '$$name' (looked in data/samples/)"; exit 2; }; \
	json_opt=""; [ "$$JSON" = "1" ] && json_opt="--json"; \
	focus_opt=""; [ -n "$$FOCUS" ] && focus_opt="--focus $$FOCUS"; \
	th_opt=""; [ -n "$$THRESHOLD" ] && th_opt="--threshold $$THRESHOLD"; \
	echo "Counting (image): $$RES"; \
	$(PY) src/count_pic.py "$$RES" $$focus_opt $$json_opt $$th_opt

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

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache

%::
	@: