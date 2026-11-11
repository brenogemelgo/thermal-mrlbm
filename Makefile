PYTHON ?= python3
PREFIX ?= /usr/local
RBC_ARGS ?= --case 1 --grid 512 --rr 3

.PHONY: all solver converter install-converter
all: solver

# The solver launcher builds both the specialized CUDA executable and convertVTI.
solver:
	$(PYTHON) scripts/run_rbc.py $(RBC_ARGS) --build-only

converter:
	$(PYTHON) scripts/build_convert_vti.py

install-converter: converter
	install -d "$(DESTDIR)$(PREFIX)/bin"
	install -m 755 build/convertVTI "$(DESTDIR)$(PREFIX)/bin/convertVTI"
