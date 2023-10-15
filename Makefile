SHELL=/bin/bash

venv:  ## Set up virtual environment
	python3 -m venv venv
	venv/bin/pip install -r requirements.txt

install: venv
	unset CONDA_PREFIX && \
	source venv/bin/activate && maturin develop -m levenshtein_lib/Cargo.toml

install-release: venv
	unset CONDA_PREFIX && \
	source venv/bin/activate && maturin develop --release -m levenshtein_lib/Cargo.toml

clean:
	-@rm -r venv
	-@cd levenshtein_lib && cargo clean

run: install
	source venv/bin/activate && python run.py

run-release: install-release
	source venv/bin/activate && python run.py
