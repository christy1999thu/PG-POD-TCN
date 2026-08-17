.PHONY: install test lint demo clean

install:
	python -m pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests

demo:
	pg-pod-tcn demo --config configs/demo.yaml

clean:
	python -c "from pathlib import Path; import shutil; [shutil.rmtree(p) for p in (Path('outputs'), Path('data/synthetic')) if p.exists()]"

