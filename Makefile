install:
	uv sync

run:
	uv run python3 -m src

debug:
	python -m debugpy 

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache

lint:
	flake8 .
	mypy . --warn-return-any \
	--warn-unused-ignores \
	--ignore-missing-imports \
	--disallow-untyped-defs \
	--check-untyped-defs

lint-strict:
	flake8 .
	mypy --strict .
