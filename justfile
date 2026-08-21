format:
     uv run ruff format

lint:
     uv run ruff check --fix

test: 
     uv run pytest -v --color=yes

test-cov:
     uv run pytest -vv --color=yes --cov facts_total

validate: format lint test test-cov