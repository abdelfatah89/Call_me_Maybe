# AGENTS.md

## Cursor Cloud specific instructions

### Overview
This is a Python-based LLM function-calling SDK that uses a local Qwen3-0.6B model to generate structured JSON responses from natural language prompts. It is a single-service, self-contained application with no external databases or APIs required at runtime.

### Quick reference
- **Install deps:** `make install` (runs `uv sync`)
- **Lint:** `make lint` (flake8 + mypy)
- **Run full pipeline:** `make run` (runs `uv run python3 -m src`)
- **Quick smoke test:** `uv run python3 test.py` (loads model and generates ~10 tokens)

### Important caveats
- The full pipeline (`make run`) runs the Qwen3-0.6B model on CPU in this environment. Each of the 11 test prompts instantiates a new model and performs token-by-token greedy decoding, so a complete run takes a long time (potentially 30+ minutes on CPU). For quick validation, use `test.py` instead.
- The model weights (~1.2 GB) are downloaded from HuggingFace Hub on first run and cached in `~/.cache/huggingface/`. Subsequent runs use the cache.
- `uv` must be installed (`pip install uv`) before `make install` will work. The update script handles this.
- The `llm_sdk/` package is excluded from both flake8 and mypy checks (see `.flake8` and `mypy.ini`).
