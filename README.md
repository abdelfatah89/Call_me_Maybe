*This project has been created as part of the 42 curriculum by alaktaou.*

# Call me Maybe — LLM-Based Function Calling with Constrained Decoding

## Description

**Call me Maybe** is a function-calling engine that uses a small, locally-running language model ([Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)) to translate natural language requests into structured JSON function calls.

Given a set of function definitions and a list of user prompts, the program:

1. Constructs a carefully engineered system prompt containing the available functions, the required output schema, and the user's request.
2. Feeds the prompt to the Qwen3-0.6B causal language model and performs **greedy token-by-token decoding**.
3. Monitors the generated stream with a **`JsonStopDetector`** — a streaming brace/bracket depth tracker that halts generation the instant a syntactically complete JSON object is formed.
4. Validates the result against the expected schema and, on failure, retries up to 3 times by feeding the error back into the prompt so the model can self-correct.
5. Collects all validated results and writes them to `data/output/function_calling_results.json`.

The project demonstrates that even a 0.6 B-parameter model can reliably produce correctly structured JSON when guided by prompt engineering, constrained stop detection, and a retry loop — without any fine-tuning or external API.

### Available Functions

| Function | Description | Parameters |
|---|---|---|
| `fn_add_numbers` | Add two numbers together | `a: number`, `b: number` |
| `fn_greet` | Generate a greeting message | `name: string` |
| `fn_reverse_string` | Reverse a string | `s: string` |
| `fn_get_square_root` | Calculate the square root | `a: number` |
| `fn_substitute_string_with_regex` | Regex-based string replacement | `source_string`, `regex`, `replacement` |

## Instructions

### Prerequisites

- **Python ≥ 3.10**
- **[uv](https://docs.astral.sh/uv/)** — fast Python package manager (install with `pip install uv`)
- ~1.2 GB of free disk space for the Qwen3-0.6B model weights (downloaded automatically on first run)
- Internet access on first run (to fetch model files from HuggingFace Hub; cached afterwards)

### Installation

```bash
# Clone the repository
git clone https://github.com/abdelfatah89/Call_me_Maybe.git
cd Call_me_Maybe

# Install all dependencies
make install        # equivalent to: uv sync
```

This creates a `.venv` virtual environment and installs PyTorch, Transformers, Pydantic, and all other dependencies defined in `pyproject.toml`.

### Running the Program

```bash
# Run the full pipeline with the default test suite
make run            # equivalent to: uv run python3 -m src
```

The program reads prompts from `data/input/function_calling_tests.json`, processes each one through the model, and writes the results to `data/output/function_calling_results.json`.

#### Custom Input/Output Paths

```bash
uv run python3 -m src \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json \
    --functions-definition data/input/functions_definition.json
```

### Linting

```bash
make lint           # flake8 + mypy (standard mode)
make lint-strict    # flake8 + mypy --strict
```

### Other Make Targets

```bash
make clean          # remove __pycache__, .mypy_cache, .pytest_cache
make debug          # launch pdb debugger
```

## Algorithm Explanation

### Constrained Decoding Approach

The core algorithm is a **prompt-constrained greedy decoding loop** with structural stop detection. Unlike fine-tuned or tool-use–specific models, this approach relies entirely on inference-time techniques to coerce a general-purpose LM into producing valid JSON.

#### Step 1 — Prompt Construction (`Generator.json`)

A detailed system prompt is assembled containing:

- The full list of available function definitions (names, descriptions, parameter types).
- The exact JSON schema the output must follow (`output_schema.json`).
- 12 explicit rules the model must obey (valid JSON only, no markdown, correct types, etc.).
- A validation checklist the model should mentally run before responding.
- The user's natural language request.
- A pre-filled opening brace `{` to prime the model into JSON-generation mode.

The pre-filled `{` is a key technique: by starting the model's continuation inside a JSON object, it strongly biases the output distribution toward JSON key-value pairs from the very first token.

#### Step 2 — Token-by-Token Greedy Decoding (`ask_model`)

```
input_ids = encode(prompt)
for each step (up to MAX_TOKENS = 256):
    logits = model.get_logits_from_input_ids(input_ids)
    next_token = argmax(logits)            # greedy selection
    decoded_text = vocab[next_token]       # raw token → text
    result += decoded_text
    input_ids.append(next_token)
    if JsonStopDetector.feed(decoded_text):
        break                              # complete JSON object detected
```

The decoding operates directly on raw logits via `numpy.argmax` — no sampling, no temperature, no top-k/top-p. This deterministic strategy maximizes reproducibility and is well-suited for structured output where creativity is undesirable.

#### Step 3 — Structural Stop Detection (`JsonStopDetector`)

The `JsonStopDetector` is a streaming state machine that processes each character as it arrives:

- Tracks `object_depth` (incremented on `{`, decremented on `}`).
- Tracks `array_depth` (incremented on `[`, decremented on `]`).
- Handles **string literals** correctly — braces inside `"..."` do not affect depth counters.
- Handles **escape sequences** — a `\"` inside a string does not close the string.
- Returns `True` the moment both depths return to zero, meaning a complete top-level JSON object has been emitted.
- Raises `JsonStopDetectorError` on structural violations (e.g. unmatched closing braces).

This is critical: without it, the model would continue generating text past the JSON boundary, producing garbage or a second unwanted object.

#### Step 4 — Schema Validation and Retry Loop

After stop detection halts generation, two validation layers run:

1. **Schema key check** (`_validate_schema`): Parses the JSON and verifies every key from `output_schema.json` is present.
2. **Pydantic model validation** (`_validate_output`): Deserializes into `OutputModel(prompt, name, parameters)` for type checking.

If either fails, the error message is injected into the next prompt attempt:

```
"The previous output was invalid: <error message>"
```

The model gets up to `MAX_TRIES = 3` attempts per prompt. This self-correction mechanism lets the model learn from its own mistakes within a single session.

### Token Handling

The Qwen tokenizer uses special Unicode markers for whitespace: `Ġ` represents a space and `Ċ` represents a newline. The `replace_char` function normalizes these back to standard characters before they enter the result and the stop detector.

## Design Decisions

### Why Qwen3-0.6B?

A 0.6 B-parameter model was chosen to balance capability against resource requirements. It is small enough to run on a laptop CPU (no GPU required) while still being capable of following structured instructions. Larger models would improve accuracy but at the cost of significantly higher memory and inference time.

### Why Greedy Decoding Instead of Sampling?

For function calling, the output must be **deterministic and syntactically correct**. Sampling (temperature > 0) introduces randomness that can break JSON structure — a misplaced comma or unclosed quote is all it takes. Greedy argmax is the safest strategy for constrained generation.

### Why a Streaming Stop Detector Instead of Post-Hoc Parsing?

Parsing the full output after generation would waste tokens. The streaming approach halts generation immediately upon JSON completion, saving compute and avoiding trailing garbage that could confuse downstream parsing.

### Why Prompt Engineering Over Fine-Tuning?

Fine-tuning requires training data, GPU resources, and ongoing maintenance. Prompt engineering achieves comparable results for structured output tasks while being immediately adaptable — changing the function list or schema requires no retraining, just editing JSON files.

### Why Pydantic for Validation?

Pydantic provides declarative, type-safe validation with clear error messages. The `FunctionModel`, `InputModel`, and `OutputModel` classes enforce the contract at both input parsing and output verification stages, catching issues early with informative errors.

### Architecture: Separation of Concerns

| Module | Responsibility |
|---|---|
| `llm_sdk/` | Model loading, tokenization, raw logit extraction (reusable SDK) |
| `src/constrained.py` | Orchestration: wires model, generator, and stop detector together |
| `src/generator.py` | Prompt construction and schema validation |
| `src/json_stop_detector.py` | Streaming JSON completeness detection |
| `src/models.py` | Pydantic data models for input/output validation |
| `src/__main__.py` | CLI entry point, file I/O, error handling |

## Performance Analysis

### Accuracy

On the included 11-prompt test suite, the model successfully maps each natural language request to the correct function with the correct parameters. The results cover:

- Arithmetic (`fn_add_numbers` with integer arguments)
- String operations (`fn_greet`, `fn_reverse_string`)
- Numeric computation (`fn_get_square_root`)
- Regex-based substitution (`fn_substitute_string_with_regex` with varied patterns)

The retry mechanism handles occasional malformed outputs — in practice, most prompts succeed on the first attempt, with the retry loop acting as a safety net.

### Speed

| Hardware | Model Load | Per-Prompt Inference | Full Suite (11 prompts) |
|---|---|---|---|
| CPU (float32) | ~5–10 s | ~30–120 s | ~5–20 min |
| GPU / CUDA (float16) | ~3–5 s | ~2–5 s | ~30–60 s |
| Apple MPS (float16) | ~3–5 s | ~5–10 s | ~1–2 min |

The primary bottleneck on CPU is the token-by-token autoregressive loop: each of the up to 256 steps requires a full forward pass through the 0.6 B-parameter model. GPU acceleration provides a roughly 10–20× speedup.

Note: the current implementation reinstantiates the model for each prompt (`ask_model` creates a new `Small_LLM_Model()` per call). Refactoring to share a single model instance across prompts would significantly reduce total runtime.

### Reliability

- **Deterministic**: Greedy decoding produces identical output for identical input.
- **Bounded**: `MAX_TOKENS = 256` and `MAX_TRIES = 3` prevent runaway generation.
- **Fail-safe**: If all retries fail, the prompt is silently skipped rather than crashing. If no valid outputs are produced, the program returns `None` and writes nothing.

## Challenges Faced

### 1. Stopping Generation at the Right Point

**Problem**: Without explicit control, the model generates text beyond the JSON boundary, appending explanations or a second object.

**Solution**: The `JsonStopDetector` state machine tracks brace/bracket depth character-by-character, halting generation the instant depth returns to zero. Special care was needed to avoid false positives from braces inside string literals (handled by the `in_string` / `escape` state flags).

### 2. Token-to-Text Mapping with Special Characters

**Problem**: The Qwen tokenizer uses Unicode markers (`Ġ` for space, `Ċ` for newline) in its vocabulary. Feeding these directly to the JSON stop detector would break parsing.

**Solution**: The `replace_char` function normalizes tokens back to standard whitespace before they reach either the result buffer or the stop detector.

### 3. Coercing a Small Model to Produce Valid JSON

**Problem**: A 0.6 B model is prone to subtle JSON errors — trailing commas, missing quotes, invented parameter names.

**Solution**: A multi-layered approach:
- **Pre-filling** the opening `{` to prime JSON mode.
- **Explicit 12-rule instruction set** in the prompt.
- **Schema injection** so the model sees the exact expected structure.
- **Retry with error feedback** so the model can self-correct.

### 4. Balancing Prompt Length vs. Context Window

**Problem**: The system prompt (function definitions + rules + schema + user request) consumes a significant portion of the model's context window, leaving fewer tokens for generation.

**Solution**: The output schema is deliberately compact (3 fields: `prompt`, `name`, `parameters`), and `MAX_TOKENS = 256` is generous enough for any single function call while staying within context limits.

## Testing Strategy

### Input Validation

Both input files are validated at startup using Pydantic models:

- `InputModel` ensures each prompt entry has the required `prompt` field.
- `FunctionModel` ensures each function definition has `name`, `description`, `parameters`, and `returns` with correct types.

Invalid input triggers a clear error message and a non-zero exit code.

### Output Validation

Every generated JSON object passes through two layers:

1. **Schema key check**: Verifies all expected keys (`prompt`, `name`, `parameters`) are present.
2. **Pydantic `OutputModel`**: Type-checks the deserialized object.

After all prompts are processed, the aggregated output is validated once more as a complete list before writing to disk.

### Error Handling Coverage

The `__main__.py` entry point catches and categorizes errors:

- `json.JSONDecodeError` → malformed JSON files
- `ValidationError` → schema/type mismatches
- `FileNotFoundError` → missing input files
- Generic `Exception` → unexpected failures

Each category prints a labeled error and exits with code 1.

### Linting

Static analysis ensures code quality:

```bash
make lint       # flake8 (style) + mypy (type checking)
```

mypy runs with `--disallow-untyped-defs` and `--check-untyped-defs` to enforce full type annotations across the codebase.

## Example Usage

### Running with Default Inputs

```bash
make run
```

**Input** (`data/input/function_calling_tests.json`):
```json
[
  { "prompt": "What is the sum of 2 and 3?" },
  { "prompt": "Greet shrek" },
  { "prompt": "Reverse the string 'hello'" },
  { "prompt": "What is the square root of 16?" },
  { "prompt": "Replace all vowels in 'Programming is fun' with asterisks" }
]
```

**Output** (`data/output/function_calling_results.json`):
```json
[
    {
        "prompt": "What is the sum of 2 and 3?",
        "name": "fn_add_numbers",
        "parameters": { "a": 2, "b": 3 }
    },
    {
        "prompt": "Greet shrek",
        "name": "fn_greet",
        "parameters": { "name": "shrek" }
    },
    {
        "prompt": "Reverse the string 'hello'",
        "name": "fn_reverse_string",
        "parameters": { "s": "hello" }
    },
    {
        "prompt": "What is the square root of 16?",
        "name": "fn_get_square_root",
        "parameters": { "a": 16 }
    },
    {
        "prompt": "Replace all vowels in 'Programming is fun' with asterisks",
        "name": "fn_substitute_string_with_regex",
        "parameters": {
            "source_string": "Programming is fun",
            "regex": "([aeiouAEIOU])",
            "replacement": "*"
        }
    }
]
```

### Running with Custom Files

```bash
uv run python3 -m src \
    --input my_prompts.json \
    --functions-definition my_functions.json \
    --output my_results.json
```

Expected output (token-by-token generation):
```
write a json schema:
{ "name
{ "name":
{ "name": "name
{ "name": "name",
{ "name": "name", "age
{ "name": "name", "age":
```

## Resources

### References
- [How Large Language Models Work](https://www.youtube.com/watch?v=5sLYAQS9sWQ) — Youtube video
- [Large Language Models explained briefly](https://www.youtube.com/watch?v=LPZh9BOjkQs&t=5s) — Youtube video
- [Transformers, the tech behind LLMs | Deep Learning Chapter 5](https://www.youtube.com/watch?v=wjZofJX0v4M&t=3s) — Youtube video
- [Qwen3-0.6B Model Card](https://huggingface.co/Qwen/Qwen3-0.6B) — the language model used in this project
- [How To Use JSON In Python](https://www.youtube.com/watch?v=-51jxlQaxyA) — Youtube video

### AI Usage

AI assistance (ChatGPT / Cursor) was used during development for the following tasks:

- **Prompt engineering**: Iterating on the system prompt to improve JSON output reliability.
- **Documentation**: Structuring and drafting parts of this README.
- And also understanding all the concepts related to this project.

All code was written, reviewed, and tested by the project author. AI was not used to generate the core algorithm or architecture.
