*This project has been created as part of the 42 curriculum by alaktaou.*

# Call me Maybe — LLM-Based Function Calling with Constrained Decoding

## Description

**Call me Maybe** is a function-calling engine that uses a locally-running language model ([Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)) to translate natural language requests into structured JSON function calls.

Given a set of function definitions (`data/input/functions_definition.json`) and a list of user prompts (`data/input/function_calling_tests.json`), the program:

1. Constructs a system prompt containing the available functions, the required output schema, and the user's request.
2. Feeds the prompt to the Qwen3-0.6B causal language model and performs greedy token-by-token decoding using `numpy.argmax` on the raw logits.
3. Monitors the generated token stream with a `JsonStopDetector` that tracks brace/bracket depth and halts generation once a syntactically complete JSON object is formed.
4. Appends the result to the output list. After processing all prompts, validates the aggregated output list with Pydantic's `OutputModel` (`src/validator.py` — `Validator.v_output`) and writes the result to `data/output/function_calling_results.json` (`src/__main__.py` — `main`).

### Project Structure

```
.
├── llm_sdk/
│   └── __init__.py          # Small_LLM_Model class — model loading, tokenization, logit extraction
├── src/
│   ├── __main__.py           # CLI entry point, file I/O, error handling
│   ├── constrained.py        # Orchestration: generate_model_output() + constrained()
│   ├── json_stop_detector.py # JsonStopDetector — streaming JSON completeness detection
│   ├── llm_model.py          # CostimizedModel — tokenization, logit extraction, generation
│   ├── models.py             # Pydantic models: FunctionModel, InputModel, OutputModel
│   ├── prompt.py             # get_instructions() — prompt/instruction construction
│   └── validator.py          # Validator — input/output schema validation
├── data/
│   ├── input/
│   │   ├── function_calling_tests.json   # 11 test prompts
│   │   └── functions_definition.json     # 5 function definitions
│   └── output/
│       └── function_calling_results.json # Generated output
├── output_schema.json        # Expected JSON output structure
├── test.py                   # Smoke test — model load + 10-step greedy decode
├── Makefile                  # install, run, lint, lint-strict, clean, debug
└── pyproject.toml            # Dependencies and build config (uv/hatch)
```

### Available Functions

Defined in `data/input/functions_definition.json`:

| Function | Description | Parameters |
|---|---|---|
| `fn_add_numbers` | Add two numbers together and return their sum. | `a: number`, `b: number` |
| `fn_greet` | Generate a greeting message for a person by name. | `name: string` |
| `fn_reverse_string` | Reverse a string and return the reversed result. | `s: string` |
| `fn_get_square_root` | Calculate the square root of a number. | `a: number` |
| `fn_substitute_string_with_regex` | Replace all occurrences matching a regex pattern in a string. | `source_string: string`, `regex: string`, `replacement: string` |

### Output Schema

Defined in `output_schema.json`, every generated result must contain:

```json
{
    "prompt": "user request",
    "name": "function-name",
    "parameters": "object"
}
```

## Instructions

### Prerequisites

- **Python ≥ 3.10** (specified in `pyproject.toml`)
- **[uv](https://docs.astral.sh/uv/)** — Python package manager (install with `pip install uv`)
- Internet access on first run to download model weights from HuggingFace Hub (cached in `~/.cache/huggingface/` afterwards)

### Installation

```bash
git clone <repository_url> Call_me_Maybe
cd Call_me_Maybe

make install        # runs: uv sync
```

This creates a `.venv` virtual environment and installs all dependencies defined in `pyproject.toml`:

- `torch>=2.0.0`
- `transformers>=4.40.0`
- `huggingface-hub>=0.20.0`
- `accelerate>=1.13.0`
- `pydantic>=2.13.2`
- `protobuf>=7.34.1`
- `sentencepiece>=0.2.1`
- `flake8>=7.3.0`
- `mypy>=1.20.2`

### Running the Program

```bash
make run            # runs: uv run python3 -m src
```

This reads prompts from `data/input/function_calling_tests.json`, processes each one through the model, and writes results to `data/output/function_calling_results.json`.

#### Custom Input/Output Paths

The CLI accepts three arguments (defined in `src/__main__.py`):

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

`make lint` runs mypy with `--warn-return-any`, `--warn-unused-ignores`, `--ignore-missing-imports`, `--disallow-untyped-defs`, and `--check-untyped-defs`.

The `llm_sdk/` directory is excluded from both flake8 (`.flake8`) and mypy (`mypy.ini`).

### Other Make Targets

```bash
make clean          # rm -rf __pycache__ */__pycache__ .mypy_cache .pytest_cache
make debug          # python -m pdb src
```

## Algorithm Explanation

### Constrained Decoding Approach

The algorithm is a greedy decoding loop with structural stop detection. It uses prompt engineering to produce valid JSON from the model.

#### Step 1 — Prompt Construction (`src/prompt.py` — `get_instructions`)

The `get_instructions` function assembles a system prompt containing:

- A preamble instructing the model to act as a function-calling assistant and return one JSON object matching the output schema.
- The output schema injected via `json.dumps(schema)`.
- A key-order hint: `prompt`, `name`, `parameters`.
- The full list of available function signatures derived from the `funcs` parameter (name + typed parameter list).
- A note that `'prompt'` must be the exact user request.
- A closing `"User request:"` label, after which the individual prompt and a pre-filled `{` are appended at generation time.

#### Step 2 — Token-by-Token Greedy Decoding (`src/llm_model.py` — `CostimizedModel.generate`)

The `generate` method:

1. Uses the `Small_LLM_Model` instance created once in `CostimizedModel.__init__` (not re-instantiated per call).
2. Initialises `result = "{"` and pre-feeds `"{"` into the `JsonStopDetector` (setting `object_depth = 1`).
3. Encodes the per-prompt suffix `f"\n{prompt}\nResult: {{"` via `self.encode()` and prepends the pre-computed `instructions_tokens` to form the full `tokens` list.
4. Runs a loop for up to `MAX_TOKENS = 96` iterations:
   - Calls `self.get_logits(tokens)` to get raw logits for the next token.
   - Applies a `_get_valid_mask` to suppress structurally invalid tokens.
   - Selects the next token via `int(np.argmax(logits))` (greedy — no sampling, no temperature).
   - Decodes the token string and normalises it (`Ġ` → space, `Ċ` → newline) inline.
   - Appends the decoded text to `result` and the token ID to `tokens`.
   - Feeds the decoded text to `detector.feed()`. If it returns `True` (complete JSON object), breaks.
5. Returns the accumulated `result` string.

#### Step 3 — Structural Stop Detection (`src/json_stop_detector.py` — `JsonStopDetector`)

`JsonStopDetector` is a character-by-character state machine with the following state:

- `started: bool` — whether the opening `{` has been seen.
- `in_string: bool` — whether the parser is inside a `"..."` string literal.
- `escape: bool` — whether the previous character was `\` (to handle `\"` correctly).
- `object_depth: int` — incremented on `{`, decremented on `}`.
- `array_depth: int` — incremented on `[`, decremented on `]`.
- `buffer: List[str]` — accumulates all characters.

The `feed` method processes each character of the input text:

- Skips whitespace before the first `{`. Raises `JsonStopDetectorError` if a non-whitespace, non-`{` character appears first.
- While `in_string` is `True`, only tracks escape sequences and the closing `"`.
- Outside strings, updates `object_depth` and `array_depth`.
- Raises `JsonStopDetectorError` if `object_depth` or `array_depth` goes below zero.
- Returns `True` when `started` is `True`, `in_string` is `False`, and both depths are zero.

#### Step 4 — Orchestration (`src/constrained.py` — `constrained`)

The `constrained` function:

1. Calls `get_instructions(funcs)` once to build the shared system instructions and encodes them into `instructions_tokens` via `model.encode()`.
2. Iterates over all prompts, calling `generate_model_output()` for each, which instantiates a fresh `JsonStopDetector`, calls `model.generate()`, and appends the result string to `output_list`.
3. Returns `output_list` to `__main__.py`, which joins the results and validates them.

### Token Handling (`src/llm_model.py` — `CostimizedModel.generate`)

The Qwen tokenizer's vocabulary uses `Ġ` (Unicode U+0120) to represent a space and `Ċ` (Unicode U+010A) to represent a newline. Inside the `generate` method, each decoded token is normalised inline:

```python
next_word = next_word.replace("Ġ", " ").replace("Ċ", "\n").replace('\\', '\\\\')
```

This normalization is applied before the text reaches the `JsonStopDetector` and the result buffer. Without it, the stop detector would receive `Ġ` and `Ċ` characters instead of actual whitespace, and the resulting JSON would contain these Unicode markers.

## Design Decisions

### Model: Qwen/Qwen3-0.6B

The default model is set in `llm_sdk/__init__.py` as `model_name: str = "Qwen/Qwen3-0.6B"`. The `Small_LLM_Model` class auto-selects the compute device with priority `mps > cuda > cpu` and sets `dtype` to `float16` on GPU/MPS or `float32` on CPU. The model is set to `eval()` mode and all parameter gradients are disabled (`requires_grad = False`).

### Greedy Decoding

The decoding loop in `CostimizedModel.generate` uses `int(np.argmax(logits))` — a single deterministic token selection per step. There is no temperature, sampling, or top-k/top-p filtering in the codebase. This produces identical output for identical input.

### Streaming Stop Detection

The `JsonStopDetector` halts generation token-by-token as soon as a complete JSON object is detected, rather than generating a fixed number of tokens and parsing afterwards. The `generate` loop checks `detector.feed(next_word)` after each token and breaks immediately on `True`.

### Prompt Pre-filling

The instructions built by `get_instructions` end with `"User request:"`, and `CostimizedModel.generate` appends `f"\n{prompt}\nResult: {{"` to the encoded tokens. The method also initialises `result = "{"` and calls `detector.feed("{")`. This means the model's first generated token continues from inside an already-opened JSON object.

### Pydantic Validation

Three Pydantic `BaseModel` classes in `src/models.py` enforce data contracts:

- `FunctionModel(name: str, description: str, parameters: dict[str, ParameterModel], returns: ParameterModel)` — validates `functions_definition.json` entries.
- `InputModel(prompt: str)` — validates `function_calling_tests.json` entries.
- `OutputModel(prompt: str, name: str, parameters: dict[str, Any])` — validates final output entries.

### Model Instantiation

`CostimizedModel` is instantiated once in `__main__.py` and passed through to all calls. The underlying `Small_LLM_Model` is created a single time in `CostimizedModel.__init__`, so the model is loaded only once per program run regardless of the number of prompts processed.

## Performance Analysis

### Device Selection

The `Small_LLM_Model.__init__` method (in `llm_sdk/__init__.py`) selects compute device automatically:

```python
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
```

On CUDA, the model uses `device_map="auto"` for automatic layer distribution. On all other devices, layers are placed manually via `self._model.to(self._device)`.

Precision is `float16` on `cuda` and `mps`, `float32` on `cpu`.

### Decoding Bounds

- `MAX_TOKENS = 96` (in `src/llm_model.py`) — maximum tokens generated per prompt before the loop exits.

### Output Behavior

- After all prompts are processed, the results are joined as `"[" + ", ".join(output_list) + "]"` and passed to `Validator.v_output()`.
- If `v_output` returns `False`, nothing is written to disk.

### Speed

| Hardware | Full Suite (11 prompts) |
|---|---|
| CPU (float32) | ~5 min |
| GPU / CUDA (float16) | ~40 s |

## Challenges Faced

### 1. Stopping Generation at the JSON Boundary

The `JsonStopDetector` in `src/json_stop_detector.py` addresses the problem of knowing when a complete JSON object has been emitted. It tracks `object_depth` and `array_depth` through each character. Braces and brackets inside string literals do not affect the counters — the `in_string` and `escape` flags handle this. The detector returns `True` the instant both depths reach zero, and the `ask_model` loop breaks immediately.

### 2. Tokenizer Special Characters

The Qwen tokenizer encodes spaces as `Ġ` and newlines as `Ċ` in its vocabulary. The `generate` method in `src/llm_model.py` normalises these inline on each decoded token before feeding text to the `JsonStopDetector` and the result buffer. Without this normalization, the stop detector would receive `Ġ` and `Ċ` characters instead of actual whitespace, and the resulting JSON would contain these Unicode markers.

### 3. Handling Malformed Model Output

The code addresses malformed output at the structural detection level:

- **`JsonStopDetectorError`**: Raised on structural violations (unmatched braces/brackets, non-`{` start character). The exception propagates up through `generate` and is caught by the top-level handler in `__main__.py`.
- **Final Pydantic validation** (`Validator.v_output`): Called once in `__main__.py` after all prompts are processed, on the aggregated JSON list string. Validates each entry against `OutputModel`. If any entry fails, nothing is written to disk.

## Testing Strategy

### Input Validation (`src/__main__.py`)

Before any inference runs, `main()` validates both input files:

- Each entry in the prompts file is validated against `InputModel(prompt: str)`.
- Each entry in the functions file is validated against `FunctionModel(name: str, description: str, parameters: dict[str, ParameterModel], returns: ParameterModel)`.

If validation fails, a `ValidationError` is raised with a descriptive message and the program exits with code 1.

### Per-Prompt Validation (`src/llm_model.py` — `CostimizedModel.generate`)

Token-by-token generation is halted as soon as `JsonStopDetector.feed()` signals a complete JSON object, ensuring each result is at minimum structurally sound before being appended to the output list.

### Final Output Validation (`src/validator.py` — `Validator.v_output`)

After all prompts are processed, `v_output` parses the entire aggregated JSON list and validates each entry against `OutputModel`. If any entry fails Pydantic validation, the function returns `False` and nothing is written to disk.

### Error Handling (`src/__main__.py`)

The `main()` function catches four exception types:

- `json.JSONDecodeError` → prints `"Error [JSON]: ..."` and exits with code 1.
- `ValidationError` → prints `"Error [VALIDATION]: ..."` and exits with code 1.
- `FileNotFoundError` → prints `"Error [FILE NOT FOUND]: ..."` and exits with code 1.
- `Exception` → prints `"Error [UNEXPECTED]: ..."` and exits with code 1.

### Static Analysis (`Makefile`)

```bash
make lint       # uv run flake8 . && uv run mypy . --warn-return-any ...
make lint-strict # uv run flake8 . && uv run mypy --strict .
```

## Example Usage

### Running with Default Inputs

```bash
make run
```

**Input** (`data/input/function_calling_tests.json` — 11 prompts):

```json
[
  { "prompt": "What is the sum of 2 and 3?" },
  { "prompt": "What is the sum of 265 and 345?" },
  { "prompt": "Greet shrek" },
  { "prompt": "Greet john" },
  { "prompt": "Reverse the string 'hello'" },
  { "prompt": "Reverse the string 'world'" },
  { "prompt": "What is the square root of 16?" },
  { "prompt": "Calculate the square root of 144" },
  { "prompt": "Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS" },
  { "prompt": "Replace all vowels in 'Programming is fun' with asterisks" },
  { "prompt": "Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'" }
]
```

**Output** (`data/output/function_calling_results.json` — 11 results):

```json
[
    {
        "prompt": "What is the sum of 2 and 3?",
        "name": "fn_add_numbers",
        "parameters": {
            "a": 2,
            "b": 3
        }
    },
    {
        "prompt": "What is the sum of 265 and 345?",
        "name": "fn_add_numbers",
        "parameters": {
            "a": 265,
            "b": 345
        }
    },
    {
        "prompt": "Greet shrek",
        "name": "fn_greet",
        "parameters": {
            "name": "shrek"
        }
    },
    {
        "prompt": "Greet john",
        "name": "fn_greet",
        "parameters": {
            "name": "john"
        }
    },
    {
        "prompt": "Reverse the string 'hello'",
        "name": "fn_reverse_string",
        "parameters": {
            "s": "hello"
        }
    },
    {
        "prompt": "Reverse the string 'world'",
        "name": "fn_reverse_string",
        "parameters": {
            "s": "world"
        }
    },
    {
        "prompt": "What is the square root of 16?",
        "name": "fn_get_square_root",
        "parameters": {
            "a": 16
        }
    },
    {
        "prompt": "Calculate the square root of 144",
        "name": "fn_get_square_root",
        "parameters": {
            "a": 144
        }
    },
    {
        "prompt": "Replace all numbers in 'Hello 34 I'm 233 years old' with NUMBERS",
        "name": "fn_substitute_string_with_regex",
        "parameters": {
            "source_string": "Hello 34 I'm 233 years old",
            "regex": "(\\d+)",
            "replacement": "NUMBERS"
        }
    },
    {
        "prompt": "Replace all vowels in 'Programming is fun' with asterisks",
        "name": "fn_substitute_string_with_regex",
        "parameters": {
            "source_string": "Programming is fun",
            "regex": "([aeiouAEIOU])",
            "replacement": "asterisk"
        }
    },
    {
        "prompt": "The dog sat on the mat with another dog",
        "name": "fn_substitute_string_with_regex",
        "parameters": {
            "source_string": "The cat sat on the mat with another cat",
            "regex": "cat",
            "replacement": "dog"
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

## Resources

### References

- [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) — the default model loaded in `llm_sdk/__init__.py`
- [How Large Language Models Work](https://www.youtube.com/watch?v=5sLYAQS9sWQ) — Youtube video
- [Large Language Models explained briefly](https://www.youtube.com/watch?v=LPZh9BOjkQs&t=5s) — Youtube video
- [Transformers, the tech behind LLMs | Deep Learning Chapter 5](https://www.youtube.com/watch?v=wjZofJX0v4M&t=3s) — Youtube video
- [How To Use JSON In Python](https://www.youtube.com/watch?v=-51jxlQaxyA) — Youtube video

### AI Usage

AI tools were used during development for assistance with prompt engineering, debugging, and documentation. All code was reviewed and tested by the project author.