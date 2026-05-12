*This project has been created as part of the 42 curriculum by alaktaou.*

# Call me Maybe — LLM-Based Function Calling with Constrained Decoding

## Description

**Call me Maybe** is a function-calling engine that uses a locally-running language model ([Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)) to translate natural language requests into structured JSON function calls.

Given a set of function definitions (`data/input/functions_definition.json`) and a list of user prompts (`data/input/function_calling_tests.json`), the program:

1. Constructs a system prompt containing the available functions, the required output schema, and the user's request (`src/generator.py` — `Generator.json`).
2. Feeds the prompt to the Qwen3-0.6B causal language model and performs greedy token-by-token decoding using `numpy.argmax` on the raw logits (`src/constrained.py` — `ask_model`).
3. Monitors the generated token stream with a `JsonStopDetector` that tracks brace/bracket depth and halts generation once a syntactically complete JSON object is formed (`src/json_stop_detector.py`).
4. Validates the result against the expected schema. On `JsonStopDetectorError`, feeds the error message back into the prompt and retries up to `MAX_TRIES = 3` times (`src/constrained.py` — `constrained`).
5. After processing all prompts, validates the aggregated output list with Pydantic's `OutputModel` and writes the result to `data/output/function_calling_results.json` (`src/__main__.py` — `main`).

### Project Structure

```
.
├── llm_sdk/
│   └── __init__.py          # Small_LLM_Model class — model loading, tokenization, logit extraction
├── src/
│   ├── __main__.py           # CLI entry point, file I/O, error handling
│   ├── constrained.py        # Orchestration: ask_model() + constrained() decoding loop
│   ├── generator.py          # Generator class — prompt construction, schema validation
│   ├── json_stop_detector.py # JsonStopDetector — streaming JSON completeness detection
│   └── models.py             # Pydantic models: FunctionModel, InputModel, OutputModel
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
    "prompt": "<prompt>",
    "name": "<function_name>",
    "parameters": {
        "<param1>": "<value>",
        "<param2>": "<value>"
    }
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

The algorithm is a greedy decoding loop with structural stop detection. It uses prompt engineering and a retry mechanism to produce valid JSON from the model.

#### Step 1 — Prompt Construction (`src/generator.py` — `Generator.json`)

The `Generator.json` method assembles a prompt containing:

- The full list of available function definitions passed as the `additional` parameter.
- If `last_error` is not `None`, the string `"The previous output was invalid: "` followed by the error message.
- Three critical instructions telling the model to fix previous errors.
- Twelve strict rules the model must follow (valid JSON only, no markdown, match the schema exactly, correct types, etc.).
- The exact JSON schema injected via `json.dumps(schema, indent=4)`.
- A three-item validation checklist.
- The user's prompt.
- A pre-filled opening brace `{` as the last character, so the model's continuation starts inside a JSON object.

#### Step 2 — Token-by-Token Greedy Decoding (`src/constrained.py` — `ask_model`)

The `ask_model` function:

1. Instantiates a new `Small_LLM_Model()` (from `llm_sdk`).
2. Loads the vocabulary file via `model.get_path_to_vocab_file()` and builds an `id_to_token` reverse mapping.
3. Creates a `JsonStopDetector()` and pre-feeds `"{"` into it (setting `object_depth = 1`).
4. Encodes the prompt into `input_ids` via `model.encode(prompt)`.
5. Runs a loop for up to `MAX_TOKENS = 256` iterations:
   - Calls `model.get_logits_from_input_ids(input_ids)` to get raw logits for the next token.
   - Selects the next token via `int(np.argmax(logits))` (greedy — no sampling, no temperature).
   - Looks up the token string in `id_to_token` and normalizes it with `replace_char` (`Ġ` → space, `Ċ` → newline).
   - Appends the decoded text to `result` and the token ID to `input_ids`.
   - Feeds the decoded text to `detector.feed()`. If it returns `True` (complete JSON object), breaks.
6. Returns the accumulated `result` string.

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

#### Step 4 — Retry Loop and Validation (`src/constrained.py` — `constrained`)

The `constrained` function processes each prompt:

```
for each prompt in input_data:
    last_error = None
    for _ in range(MAX_TRIES = 3):
        try:
            result = gen.json(prompt, schema, functions, last_error)
            if result is not None:      # _validate_schema passed
                output_list.append(result)
                break
        except JsonStopDetectorError as e:
            last_error = str(e)         # fed back into next attempt's prompt
```

There are two distinct validation mechanisms:

1. **Per-prompt schema key check** (`Generator._validate_schema`): Called inside `gen.json()` after `ask_model` returns. Parses the JSON string with `json.loads()` and checks that every key from `output_schema.json` (`prompt`, `name`, `parameters`) exists in the result. If a key is missing, `gen.json()` returns `None` and the retry loop continues — but no error message is fed back.

2. **Final Pydantic validation** (`Generator._validate_output`): Called once after all prompts are processed, on the aggregated JSON list string. Parses each item with `OutputModel(prompt: str, name: str, parameters: dict[str, Any])`. If any item fails, the function returns `None` and nothing is written to disk.

Only `JsonStopDetectorError` produces an error message that is fed back into the next retry's prompt. Schema validation failure triggers a silent retry.

### Token Handling (`src/constrained.py` — `replace_char`)

The Qwen tokenizer's vocabulary uses `Ġ` (Unicode U+0120) to represent a space and `Ċ` (Unicode U+010A) to represent a newline. The `replace_char` function:

```python
def replace_char(text: str) -> str:
    return text.replace("Ġ", " ").replace("Ċ", "\n")
```

This normalization is applied to each decoded token in `ask_model` before the text reaches the `JsonStopDetector` and the result buffer.

## Design Decisions

### Model: Qwen/Qwen3-0.6B

The default model is set in `llm_sdk/__init__.py` as `model_name: str = "Qwen/Qwen3-0.6B"`. The `Small_LLM_Model` class auto-selects the compute device with priority `mps > cuda > cpu` and sets `dtype` to `float16` on GPU/MPS or `float32` on CPU. The model is set to `eval()` mode and all parameter gradients are disabled (`requires_grad = False`).

### Greedy Decoding

The decoding loop in `ask_model` uses `int(np.argmax(logits))` — a single deterministic token selection per step. There is no temperature, sampling, or top-k/top-p filtering in the codebase. This produces identical output for identical input.

### Streaming Stop Detection

The `JsonStopDetector` halts generation token-by-token as soon as a complete JSON object is detected, rather than generating a fixed number of tokens and parsing afterwards. The `ask_model` loop checks `detector.feed(next_word)` after each token and breaks immediately on `True`.

### Prompt Pre-filling

The prompt in `Generator.json` ends with `{'{'}` and `ask_model` initializes `result = "{"` with `detector.feed("{")`. This means the model's first generated token continues from inside an already-opened JSON object.

### Pydantic Validation

Three Pydantic `BaseModel` classes in `src/models.py` enforce data contracts:

- `FunctionModel(name: str, description: str, parameters: dict[str, ParameterModel], returns: ParameterModel)` — validates `functions_definition.json` entries.
- `InputModel(prompt: str)` — validates `function_calling_tests.json` entries.
- `OutputModel(prompt: str, name: str, parameters: dict[str, Any])` — validates final output entries.

### Model Instantiation

The `ask_model` function in `src/constrained.py` creates a new `Small_LLM_Model()` on every call. Since `constrained` calls `ask_model` via `gen.json()` for each prompt (and potentially each retry), the model is loaded multiple times during a full run.

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

- `MAX_TOKENS = 256` (in `src/constrained.py`) — maximum tokens generated per prompt before the loop exits.
- `MAX_TRIES = 3` (in `src/constrained.py`) — maximum retry attempts per prompt.

### Output Behavior

- If all `MAX_TRIES` attempts fail for a prompt (either `JsonStopDetectorError` or `_validate_schema` returning `False`), that prompt is skipped — nothing is appended to `output_list` for it.
- After all prompts are processed, the results are joined as `"[" + ", ".join(output_list) + "]"` and passed to `_validate_output`.
- If `_validate_output` returns `False`, the `constrained` function returns `None` and `__main__.py` writes nothing to disk.

## Challenges Faced

### 1. Stopping Generation at the JSON Boundary

The `JsonStopDetector` in `src/json_stop_detector.py` addresses the problem of knowing when a complete JSON object has been emitted. It tracks `object_depth` and `array_depth` through each character. Braces and brackets inside string literals do not affect the counters — the `in_string` and `escape` flags handle this. The detector returns `True` the instant both depths reach zero, and the `ask_model` loop breaks immediately.

### 2. Tokenizer Special Characters

The Qwen tokenizer encodes spaces as `Ġ` and newlines as `Ċ` in its vocabulary. The `replace_char` function in `src/constrained.py` normalizes these before feeding text to the `JsonStopDetector` and the result buffer. Without this normalization, the stop detector would receive `Ġ` and `Ċ` characters instead of actual whitespace, and the resulting JSON would contain these Unicode markers.

### 3. Handling Malformed Model Output

The code addresses malformed output at multiple levels:

- **`JsonStopDetectorError`**: Raised on structural violations (unmatched braces/brackets, non-`{` start character). Caught in the `constrained` function's retry loop, where `str(e)` is stored in `last_error` and injected into the next prompt via `"The previous output was invalid: " + last_error`.
- **`_validate_schema`**: Returns `False` if the parsed JSON is missing any of the required keys (`prompt`, `name`, `parameters`). This causes `gen.json()` to return `None`, triggering a silent retry.
- **`_validate_output`**: Final check using Pydantic's `OutputModel` on the complete result list.

### 4. Compact Output Schema

The output schema in `output_schema.json` contains only three fields: `prompt`, `name`, and `parameters`. The `Generator.json` method injects this schema into the prompt via `json.dumps(schema, indent=4)`, and the twelve strict rules reference it explicitly. The `MAX_TOKENS = 256` limit provides enough room for the model to generate a single function call JSON object.

## Testing Strategy

### Input Validation (`src/__main__.py`)

Before any inference runs, `main()` validates both input files:

- Each entry in the prompts file is validated against `InputModel(prompt: str)`.
- Each entry in the functions file is validated against `FunctionModel(name: str, description: str, parameters: dict[str, ParameterModel], returns: ParameterModel)`.

If validation fails, a `ValidationError` is raised with a descriptive message and the program exits with code 1.

### Per-Prompt Validation (`src/generator.py` — `_validate_schema`)

After each successful `ask_model` call, `_validate_schema` parses the JSON string and verifies that every key from `output_schema.json` is present in the result.

### Final Output Validation (`src/generator.py` — `_validate_output`)

After all prompts are processed, `_validate_output` parses the entire aggregated JSON list and validates each entry against `OutputModel`. If any entry fails Pydantic validation, the function returns `False` and nothing is written to disk.

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

### Smoke Test (`test.py`)

`test.py` loads the model, encodes the string `"write a json schema: \n{"`, and runs `MAX_TOKENS = 10` greedy decoding steps, printing the accumulated text after each step:

```bash
uv run python3 test.py
```

Expected output (tokens include Qwen's vocabulary markers since `test.py` does not call `replace_char`):

```
writeĠaĠjsonĠschema:ĠĊ{
writeĠaĠjsonĠschema:ĠĊ{Ġ"
writeĠaĠjsonĠschema:ĠĊ{Ġ"name
writeĠaĠjsonĠschema:ĠĊ{Ġ"name":
writeĠaĠjsonĠschema:ĠĊ{Ġ"name":Ġ"
writeĠaĠjsonĠschema:ĠĊ{Ġ"name":Ġ"name
writeĠaĠjsonĠschema:ĠĊ{Ġ"name":Ġ"name",
writeĠaĠjsonĠschema:ĠĊ{Ġ"name":Ġ"name",Ġ"
writeĠaĠjsonĠschema:ĠĊ{Ġ"name":Ġ"name",Ġ"age
writeĠaĠjsonĠschema:ĠĊ{Ġ"name":Ġ"name",Ġ"age":
```

Note: `Ġ` represents a space and `Ċ` represents a newline in Qwen's tokenizer vocabulary.

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
        "parameters": { "a": 2, "b": 3 }
    },
    {
        "prompt": "What is the sum of 265 and 345?",
        "name": "fn_add_numbers",
        "parameters": { "a": 265, "b": 345 }
    },
    {
        "prompt": "Greet shrek",
        "name": "fn_greet",
        "parameters": { "name": "shrek" }
    },
    {
        "prompt": "Greet john",
        "name": "fn_greet",
        "parameters": { "name": "john" }
    },
    {
        "prompt": "Reverse the string 'hello'",
        "name": "fn_reverse_string",
        "parameters": { "s": "hello" }
    },
    {
        "prompt": "Reverse the string 'world'",
        "name": "fn_reverse_string",
        "parameters": { "s": "world" }
    },
    {
        "prompt": "What is the square root of 16?",
        "name": "fn_get_square_root",
        "parameters": { "a": 16 }
    },
    {
        "prompt": "Calculate the square root of 144",
        "name": "fn_get_square_root",
        "parameters": { "a": 144 }
    },
    {
        "prompt": "Replace all numbers in 'Hello 34 I'm 233 years old' with NUMBERS",
        "name": "fn_substitute_string_with_regex",
        "parameters": {
            "source_string": "Hello 34 I'm 233 years old",
            "regex": "([0-9]+)",
            "replacement": "NUMBERS"
        }
    },
    {
        "prompt": "Replace all vowels in 'Programming is fun' with asterisks",
        "name": "fn_substitute_string_with_regex",
        "parameters": {
            "source_string": "Programming is fun",
            "regex": "([aeiouAEIOU])",
            "replacement": "*"
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
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/) — `AutoModelForCausalLM` and `AutoTokenizer` used in `llm_sdk/__init__.py`
- [Hugging Face Hub](https://huggingface.co/docs/huggingface_hub/) — `hf_hub_download` used to fetch vocabulary files
- [Pydantic](https://docs.pydantic.dev/) — `BaseModel` used for data validation in `src/models.py`
- [PyTorch](https://pytorch.org/docs/) — tensor operations and model inference throughout `llm_sdk/`
- [uv](https://docs.astral.sh/uv/) — package manager used in `Makefile`

### AI Usage

AI tools were used during development for assistance with prompt engineering, debugging, and documentation. All code was reviewed and tested by the project author.
