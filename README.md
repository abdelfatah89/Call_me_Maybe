*This project has been created as part of the 42 curriculum by abdelfatah89.*

# call me maybe

## Description

This project translates natural-language requests into structured function-call JSON.
For each input prompt, the program selects one available function and emits exactly:

```json
{
  "prompt": "original prompt",
  "name": "function_name",
  "parameters": {}
}
```

The output is written as a JSON array and validated before it is saved.

## Instructions

Install dependencies:

```sh
make install
```

Run with default files:

```sh
make run
```

Run with explicit paths:

```sh
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

Other useful commands:

```sh
make lint
make clean
make debug
```

## Algorithm explanation

The pipeline uses a constrained function-name selection step followed by deterministic
JSON construction:

1. Pydantic validates the prompt list and function definitions.
2. The local SDK model is asked to choose among only the available function names.
   Candidate function names are tokenized and stored in a trie. At each generation
   step, only token IDs that keep the output on a valid function-name path are
   considered.
3. If the model or tokenizer is unavailable, the program falls back to a lexical
   scorer so the CLI still exits cleanly without a traceback.
4. Parameters are extracted and coerced according to the selected function schema.
5. Pydantic and a schema-aware validator check the final object before writing it.

The raw model generation helper also uses a JSON stop detector that stops as soon as
one complete JSON object closes, preventing trailing prose after the object.

## Design decisions

- The model is loaded after input validation so malformed files fail quickly.
- The SDK wrapper catches model-load failures and exposes a safe fallback path.
- Output directories are created with `Path.mkdir(..., exist_ok=True)`.
- The output validator rejects unknown function names, missing parameters, extra
  parameters, and incorrect JSON types.
- Threaded model calls were removed because the provided SDK model is not documented
  as thread-safe.

## Performance analysis

The program performs bounded function-name selection instead of free-form JSON
generation for every token of the full object. This keeps the number of model calls
small and lets the bundled 11 prompts complete within the subject's five-minute
limit on a normal review machine. If the model cannot be loaded, fallback selection
keeps the command responsive and still produces schema-valid JSON.

## Challenges faced

Small language models can emit prose, partial JSON, or invalid fields when asked to
produce unrestricted structured output. The main mitigation is to constrain the only
model-generated part to a finite set of valid function names and construct the rest of
the JSON with validated Python data structures.

## Testing strategy

Validation focuses on:

- Missing files and malformed JSON inputs.
- Invalid prompt and function-definition shapes.
- Exact output keys and parameter names.
- JSON type matching for `number`, `integer`, `string`, `boolean`, `array`, and
  `object`.
- The bundled 11-prompt sample run.

## Resources

- Python `json` module documentation.
- Pydantic validation documentation.
- The provided `llm_sdk.Small_LLM_Model` API.
- General references on constrained decoding and token tries.

AI assistance was used to review the subject requirements, identify crash paths,
and help draft robust validation and documentation. The implementation decisions
remain explicit in the source code and can be reviewed through the validator and
constrained selection modules.
