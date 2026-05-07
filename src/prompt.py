import json
from functools import lru_cache
from typing import Dict, Any, Optional, List


@lru_cache(maxsize=1)
def _load_schema() -> str:
    # Read once and cache the serialised form so we don't pay disk + json.dumps
    # cost on every prompt build (11 prompts -> 11 reads becomes 1 read).
    with open("output_schema.json", "r") as f:
        schema: Dict[str, Any] = json.load(f)
    return json.dumps(schema)


def get_prompt(prompt: str, funcs: List[Dict[str, Any]],
               last_error: Optional[str] = None) -> str:
    schema_str = _load_schema()

    final_prompt = f"""
You are a function-calling assistant.
Select the best function for the user request and return ONLY one valid JSON object matching the schema.

Functions:
{funcs}
{"\nPrevious error: " + last_error if last_error else ""}

Rules:

Fix any previous error.
Output ONLY JSON (no text, no markdown).
Match the schema EXACTLY: {schema_str}
Do not add/remove/rename fields.
"name" most be one of available functions.
"prompt" = exact user request.
"parameters" = valid fields only, correct types.
No assumptions, no extra data.
Ensure valid JSON.

Order of keys:
prompt, name, parameters.

User request:
{prompt}

Result:
{'{'}
"""
    return final_prompt
