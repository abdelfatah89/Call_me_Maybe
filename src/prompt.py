import json
from typing import Dict, Any, Optional, List


def get_prompt(prompt: str, funcs: List[Dict[str, Any]],
               last_error: Optional[str] = None) -> str:
    schema_path = "output_schema.json"
    with open(schema_path, "r") as f:
        schema: Dict[str, Any] = json.load(f)

    final_prompt = f"""
You are a function-calling assistant.
Select the best function for the user request and return ONLY one valid JSON object matching the schema.

Functions:
{funcs}
{"\nPrevious error: " + last_error if last_error else ""}

Rules:

Fix any previous error.
Output ONLY JSON (no text, no markdown).
Match the schema EXACTLY: {json.dumps(schema)}
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
