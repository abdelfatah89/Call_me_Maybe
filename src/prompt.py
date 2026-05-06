import json
from typing import Any, Dict, List, Optional


def get_prompt(prompt: str, funcs: List[Dict[str, Any]],
               last_error: Optional[str] = None) -> str:
    """Build a compact JSON-only prompt for optional raw generation."""
    schema: Dict[str, Any] = {
        "prompt": "<exact user request>",
        "name": "<function_name>",
        "parameters": {"<param>": "<value>"},
    }
    previous_error = f"\nPrevious error: {last_error}" if last_error else ""

    final_prompt = f"""
You are a function-calling assistant.
Select the best function for the user request and return ONLY one valid
JSON object matching the schema.

Functions:
{funcs}
{previous_error}

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
