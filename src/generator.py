from typing import Any, List, Dict, Optional
import json
from .models import OutputModel


class ValidationError(Exception):
    pass


class Generator:
    def __init__(self, ask_model) -> None:
        self.ask_model = ask_model

    def json(self, prompt: str,
             schema: Dict[str, Any],
             additional: List[Dict[str, Any]],
             last_error: Optional[str] = None) -> Optional[str]:

        final_prompt = f"""
You are a function-calling assistant.

Your task is to select the most appropriate function for the user's request
and return ONLY a valid JSON object that strictly follows the required schema.

Available functions:
{additional}

{"The previous output was invalid: " + last_error if last_error else ""}

CRITICAL INSTRUCTIONS:
- If there was a previous error, you MUST fix it in this response.
- Carefully analyze the error and ensure it does NOT happen again.
- Double-check your JSON before returning it.

STRICT RULES (must be followed exactly):
1. Output MUST be a single valid JSON object.
2. Do NOT include any explanations, text, or comments.
3. Do NOT use markdown or code fences.
4. Do NOT output anything before or after the JSON.
5. The JSON structure MUST exactly match this schema:
{json.dumps(schema, indent=4)}
6. Do NOT add, remove, or rename any fields.
7. "name" MUST be one of the available functions.
8. "prompt" MUST exactly reflect the user's request.
9. "parameters" MUST include only valid parameter names.
10. All parameter values MUST match their required types:
   - number → JSON number (no quotes)
   - string → JSON string
   - boolean → true or false
11. Do NOT invent or assume missing parameters.
12. Ensure the output is valid JSON (no trailing commas, correct quotes, etc.).

VALIDATION CHECK (before responding):
- Is the JSON valid?
- Does it match the schema exactly?
- Did you fix the previous error?

User request:
{prompt}

Result:
{'{'}
"""

        result = self.ask_model(final_prompt)
        if self._validate_schema(result, schema):
            return result
        return None

    def _validate_schema(self, data: str, schema: dict[str, str]) -> bool:
        data = json.loads(data)
        for key in schema.keys():
            if key not in data:
                return False
        return True

    @staticmethod
    def _validate_output(row_data: str) -> bool:
        list_data: List[Dict[str, Any]] = json.loads(row_data)
        for data in list_data:
            try:
                _ = OutputModel(**data)
            except Exception:
                return False
        return True
