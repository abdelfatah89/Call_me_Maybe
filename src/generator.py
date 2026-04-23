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
"""

        print(final_prompt)
        result = self.ask_model(final_prompt)
        if self._validate_schema(result, schema):
            return result
        return None

    def _validate_schema(self, data: str, schema: dict[str, str]) -> bool:
        data = json.loads(data)
        for key in schema.keys():
            if key not in data:
                return False
                raise ValidationError(
                  f"Error [OUTPUT VALIDATION]: Missing key: {key}")
        return True

    @staticmethod
    def _validate_output(row_data: str) -> bool:
        list_data: List[Dict[str, Any]] = json.loads(row_data)
        for data in list_data:
            try:
                _ = OutputModel(**data)
            except Exception as e:
                return False
                raise ValidationError(f"Error [OUTPUT VALIDATION]: '{e}' ")
        return True


# def build_system_prompt(functions: list[dict[str, Any]],
#                         user_question: str) -> str:
#     functions_json = json.dumps(functions, indent=2)

#     rules = f"""
# You are a function-calling assistant.

# Your job is to choose the best function for the user's request
# and return ONLY a valid JSON object.

# Available functions:
# {functions_json}

# Rules:
# 1. Return exactly one JSON object.
# 2. Do not add explanations.
# 3. Do not add markdown.
# 4. Do not add code fences.
# 5. The JSON must have exactly this Format:
# {{
#   "prompt": "<prompt>",
#   "name": "<function_name>",
#   "parameters": {{
#     "<param1>": <value>,
#     "<param2>": <value>
#   }}
# }}
# Make sure you respect this format
# 6. The "prompt" must be the user's request.
# 7. The "name" must be one of the available functions.
# 8. The "parameters" object must contain the correct parameter
# names and value types.
# 9. Do not invent extra fields.
# 10. If a parameter type is number, output a JSON number.
# 11. If a parameter type is string, output a JSON string.
# 12. If a parameter type is boolean, output true or false."""

#     user_request = """User request:last_errorlast_error
# {user_question}"""

#     return rules + user_request
