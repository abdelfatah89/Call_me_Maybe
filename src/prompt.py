import json
from typing import Dict, Any, List
from functools import lru_cache


@lru_cache(maxsize=1)
def _load_schema() -> Any:
    schema_path = "output_schema.json"
    with open(schema_path, "r") as f:
        return json.load(f)


def get_instructions(funcs: List[Dict[str, Any]]) -> str:
    schema = _load_schema()

    func_lines = []
    for fn in funcs:
        params = ", ".join(
            f"{k}: {v['type']}"
            for k, v in fn["parameters"].items()
        )
        func_lines.append(
            f"- {fn['name']}({params})")
    func_summary = "\n".join(func_lines)

    instructions = (
        f"You are a function-calling assistant.\n"
        f"Return one JSON object matching this schema exactly:"
        f"\n{json.dumps(schema)}\n"
        f"Available functions:\n{func_summary}\n"
        f"Key order: prompt, name, parameters.\n"
        f"User request:"
    )
    return instructions
