import json
from typing import Any, Dict


def get_schema(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r") as f:
        schema = json.load(f)
    return schema
