from .models import FunctionModel, InputModel, OutputModel
from pathlib import Path
from typing import List, Dict, Any
import json


class ValidationError(Exception):
    pass


class Validator:
    def __init__(self):
        pass

    def v_paths(self, paths: List[str]) -> None:
        for path in paths:
            if not Path(path).exists():
                raise FileNotFoundError(
                        f"File {path} not found")

    def v_input_file(self, input_file: str) -> None:
        with open(input_file, "r") as f:
            input_data = json.load(f)
            try:
                _ = [InputModel(**data) for data in input_data]
            except Exception as e:
                raise ValidationError(
                    f"Error [INPUT VALIDATION]: Invalid input data: {e}")

    def v_funcs_file(self, funcs_file: str) -> None:
        with open(funcs_file, "r") as f:
            functions_definition = json.load(f)
            try:
                _ = [
                    FunctionModel(**func)
                    for func in functions_definition
                    ]
            except Exception as e:
                raise ValidationError(
                    "Error [FUNCTIONS DEFINITION VALIDATION]:"
                    f"Invalid functions definition: {e}")

    def v_schema(self, data: str, schema: dict[str, str]):
        data = json.loads(data)
        for key in schema.keys():
            if key not in data:
                raise ValidationError(f"Missing required field: {key}")
        return True

    @staticmethod
    def v_output(row_data: str) -> bool:
        list_data: List[Dict[str, Any]] = json.loads(row_data)
        for data in list_data:
            try:
                _ = OutputModel(**data)
            except Exception:
                return False
        return True
