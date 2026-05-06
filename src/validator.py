from pathlib import Path
from typing import Any, Dict, List, Sequence
import json

from .models import FunctionModel, InputModel, OutputModel


class ValidationError(Exception):
    """Raised when input, function definitions, or output fail validation."""

    pass


class Validator:
    """Validate project files and generated function-call objects."""

    def v_paths(self, paths: List[str]) -> None:
        """Ensure every required input path exists."""
        for path in paths:
            if not Path(path).exists():
                raise FileNotFoundError(
                        f"File {path} not found")

    def v_input_file(self, input_file: str) -> List[InputModel]:
        """Load and validate the prompt input file."""
        with open(input_file, "r", encoding="utf-8") as f:
            input_data = json.load(f)
        return self.v_input_data(input_data)

    def v_funcs_file(self, funcs_file: str) -> List[FunctionModel]:
        """Load and validate the function-definition file."""
        with open(funcs_file, "r", encoding="utf-8") as f:
            functions_definition = json.load(f)
        return self.v_funcs_data(functions_definition)

    def v_input_data(self, input_data: Any) -> List[InputModel]:
        """Validate deserialized prompt data."""
        if not isinstance(input_data, list):
            raise ValidationError("Input file must contain a JSON array")
        try:
            return [InputModel(**data) for data in input_data]
        except Exception as e:
            raise ValidationError(
                f"Error [INPUT VALIDATION]: Invalid input data: {e}"
            ) from e

    def v_funcs_data(self, functions_definition: Any) -> List[FunctionModel]:
        """Validate deserialized function definitions."""
        if not isinstance(functions_definition, list):
            raise ValidationError(
                "Functions definition file must contain a JSON array"
            )
        if not functions_definition:
            raise ValidationError(
                "Functions definition file must contain at least one function"
            )
        try:
            return [FunctionModel(**func) for func in functions_definition]
        except Exception as e:
            raise ValidationError(
                "Error [FUNCTIONS DEFINITION VALIDATION]: "
                f"Invalid functions definition: {e}"
            ) from e

    def v_schema(self, data: str, schema: dict[str, str]) -> bool:
        """Validate that all schema keys are present in a JSON object."""
        data = json.loads(data)
        for key in schema.keys():
            if key not in data:
                raise ValidationError(f"Missing required field: {key}")
        return True

    @staticmethod
    def v_output(
        row_data: str,
        functions: Sequence[FunctionModel] | None = None,
    ) -> bool:
        """Validate generated output JSON and optional function schemas."""
        list_data: List[Dict[str, Any]] = json.loads(row_data)
        functions_by_name = {
            function.name: function for function in functions or []
        }
        for data in list_data:
            try:
                output = OutputModel(**data)
            except Exception as e:
                raise ValidationError(f"Invalid output object: {e}") from e
            if functions_by_name:
                Validator._validate_output_against_function(
                    output,
                    functions_by_name,
                )
        return True

    @staticmethod
    def _validate_output_against_function(
        output: OutputModel,
        functions_by_name: Dict[str, FunctionModel],
    ) -> None:
        """Ensure output parameters exactly match the selected function."""
        function = functions_by_name.get(output.name)
        if function is None:
            raise ValidationError(f"Unknown function name: {output.name}")
        expected = set(function.parameters.keys())
        actual = set(output.parameters.keys())
        if actual != expected:
            raise ValidationError(
                f"Parameters for {output.name} must be {sorted(expected)}, "
                f"got {sorted(actual)}"
            )
        for name, parameter in function.parameters.items():
            value = output.parameters[name]
            if not Validator._value_matches_type(value, parameter.type):
                raise ValidationError(
                    f"Parameter {name!r} for {output.name} must be "
                    f"{parameter.type}, got {type(value).__name__}"
                )

    @staticmethod
    def _value_matches_type(value: Any, expected_type: str) -> bool:
        """Return whether a Python value matches a JSON-schema-like type."""
        if expected_type == "number":
            return (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
            )
        if expected_type == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if expected_type == "string":
            return isinstance(value, str)
        if expected_type == "boolean":
            return isinstance(value, bool)
        if expected_type == "array":
            return isinstance(value, list)
        if expected_type == "object":
            return isinstance(value, dict)
        return True
