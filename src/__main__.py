import argparse
import json
import sys
from .models import FunctionModel, InputModel
from .constrained import constrained
from .generator import ValidationError
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
    )
    parser.add_argument(
        "--functions-definition",
        default="data/input/functions_definition.json",
    )
    return parser.parse_args()


def main() -> None:
    try:
        args = _parse_args()
        input_file: str = args.input
        output_file: str = args.output
        functions_definition_file: str = args.functions_definition

        for path in [input_file, functions_definition_file]:
            if not Path(path).exists():
                raise FileNotFoundError(f"File {path} not found")

        with open(input_file, "r") as f:
            input_data = json.load(f)
            try:
                _ = [InputModel(**data) for data in input_data]
            except Exception as e:
                raise ValidationError(
                    f"Error [INPUT VALIDATION]: Invalid input data: {e}")

        with open(functions_definition_file, "r") as f:
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

        output = constrained(functions_definition, input_data)
        if output is not None:
            output_data: str = json.loads(output)
            with open(output_file, "w") as f:
                json.dump(output_data, f, indent=4)

    except json.JSONDecodeError as e:
        print(f"Error [JSON]: {e}")
        sys.exit(1)
    except ValidationError as e:
        print(f"Error [VALIDATION]: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"Error [FILE NOT FOUND]: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error [UNEXPECTED]: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
