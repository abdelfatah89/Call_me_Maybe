import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .validator import Validator, ValidationError
from .llm_model import CostimizedModel
from .constrained import constrained


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments using the subject-compatible names."""
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
        "--functions_definition",
        dest="functions_definition",
        default="data/input/functions_definition.json",
    )
    return parser.parse_args()


def load_json(file: str) -> Any:
    """Load JSON from a file with UTF-8 decoding."""
    with open(file, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    """Run the function-calling pipeline and write a valid JSON output file."""
    try:
        validator = Validator()

        # init paths
        args = _parse_args()
        input_file: str = args.input
        output_file: str = args.output
        functions_definition_file: str = args.functions_definition
        paths = [input_file, functions_definition_file]

        # Validation
        validator.v_paths(paths)
        prompts = validator.v_input_file(input_file)
        funcs = validator.v_funcs_file(functions_definition_file)

        # The model wrapper falls back gracefully if the local model is absent.
        model = CostimizedModel()

        # Generate output
        output_list = constrained(model, prompts, funcs)

        output = json.dumps(output_list, ensure_ascii=False)

        # Validate output and save to file
        valid = validator.v_output(output, funcs)
        if valid:
            output_data = json.loads(output)
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
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
