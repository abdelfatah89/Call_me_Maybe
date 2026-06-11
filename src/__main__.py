import json
import sys
import os
import argparse
import traceback
from typing import Any
from .validator import Validator, ValidationError
from .llm_model import CostimizedModel
from .constrained import constrained


RED = '\033[91m'
RESET = '\033[0m'


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


def load_json(file: str) -> Any:
    with open(file) as f:
        loaded = json.load(f)
    return loaded


def main() -> None:
    try:
        validator = Validator()
        model = CostimizedModel()

        # init paths
        args = _parse_args()
        input_file: str = args.input
        output_file: str = args.output
        functions_definition_file: str = args.functions_definition
        paths = [input_file, functions_definition_file]

        # Validation
        validator.v_paths(paths)
        validator.v_input_file(input_file)
        validator.v_funcs_file(functions_definition_file)

        # load data from json files
        funcs = load_json(functions_definition_file)
        prompts = load_json(input_file)
        prompts = [p['prompt'] for p in prompts]

        # Generate output
        output_list = constrained(model, prompts, funcs)

        output = "[" + ", ".join(output_list) + "]"

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Validate output and save to file
        valid = validator.v_output(output)
        if valid:
            output_data: str = json.loads(output)
            with open(output_file, "w") as f:
                json.dump(output_data, f, indent=4)
        else:
            print(
                f"{RED}Error: Output validation failed. "
                f"No output file generated.{RESET}")

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
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
