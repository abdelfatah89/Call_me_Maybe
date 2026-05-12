from src.json_stop_detector import JsonStopDetector
from .prompt import get_instructions
from .llm_model import CostimizedModel
from typing import Any, List
from concurrent.futures import ThreadPoolExecutor


GREEN = '\033[92m'
BLUE = '\033[94m'
BOLD = '\033[1m'
RESET = '\033[0m'


def generate_model_output(
        model: CostimizedModel,
        prompt: str,
        instructions_tokens: List[int],
        output_list: List[str]
        ) -> None:

    detector = JsonStopDetector()

    print(f"{BOLD}{BLUE}User Request:{RESET} {prompt}")
    result = model.generate(
        prompt, instructions_tokens, detector)

    output_list.append(result)


def constrained(model: CostimizedModel,
                prompts: List[str],
                funcs: Any) -> List[str]:
    output_list: List[str] = []

    instructions = get_instructions(funcs)
    instructions_tokens = model.encode(instructions)
    for prompt in prompts:
        generate_model_output(
            model, prompt,
            instructions_tokens, output_list)

    # with ThreadPoolExecutor(max_workers=3) as executor:
    #     _ = {
    #         executor.submit(
    #             generate_model_output,
    #             model, prompt,
    #             instructions_tokens, output_list
    #         )
    #         for prompt in prompts
    #     }

    return output_list
