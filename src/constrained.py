from typing import Any, List

from src.json_stop_detector import JsonStopDetector

from .llm_model import CostimizedModel
from .prompt import get_prompt


MAX_TRIES = 3


def get_output(model: CostimizedModel,
               prompt: str,
               funcs: Any) -> str:
    detector = JsonStopDetector()
    full_prompt = get_prompt(prompt, funcs, None)
    return model.generate(full_prompt, detector)


def constrained(model: CostimizedModel,
                prompts: List[str],
                funcs: Any) -> List[str]:
    # Threading was actively harmful: one model + Python's GIL on CPU torch ops
    # means the threads serialise anyway, while paying thread-creation overhead
    # and giving non-deterministic ordering. Sequential iteration also lets the
    # model fully exploit its KV cache per prompt.
    output_list: List[str] = []
    for prompt in prompts:
        if not prompt:
            continue
        output_list.append(get_output(model, prompt, funcs))
    return output_list
