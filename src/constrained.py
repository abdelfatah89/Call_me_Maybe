from src.json_stop_detector import JsonStopDetector
from .prompt import get_prompt
from .llm_model import CostimizedModel
from typing import Any, List
import threading


MAX_TRIES = 3


def get_output(model: CostimizedModel,
               prompt: str,
               funcs: Any,
               output_list: List[str]) -> None:

    last_error = None
    # for _ in range(MAX_TRIES):
        # try:
    detector = JsonStopDetector()
    prompt = get_prompt(prompt, funcs, None)
    result = model.generate(prompt, detector)
    output_list.append(result)
        # except Exception as e:
        #     last_error = str(e)

from concurrent.futures import ThreadPoolExecutor, as_completed


def constrained(model: CostimizedModel,
                prompts: List[str],
                funcs: Any,
                max_workers: int = 8) -> List[str]:

    output_list: List[str] = []

    def task(prompt: str):
        return get_output(model, prompt, funcs, output_list)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(task, p)
            for p in prompts if p
        ]

        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                output_list.append(result)

    return output_list
