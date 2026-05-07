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
    # try:
    detector = JsonStopDetector()
    prompt = get_prompt(prompt, funcs, None)
    result = model.generate(prompt, detector)
    output_list.append(result)
    # except Exception as e:
    #     print(e)
    #     last_error = str(e)


def constrained(model: CostimizedModel,
                prompts: List[str],
                funcs: Any) -> List[str]:
    output_list: List[str] = []

    for i in range(0, len(prompts), 2):
        t1, t2 = None, None

        if prompts[i]:
            t1 = threading.Thread(
                target=get_output,
                args=(model, prompts[i], funcs, output_list)
            )
            t1.start()

        if i + 1 < len(prompts) and prompts[i + 1]:
            t2 = threading.Thread(
                target=get_output,
                args=(model, prompts[i + 1], funcs, output_list)
            )
            t2.start()

        if t1:
            t1.join()
        if t2:
            t2.join()

    return output_list
