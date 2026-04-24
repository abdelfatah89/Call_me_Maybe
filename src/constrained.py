from llm_sdk import Small_LLM_Model
from src.generator import Generator
from .json_stop_detector import JsonStopDetector, JsonStopDetectorError
from typing import Any, List, Dict, Optional
import numpy as np
import json


MAX_TOKENS = 256
MAX_TRIES = 3


def get_schema(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r") as f:
        schema = json.load(f)
    return schema


def ask_model(prompt: str) -> str:
    model = Small_LLM_Model()
    detector = JsonStopDetector()

    # The '{' we pre-filled is part of the response we return.
    result = "{"
    detector.feed("{")

    for _ in range(MAX_TOKENS):
        ids = model.encode(prompt)
        logits = model.get_logits_from_input_ids(ids.tolist()[0])
        new_ids = int(np.argmax(logits))
        next_word = model.decode([new_ids])

        prompt += next_word
        result += next_word
        if detector.feed(next_word):
            break

    return result


def constrained(functions: List[Dict[str, Any]],
                input_data: List[Dict[str, str]]) -> Optional[str]:

    output_list: List[str] = []
    gen = Generator(ask_model)
    schema = get_schema("output_schema.json")

    for data in input_data:
        last_error: Optional[str] = None
        for _ in range(MAX_TRIES):
            try:
                result = gen.json(data["prompt"], schema,
                                  functions, last_error)
                if result is not None:
                    output_list.append(result)
                    break
            except JsonStopDetectorError as e:
                last_error = str(e)

    output = "[" + ", ".join(output_list) + "]"
    valid = gen._validate_output(output)
    if valid:
        return output

    return None
