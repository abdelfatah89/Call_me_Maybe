from llm_sdk import Small_LLM_Model
from src.generator import Generator
from .json_stop_detector import JsonStopDetector, JsonStopDetectorError
from typing import Any, List, Dict, Optional
import numpy as np
import json
import threading


MAX_TOKENS = 256
MAX_TRIES = 3


def get_schema(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r") as f:
        schema: Dict[str, Any] = json.load(f)
    return schema


def char_replace(text: str) -> str:
    return text.replace(" ", "Ġ").replace("\n", "Ċ")


def replace_char(text: str) -> str:
    return text.replace("Ġ", " ").replace("Ċ", "\n")


def ask_model(prompt: str) -> str:
    model = Small_LLM_Model()
    vocab = model.get_path_to_vocab_file()
    detector = JsonStopDetector()

    # The '{' we pre-filled is part of the response we return.
    result = "{"
    detector.feed("{")

    with open(vocab, 'r') as f:
        token_to_id = json.load(f)
    id_to_token = {int(v): k for k, v in token_to_id.items()}

    token_ids = model.encode(prompt)
    input_ids = token_ids.tolist()[0]
    tokens = [id_to_token[i] for i in input_ids]

    for _ in range(MAX_TOKENS):
        logits = model.get_logits_from_input_ids(input_ids)
        new_ids = int(np.argmax(logits))
        tokens = [id_to_token[i] for i in [new_ids]]
        next_word = replace_char("".join(tokens))
        prompt += next_word
        result += next_word
        input_ids.append(new_ids)
        if detector.feed(next_word):
            break

    return result


def get_output(gen: Generator,
               data: Dict[str, str],
               output_list: List[str],
               schema: Dict[str, str],
               functions: List[Dict[str, Any]]) -> None:

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


def constrained(functions: List[Dict[str, Any]],
                input_data: List[Dict[str, str]]) -> Optional[str]:

    output_list: List[str] = []
    gen = Generator(ask_model)
    schema = get_schema("output_schema.json")

    for i in range(0, len(input_data), 2):
        t1, t2 = None, None

        if input_data[i]:
            t1 = threading.Thread(
                target=get_output,
                args=(gen, input_data[i], output_list, schema, functions)
            )
            t1.start()

        if i + 1 < len(input_data) and input_data[i + 1]:
            t2 = threading.Thread(
                target=get_output,
                args=(gen, input_data[i + 1], output_list, schema, functions)
            )
            t2.start()

        if t1:
            t1.join()
        if t2:
            t2.join()

        # get_output(gen, input_data[i], output_list, schema, functions)
        # get_output(gen, input_data[i + 1], output_list, schema, functions)

        # last_error: Optional[str] = None
        # for _ in range(MAX_TRIES):
        #     try:
        #         result = gen.json(data["prompt"], schema,
        #                           functions, last_error)
        #         if result is not None:
        #             output_list.append(result)
        #             break
        #     except JsonStopDetectorError as e:
        #         last_error = str(e)

    output = "[" + ", ".join(output_list) + "]"
    valid = gen._validate_output(output)
    if valid:
        return output

    return None
