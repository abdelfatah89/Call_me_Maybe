from llm_sdk import Small_LLM_Model
from typing import List, Dict, Tuple
import json
from functools import lru_cache
from .json_stop_detector import JsonStopDetector
import numpy as np
from datetime import datetime


MAX_TOKENS = 96


class CostimizedModel:
    def __init__(self) -> None:
        self.model = Small_LLM_Model()

        self.token_to_id, self.id_to_token = self._load_vocab()
        self.merges = self._load_merges()

    def _load_vocab(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        vocab_path = self.model.get_path_to_vocab_file()
        with open(vocab_path) as f:
            token_to_id = json.load(f)
        id_to_token = {int(v): k for k, v in token_to_id.items()}
        return token_to_id, id_to_token

    def _load_merges(self) -> Dict[Tuple[str, str], int]:
        merges = {}
        merges_path = self.model.get_path_to_merges_file()
        with open(merges_path, 'r') as f:
            for i, line in enumerate(f):
                p1, p2 = line.split()
                merges[(p1, p2)] = i
        return merges

    def tokenize(self, text: str) -> List[str]:
        text = text.replace(" ", "Ġ").replace("\n", "Ċ")
        tokens: List[str] = list(text)

        while True:
            best_pair = None
            best_rank = float('inf')

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i+1])
                rank = self.merges.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            new_tokens = []
            i = 0
            while i < len(tokens):
                if (i < len(tokens) - 1
                    and (tokens[i], tokens[i+1]) == best_pair):
                    new_tokens.append(tokens[i] + tokens[i+1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    @lru_cache(maxsize=1024)
    def _get_one_logit(self, input_id: int) -> float:
        logit = self.model.get_logits_from_input_ids([input_id])
        return logit[0]

    def get_logits(self, input_ids: List[int]) -> List[float]:
        return self.model.get_logits_from_input_ids(input_ids)

    def encode(self, text: str) -> List[int]:
        tokens = self.tokenize(text)
        return [self.token_to_id[t] for t in tokens]

    @lru_cache(maxsize=5120)
    def decode(self, ids: Tuple[int, ...]) -> str:
        return "".join(self.id_to_token[i] for i in ids)

    def generate(self, prompt: str, detector: JsonStopDetector) -> str:
        result = "{"
        detector.feed("{")

        tokens = self.encode(prompt)
        for _ in range(MAX_TOKENS):
            logits = self.get_logits(tokens)
            logits = np.array(logits, np.int64)
            new_id = int(np.argmax(logits))
            next_word = self.decode((new_id,))
            next_word = next_word.replace("Ġ", " ").replace("Ċ", "\n")
            result += next_word
            tokens.append(new_id)
            if detector.feed(next_word):
                break

        return result
