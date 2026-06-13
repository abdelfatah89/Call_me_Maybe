from llm_sdk import Small_LLM_Model
from typing import List, Dict, Tuple, Set
import json
from functools import lru_cache
from .json_stop_detector import JsonStopDetector
import numpy as np


MAX_TOKENS = 96
NEG_INF = float("-inf")


class CostimizedModel:
    def __init__(self) -> None:
        self.model = Small_LLM_Model()

        self.token_to_id, self.id_to_token = self._load_vocab()
        self.merges = self._load_merges()

        dummy_logits = self.model.get_logits_from_input_ids([0])
        self._logits_size: int = len(dummy_logits)

        self._precompute_token_sets()

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

    def _precompute_token_sets(self) -> None:
        open_brace:     Set[int] = set()
        close_brace:    Set[int] = set()
        open_bracket:   Set[int] = set()
        close_bracket:  Set[int] = set()
        quote:          Set[int] = set()
        colon:          Set[int] = set()
        comma:          Set[int] = set()
        digit_or_minus: Set[int] = set()
        general_content: Set[int] = set()
        whitespace:     Set[int] = set()

        for tid, raw in self.id_to_token.items():
            if tid >= self._logits_size:
                continue

            t = raw.replace("Ġ", " ").replace("Ċ", "\n")
            stripped = t.strip()
            if not t:
                continue

            if t == "{" or stripped == "{":
                open_brace.add(tid)
            if t == "}" or stripped == "}":
                close_brace.add(tid)
            if t == "[" or stripped == "[":
                open_bracket.add(tid)
            if t == "]" or stripped == "]":
                close_bracket.add(tid)
            if t == '"' or stripped == '"':
                quote.add(tid)
            if t == ":" or stripped == ":":
                colon.add(tid)
            if t == "," or stripped == ",":
                comma.add(tid)
            if stripped and (stripped[0].isdigit() or stripped[0] == "-"):
                digit_or_minus.add(tid)
            if t.strip() == "" and t != "":
                whitespace.add(tid)
            if not any(c in t for c in '{}[]"'):
                general_content.add(tid)

        self._open_brace = open_brace
        self._close_brace = close_brace
        self._open_bracket = open_bracket
        self._close_bracket = close_bracket
        self._quote = quote
        self._colon = colon
        self._comma = comma
        self._digit_or_minus = digit_or_minus
        self._general_content = general_content
        self._whitespace = whitespace

    @lru_cache(maxsize=256)
    def _tokenize_cached(self, text: str) -> List[str]:
        text = text.replace(" ", "Ġ").replace("\n", "Ċ")
        tokens: List[str] = list(text)

        while True:
            best_pair = None
            best_rank = float('inf')

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
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
                        and (tokens[i], tokens[i + 1]) == best_pair):
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def tokenize(self, text: str) -> List[str]:
        return self._tokenize_cached(text)

    def get_logits(self, input_ids: List[int]) -> List[float]:
        return self.model.get_logits_from_input_ids(input_ids)

    def encode(self, text: str) -> List[int]:
        tokens = self.tokenize(text)
        return [self.token_to_id[t] for t in tokens]

    def decode(self, new_id: int) -> str:
        return self.id_to_token[new_id]

    def _get_valid_mask(
        self,
        detector: JsonStopDetector,
        partial: str,
    ) -> np.ndarray:

        mask = np.ones(self._logits_size, dtype=bool)

        if detector.is_complete():
            mask[:] = False
            return mask

        if detector.in_string:
            return mask

        stripped_partial = partial.rstrip()

        for tid in self._general_content:
            raw = self.id_to_token.get(tid, "")
            t = raw.replace("Ġ", " ").replace("Ċ", "\n").strip()
            if t.isalpha() and len(t) > 1:
                mask[tid] = False

        if stripped_partial.endswith("}") and detector.object_depth == 0:
            mask[:] = False

        return mask

    def generate(self, prompt: str,
                 instructions_tokens: List[int],
                 detector: JsonStopDetector) -> str:

        output_start = "{{ \"prompt\": "
        result = output_start
        detector.feed(output_start)
        from .constrained import BOLD, GREEN, RESET
        print(f"{BOLD}{GREEN}LLM Reply (Qwen/Qwen3-0.6B):"
              f"{RESET}\n {output_start}", end="")

        prompt_tokens = self.encode(f"\n{prompt}\nResult: "
                                    f"{output_start}")
        tokens = instructions_tokens + prompt_tokens

        for _ in range(MAX_TOKENS):
            _logits = self.get_logits(tokens)
            logits = np.array(_logits, dtype=np.float32)

            valid_mask = self._get_valid_mask(detector, result)
            if valid_mask.any():
                logits[~valid_mask] = NEG_INF

            new_id = int(np.argmax(logits))
            next_word = self.decode(new_id)
            next_word = next_word.replace("Ġ", " ").replace(
                "Ċ", "\n").replace('\\', '\\\\')

            result += next_word
            print(next_word, end="", flush=True)
            tokens.append(new_id)

            if detector.feed(next_word):
                break

        return result
