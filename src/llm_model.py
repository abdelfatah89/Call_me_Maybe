from llm_sdk import Small_LLM_Model
from typing import Dict
from .json_stop_detector import JsonStopDetector


MAX_TOKENS = 96


class CostimizedModel:
    def __init__(self) -> None:
        self.model = Small_LLM_Model()
        self.id_to_token: Dict[int, str] = self._build_id_to_token()

    def _build_id_to_token(self) -> Dict[int, str]:
        # Pre-compute a flat ``id -> raw BPE token`` map once at startup so the
        # per-step decode is an O(1) dict lookup with no caching overhead.
        tokenizer = self.model._tokenizer
        vocab = tokenizer.get_vocab()
        return {int(idx): tok for tok, idx in vocab.items()}

    def encode(self, text: str) -> list[int]:
        return self.model.encode_ids(text)

    def decode_token(self, token_id: int) -> str:
        token = self.id_to_token.get(token_id)
        if token is None:
            token = self.model.decode_id(token_id)
        # GPT-2/Qwen BPE uses U+0120 ("G-with-dot") for leading spaces and
        # U+010A for newlines.
        return token.replace("\u0120", " ").replace("\u010A", "\n")

    def generate(self, prompt: str, detector: JsonStopDetector) -> str:
        result = "{"
        detector.feed("{")

        # Encode the full prompt once via the SDK's real tokenizer. This avoids
        # the previous O(n^2) BPE merge loop that re-tokenised the whole
        # growing context at every generation step.
        input_ids = self.encode(prompt)

        # First forward pass primes the KV cache over the entire prompt; later
        # steps feed only the single new token, turning each step from a full
        # O(context) forward into a constant-time append.
        next_id, past_key_values = self.model.argmax_next_token(input_ids)

        for _ in range(MAX_TOKENS):
            next_word = self.decode_token(next_id)
            result += next_word
            if detector.feed(next_word):
                break

            next_id, past_key_values = self.model.argmax_next_token(
                [next_id], past_key_values=past_key_values
            )

        return result
