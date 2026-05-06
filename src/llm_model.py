import os
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from llm_sdk import Small_LLM_Model

from .json_stop_detector import JsonStopDetector, JsonStopDetectorError


MAX_TOKENS = 96
MODEL_LOAD_TIMEOUT_SECONDS = 45.0


class ModelUnavailableError(Exception):
    """Raised when the optional local model cannot be used."""

    pass


class CostimizedModel:
    """Safe wrapper around the provided SDK with bounded generation helpers."""

    def __init__(self) -> None:
        self.model: Optional[Small_LLM_Model] = None
        self.load_error: Optional[str] = None
        if os.getenv("CALL_ME_MAYBE_USE_LLM", "1").lower() in {
            "0",
            "false",
            "no",
        }:
            self.load_error = "model loading disabled by environment"
            return
        try:
            timeout = float(
                os.getenv(
                    "CALL_ME_MAYBE_MODEL_LOAD_TIMEOUT",
                    str(MODEL_LOAD_TIMEOUT_SECONDS),
                )
            )
        except ValueError:
            timeout = MODEL_LOAD_TIMEOUT_SECONDS
        self._load_with_timeout(timeout)

    def _load_with_timeout(self, timeout: float) -> None:
        """Load the SDK model without letting startup block forever."""
        thread = threading.Thread(target=self._load_model, daemon=True)
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            self.load_error = f"model load exceeded {timeout:.1f} seconds"

    def _load_model(self) -> None:
        """Load the SDK model and record errors instead of raising them."""
        try:
            self.model = Small_LLM_Model()
        except Exception as exc:
            self.load_error = str(exc)

    def is_available(self) -> bool:
        """Return whether the underlying SDK model loaded successfully."""
        return self.model is not None

    def _require_model(self) -> Small_LLM_Model:
        """Return the SDK model or raise a clear model-unavailable error."""
        if self.model is None:
            message = self.load_error or "model is not loaded"
            raise ModelUnavailableError(message)
        return self.model

    def encode(self, text: str) -> List[int]:
        """Encode text with the public SDK tokenizer API."""
        tensor = self._require_model().encode(text)
        ids = tensor.tolist()
        if ids and isinstance(ids[0], list):
            return [int(token_id) for token_id in ids[0]]
        return [int(token_id) for token_id in ids]

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs with the public SDK tokenizer API."""
        return self._require_model().decode(ids)

    def get_logits(self, input_ids: List[int]) -> List[float]:
        """Get next-token logits from the public SDK model API."""
        return self._require_model().get_logits_from_input_ids(input_ids)

    def generate(self, prompt: str, detector: JsonStopDetector) -> str:
        """Generate text and stop immediately after one complete JSON object."""
        tokens = self.encode(prompt)
        result = ""
        if prompt.rstrip().endswith("{"):
            result = "{"
            detector.feed("{")

        for _ in range(MAX_TOKENS):
            logits = self.get_logits(tokens)
            new_id = int(np.argmax(logits))
            next_text = self.decode([new_id])
            result += next_text
            tokens.append(new_id)
            try:
                if detector.feed(next_text):
                    break
            except JsonStopDetectorError as exc:
                raise ModelUnavailableError(str(exc)) from exc

        return result.strip()

    def select_function(
        self,
        prompt: str,
        functions: Sequence[Tuple[str, str]],
    ) -> Optional[str]:
        """Choose a function name using trie-constrained LLM decoding."""
        candidates = [name for name, _ in functions]
        if not candidates:
            return None
        try:
            context = self._selection_prompt(prompt, functions)
            context_ids = self.encode(context)
            trie = self._build_candidate_trie(candidates)
            generated: List[int] = []
            node = trie
            for _ in range(max(len(self.encode(name)) for name in candidates)):
                children: Dict[int, Dict[str, Any]] = node.get(
                    "children",
                    {},
                )
                if not children:
                    break
                logits = self.get_logits(context_ids + generated)
                next_id = max(
                    children.keys(),
                    key=lambda token_id: logits[int(token_id)],
                )
                generated.append(int(next_id))
                node = children[int(next_id)]
                if node.get("name") is not None:
                    return str(node["name"])
            return None
        except Exception:
            return None

    @staticmethod
    def _selection_prompt(
        prompt: str,
        functions: Sequence[Tuple[str, str]],
    ) -> str:
        """Build a short prompt for function-name selection."""
        lines = [
            "Choose the single best function for the user request.",
            "Return only the exact function name.",
            "Available functions:",
        ]
        for name, description in functions:
            lines.append(f"- {name}: {description}")
        lines.extend(["User request:", prompt, "Function name: "])
        return "\n".join(lines)

    def _build_candidate_trie(
        self,
        candidates: Sequence[str],
    ) -> Dict[str, Any]:
        """Build a token trie from allowed function names."""
        root: Dict[str, Any] = {"children": {}}
        for name in candidates:
            node = root
            for token_id in self.encode(name):
                children = node.setdefault("children", {})
                node = children.setdefault(int(token_id), {"children": {}})
            node["name"] = name
        return root
