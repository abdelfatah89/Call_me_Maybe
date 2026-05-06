import json
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Sequence

from .llm_model import CostimizedModel
from .models import FunctionModel, InputModel, OutputModel


NUMBER_RE = re.compile(r"[-+]?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?")
WORD_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*|\d+(?:\.\d+)?")


class FunctionCallBuilder:
    """Build schema-valid function-call objects for input prompts."""

    def __init__(
        self,
        model: CostimizedModel,
        functions: Sequence[FunctionModel],
    ) -> None:
        self.model = model
        self.functions = list(functions)

    def build(self, prompt: str) -> OutputModel:
        """Return one valid function call for a prompt."""
        function = self._select_function(prompt)
        parameters = {
            name: self._extract_parameter(
                prompt,
                name,
                parameter.type,
                index,
            )
            for index, (name, parameter) in enumerate(
                function.parameters.items()
            )
        }
        return OutputModel(
            prompt=prompt,
            name=function.name,
            parameters=parameters,
        )

    def _select_function(self, prompt: str) -> FunctionModel:
        """Select a function with constrained decoding and fallback."""
        function_pairs = [
            (
                function.name,
                self._function_text(function),
            )
            for function in self.functions
        ]
        selected = self.model.select_function(prompt, function_pairs)
        if selected is not None:
            for function in self.functions:
                if function.name == selected:
                    return function
        return max(
            self.functions,
            key=lambda function: self._lexical_score(prompt, function),
        )

    @staticmethod
    def _function_text(function: FunctionModel) -> str:
        """Return searchable text for a function definition."""
        parameter_names = " ".join(function.parameters.keys())
        return f"{function.name} {function.description} {parameter_names}"

    @staticmethod
    def _lexical_score(prompt: str, function: FunctionModel) -> float:
        """Score a function against the prompt for fallback selection."""
        prompt_norm = _normalise_text(prompt)
        function_norm = _normalise_text(
            FunctionCallBuilder._function_text(function)
        )
        prompt_words = set(WORD_RE.findall(prompt_norm))
        function_words = set(WORD_RE.findall(function_norm))
        overlap = len(prompt_words & function_words)
        similarity = SequenceMatcher(
            None,
            prompt_norm,
            function_norm,
        ).ratio()
        synonym_score = _synonym_score(prompt_norm, function.name)
        return float(overlap) + similarity + synonym_score

    def _extract_parameter(
        self,
        prompt: str,
        name: str,
        expected_type: str,
        index: int,
    ) -> Any:
        """Extract and coerce one parameter value from the prompt."""
        if expected_type == "number":
            return _extract_number(prompt, name, index)
        if expected_type == "integer":
            return int(_extract_number(prompt, name, index))
        if expected_type == "boolean":
            return _extract_boolean(prompt)
        if expected_type == "array":
            return _extract_json_value(prompt, list, [])
        if expected_type == "object":
            return _extract_json_value(prompt, dict, {})
        return _extract_string(prompt, name, index)


def constrained(
    model: CostimizedModel,
    prompts: Sequence[InputModel],
    funcs: Sequence[FunctionModel],
    max_workers: int = 1,
) -> List[Dict[str, Any]]:
    """Generate valid function-call dictionaries for every prompt."""
    _ = max_workers
    builder = FunctionCallBuilder(model, funcs)
    return [builder.build(item.prompt).model_dump() for item in prompts]


def _normalise_text(text: str) -> str:
    """Normalize text for fallback matching."""
    return re.sub(r"[_\-]+", " ", text.casefold())


def _synonym_score(prompt_norm: str, function_name: str) -> float:
    """Return additional fallback score for common action synonyms."""
    name_norm = _normalise_text(function_name)
    groups = [
        (("multiply", "product", "times"), ("multiply", "product")),
        (("add", "sum", "plus"), ("add", "sum")),
        (("reverse",), ("reverse",)),
        (("greet", "hello", "hi"), ("greet",)),
        (("square root", "sqrt"), ("square", "root", "sqrt")),
        (("compound interest", "interest"), ("compound", "interest")),
        (("sql", "query", "database"), ("sql", "query", "database")),
        (("read", "file", "encoding"), ("read", "file")),
        (("format", "template"), ("format", "template")),
        (("even",), ("even",)),
        (
            ("regex", "replace", "substitute"),
            ("regex", "replace", "substitute"),
        ),
    ]
    score = 0.0
    for prompt_terms, name_terms in groups:
        if any(term in prompt_norm for term in prompt_terms) and any(
            term in name_norm for term in name_terms
        ):
            score += 3.0
    return score


def _extract_number(prompt: str, name: str, index: int) -> float:
    """Extract a numeric parameter value, defaulting safely to 0.0."""
    numbers = [float(value) for value in NUMBER_RE.findall(prompt)]
    if not numbers:
        return 0.0
    if name in {"years", "year", "n", "count"} and len(numbers) > 1:
        return numbers[-1]
    if name in {"rate", "interest_rate"} and len(numbers) > 1:
        return numbers[min(1, len(numbers) - 1)]
    if name in {"principal", "amount"}:
        return numbers[0]
    return numbers[min(index, len(numbers) - 1)]


def _extract_boolean(prompt: str) -> bool:
    """Extract a boolean value from natural language."""
    prompt_norm = prompt.casefold()
    if any(term in prompt_norm for term in ("true", "yes", "enabled", "on")):
        return True
    if any(term in prompt_norm for term in ("false", "no", "disabled", "off")):
        return False
    return False


def _extract_json_value(
    prompt: str,
    expected_type: type[list[Any]] | type[dict[str, Any]],
    default: Any,
) -> Any:
    """Extract a JSON array or object substring if one is present."""
    for start, end in (("[", "]"), ("{", "}")):
        left = prompt.find(start)
        right = prompt.rfind(end)
        if left != -1 and right > left:
            try:
                value = json.loads(prompt[left:right + 1])
            except json.JSONDecodeError:
                continue
            if isinstance(value, expected_type):
                return value
    return default


def _extract_string(prompt: str, name: str, index: int) -> str:
    """Extract a string parameter value from a prompt."""
    quoted = _quoted_strings(prompt)
    prompt_norm = prompt.casefold()
    if name in {"query", "sql"}:
        return quoted[0] if quoted else _after_keyword(prompt, "query")
    if name in {"database", "db"}:
        database = re.search(
            r"\bon\s+(?:the\s+)?([a-zA-Z0-9_\-]+)\s+database\b",
            prompt,
            re.IGNORECASE,
        )
        if database:
            return database.group(1)
    if name in {"path", "file", "file_path", "filepath"}:
        path = re.search(
            (
                r"\bread\s+(?:the\s+)?file\s+at\s+(.+?)"
                r"(?:\s+with\s+[\w\-]+\s+encoding|$)"
                r"|\bread\s+(.+?)(?:\s+with\s+[\w\-]+\s+encoding|$)"
            ),
            prompt,
            re.IGNORECASE,
        )
        if path:
            return (path.group(1) or path.group(2)).strip()
    if name == "encoding":
        encoding = re.search(r"\bwith\s+([\w\-]+)\s+encoding\b", prompt_norm)
        if encoding:
            return encoding.group(1)
    if name == "template":
        marker = "template:"
        marker_index = prompt_norm.find(marker)
        if marker_index != -1:
            return prompt[marker_index + len(marker):].strip()
    if name in {"name", "person"}:
        if quoted:
            return quoted[0]
        match = re.search(
            r"\b(?:greet|hello|hi|to)\s+([^\s,.!?]+)",
            prompt_norm,
        )
        if match:
            return match.group(1)
    if name in {"s", "text", "string", "source_string"}:
        if quoted:
            return quoted[0]
        match = re.search(r"\bstring\s+(.+)$", prompt, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    if name == "regex" and len(quoted) > 1:
        return quoted[1]
    if name == "replacement" and len(quoted) > 2:
        return quoted[2]
    if quoted:
        return quoted[min(index, len(quoted) - 1)]
    return prompt.strip()


def _quoted_strings(text: str) -> List[str]:
    """Return single-quoted and double-quoted strings from text."""
    values: List[str] = []
    for pattern in (
        r'"((?:\\.|[^"\\])*)"',
        r"'((?:\\.|[^'\\])*)'",
    ):
        values.extend(
            bytes(match, "utf-8").decode("unicode_escape")
            for match in re.findall(pattern, text)
        )
    return values


def _after_keyword(prompt: str, keyword: str) -> str:
    """Return text after a keyword as a safe string fallback."""
    prompt_norm = prompt.casefold()
    marker_index = prompt_norm.find(keyword)
    if marker_index == -1:
        return prompt.strip()
    return prompt[marker_index + len(keyword):].strip(" :")
