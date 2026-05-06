from typing import List


class JsonStopDetectorError(Exception):
    pass


class JsonStopDetector:
    """Track generated text and report when one JSON object is complete."""

    def __init__(self) -> None:
        self.started = False
        self.in_string = False
        self.escape = False
        self.object_depth = 0
        self.array_depth = 0
        self.buffer: List[str] = []

    def feed(self, text: str) -> bool:
        for char in text:
            self.buffer.append(char)

            if not self.started:
                if char.isspace():
                    continue
                if char == "{":
                    self.started = True
                    self.object_depth = 1
                    continue
                raise JsonStopDetectorError(
                    f"JSON must start with {'{'!r}, got {char!r}")

            if self.in_string:
                if self.escape:
                    self.escape = False
                elif char == "\\":
                    self.escape = True
                elif char == '"':
                    self.in_string = False
                continue

            if char == '"':
                self.in_string = True
            elif char == "{":
                self.object_depth += 1
            elif char == "}":
                self.object_depth -= 1
                if self.object_depth < 0:
                    raise JsonStopDetectorError("Too many closing braces")
            elif char == "[":
                self.array_depth += 1
            elif char == "]":
                self.array_depth -= 1
                if self.array_depth < 0:
                    raise JsonStopDetectorError("Too many closing brackets")

            if (
                self.started
                and not self.in_string
                and self.object_depth == 0
                and self.array_depth == 0
            ):
                return True

        return False

    def get_json_text(self) -> str:
        return "".join(self.buffer)
