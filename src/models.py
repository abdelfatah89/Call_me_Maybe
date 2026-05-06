from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator


class ParameterModel(BaseModel):
    """Describe a function parameter or return type."""

    model_config = ConfigDict(extra="ignore")

    type: str

    @field_validator("type")
    @classmethod
    def normalise_type(cls, value: str) -> str:
        """Normalize JSON-schema-like types to a lower-case form."""
        cleaned = value.strip().lower()
        if not cleaned:
            raise ValueError("type must not be empty")
        return cleaned


class FunctionModel(BaseModel):
    """Represent one function available to the function-calling system."""

    model_config = ConfigDict(extra="ignore")

    name: str
    description: str
    parameters: dict[str, ParameterModel]
    returns: ParameterModel


class InputModel(BaseModel):
    """Represent one natural-language prompt to process."""

    model_config = ConfigDict(extra="forbid")

    prompt: str


class OutputModel(BaseModel):
    """Represent one generated function call."""

    model_config = ConfigDict(extra="forbid")

    prompt: str
    name: str
    parameters: dict[str, Any]
