from pydantic import BaseModel
from typing import Any


class ParameterModel(BaseModel):
    type: str


class FunctionModel(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParameterModel]
    returns: ParameterModel


class InputModel(BaseModel):
    prompt: str


class OutputModel(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, Any]
