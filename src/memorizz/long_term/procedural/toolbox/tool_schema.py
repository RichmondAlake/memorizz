# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

from __future__ import annotations

from typing import List

from pydantic import BaseModel


class ParameterSchema(BaseModel):
    """
    A schema for the parameter.
    """

    name: str
    description: str
    type: str
    required: bool


class FunctionSchema(BaseModel):
    """
    A schema for the function.
    """

    name: str
    description: str
    parameters: list[ParameterSchema]
    required: List[str]
    queries: List[str]


class ToolSchemaType(BaseModel):
    """
    A schema for the tool.
    This can be the OpenAI function calling schema or Google function calling schema.
    """

    type: str
    function: FunctionSchema
