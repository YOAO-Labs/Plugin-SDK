from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

FORMAT_ADAPTER_PROTOCOL = "yoao-format-adapter/v1"


class ParsedAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class_name: StrictStr = Field(min_length=1, max_length=128)
    tool_unit_id: StrictStr = Field(min_length=1, max_length=64)
    geometry: dict[str, Any]
    source_id: StrictStr | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class ParsedItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    media_path: StrictStr = Field(min_length=1, max_length=512)
    width: StrictInt | None = Field(default=None, gt=0)
    height: StrictInt | None = Field(default=None, gt=0)
    losses: list[StrictStr] = Field(default_factory=list)
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    annotations: list[ParsedAnnotation] = Field(default_factory=list)
    mask_objects: dict[str, dict[str, Any]] = Field(default_factory=dict)


class ImportParseResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol_version: Literal["yoao-format-adapter/v1"] = FORMAT_ADAPTER_PROTOCOL
    items: list[ParsedItem] = Field(default_factory=list)


__all__ = [
    "FORMAT_ADAPTER_PROTOCOL",
    "ImportParseResponse",
    "ParsedAnnotation",
    "ParsedItem",
]
