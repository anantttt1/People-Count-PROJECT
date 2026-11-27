"""Pydantic models: validated settings and WebSocket/REST payloads."""
from __future__ import annotations

import re
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def clean_name(value: str, default: str = "Untitled session") -> str:
    value = _CONTROL.sub("", value or "")
    value = " ".join(value.split())[:80]
    return value or default


class LineModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x1: float = Field(0.05, ge=0.0, le=1.0)
    y1: float = Field(0.55, ge=0.0, le=1.0)
    x2: float = Field(0.95, ge=0.0, le=1.0)
    y2: float = Field(0.55, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _distinct(self) -> "LineModel":
        if abs(self.x1 - self.x2) < 0.02 and abs(self.y1 - self.y2) < 0.02:
            raise ValueError("counting line is too short")
        return self


class DetectionSettings(BaseModel):
    """Everything that tunes detection, tracking and counting. Admin defaults use the same model."""

    model_config = ConfigDict(extra="forbid")

    confidence: float = Field(0.40, ge=0.10, le=0.95, description="Score needed to start / confirm a track")
    low_confidence: float = Field(0.15, ge=0.05, le=0.90, description="Lower score kept only to follow occluded people")
    iou: float = Field(0.50, ge=0.20, le=0.90, description="Non-max suppression IoU threshold")
    min_box_height: float = Field(0.03, ge=0.0, le=0.8, description="Ignore people shorter than this share of the frame")
    lost_track_seconds: float = Field(2.0, ge=0.3, le=10.0, description="How long a hidden person keeps their ID")
    line: LineModel = Field(default_factory=LineModel)
    invert_direction: bool = False
    anchor: Literal["center", "bottom"] = "center"
    max_capacity: int | None = Field(50, ge=1, le=100_000)
    show_confidence: bool = True
    audio_alerts: bool = False

    @model_validator(mode="after")
    def _thresholds(self) -> "DetectionSettings":
        if self.low_confidence >= self.confidence:
            raise ValueError("low_confidence must be lower than confidence")
        return self


class StartMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["start"]
    name: str = Field("Untitled session", max_length=200)
    settings: DetectionSettings | None = None

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        return clean_name(v)


class ConfigMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["config"]
    settings: DetectionSettings


class ControlMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["reset", "stop", "ping", "pause", "resume"]


ClientMessage = Annotated[Union[StartMessage, ConfigMessage, ControlMessage], Field(discriminator="type")]
client_message_adapter: TypeAdapter = TypeAdapter(ClientMessage)


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    token: str = Field(..., min_length=1, max_length=512)
