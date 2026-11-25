from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

NOTIFICATION_CHANNEL_PROTOCOL = "yoao-notification-channel/v1"


class DeliveryEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_version: Literal["1.0"] = "1.0"
    event: StrictStr
    delivery_id: StrictStr
    occurred_at: StrictStr
    data: dict[str, Any] = Field(default_factory=dict)


class DeliverRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol_version: Literal["yoao-notification-channel/v1"] = NOTIFICATION_CHANNEL_PROTOCOL
    channel_id: StrictStr
    envelope: DeliveryEnvelope
    recipient_address: dict[str, Any]
    locale: StrictStr = "zh-CN"


class DeliverResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["delivered", "retryable", "permanent"]
    provider_message_id: StrictStr | None = None
    reason: StrictStr | None = None
    retry_after_seconds: StrictInt | None = Field(default=None, ge=5, le=3600)


__all__ = [
    "NOTIFICATION_CHANNEL_PROTOCOL",
    "DeliverRequest",
    "DeliverResult",
    "DeliveryEnvelope",
]
