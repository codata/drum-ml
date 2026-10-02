"""Export Target Dataset Models (OpenAI, ShareGPT, Anthropic, DPO)."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from drum_ml.models.scaffolds import ArchetypeType


class OpenAIChatMessage(BaseModel):
    role: str  # "system" | "user" | "assistant"
    content: str


class OpenAIChatRecord(BaseModel):
    id: str
    archetype: ArchetypeType
    entity_uri: str
    messages: list[OpenAIChatMessage]
    metadata: dict[str, Any] = Field(default_factory=dict)


class ShareGPTTurn(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    from_: str = Field(..., alias="from")  # "human" | "gpt" | "system"
    value: str


class ShareGPTRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    conversations: list[ShareGPTTurn]
    metadata: dict[str, Any] = Field(default_factory=dict)


class DPOPreferenceRecord(BaseModel):
    """Direct Preference Optimization (DPO) triple with verified prompt, chosen, and rejected answers."""

    id: str
    archetype: ArchetypeType
    entity_uri: str
    prompt: str
    chosen: str
    rejected: str
    rejection_reason: str
    metadata: dict[str, Any] = Field(default_factory=dict)
