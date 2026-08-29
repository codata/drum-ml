"""Export Target Dataset Models (OpenAI, ShareGPT, Anthropic, DPO)."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from drum_ml.models.scaffolds import ArchetypeType, PersonaType


class OpenAIChatMessage(BaseModel):
    role: str # "system" | "user" | "assistant"
    content: str


class OpenAIChatRecord(BaseModel):
    id: str
    archetype: ArchetypeType
    entity_uri: str
    messages: List[OpenAIChatMessage]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ShareGPTTurn(BaseModel):
    from_: str = Field(..., alias="from") # "human" | "gpt" | "system"
    value: str


class ShareGPTRecord(BaseModel):
    id: str
    conversations: List[ShareGPTTurn]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DPOPreferenceRecord(BaseModel):
    """Direct Preference Optimization (DPO) triple with verified prompt, chosen, and rejected answers."""
    id: str
    archetype: ArchetypeType
    entity_uri: str
    prompt: str
    chosen: str
    rejected: str
    rejection_reason: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
