"""Agent Library (component ①): cheap storage of many 'quirky' agents."""
from .base import AgentBackend, AgentResponse, AgentSpec
from .library import AgentLibrary
from .mock import MockBackend
from .personas import BUILTIN_PERSONAS, build_starter_library

__all__ = [
    "AgentSpec",
    "AgentResponse",
    "AgentBackend",
    "AgentLibrary",
    "MockBackend",
    "BUILTIN_PERSONAS",
    "build_starter_library",
]
