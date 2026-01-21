"""
rlm - Recursive Language Model exploration for regulatory compliance
---------------------------------------------------------------------
Agentic document exploration using LLM + tools for multi-hop retrieval.
"""

from .paragraph_index import (
    ParagraphSpan,
    Section,
    DocumentParagraphIndex,
    GlobalParagraphIndex,
    build_all_indexes,
)
from .rlm_environment import (
    RLMEnvironment,
    RLM_TOOLS,
    SpanRef,
    DocInfo,
)
from .rlm_controller import RLMController

__all__ = [
    # Paragraph indexing
    "ParagraphSpan",
    "Section",
    "DocumentParagraphIndex",
    "GlobalParagraphIndex",
    "build_all_indexes",
    # Environment
    "RLMEnvironment",
    "RLM_TOOLS",
    "SpanRef",
    "DocInfo",
    # Controller
    "RLMController",
]
