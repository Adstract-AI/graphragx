"""Model interfaces that do not require importing optional ML dependencies."""

from __future__ import annotations

from abc import ABC


class AnswerRetrieverModel(ABC):
    """Base type for trainable answer-retriever model implementations."""

