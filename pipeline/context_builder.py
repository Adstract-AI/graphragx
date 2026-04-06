"""Centralized utilities for building step contexts between pipeline stages."""

from __future__ import annotations

from typing import Optional

from pipeline.abstract import StepContext, StepResult


class StepContextBuilder:
    """Creates step contexts from prior step outputs."""

    def create_context(
            self,
            result: Optional[StepResult],
            outcome: bool = True,
            exception=None,
    ) -> StepContext:
        """Wrap a previous result into a context for the next step."""
        return StepContext(
            result=result,
            outcome=outcome,
            exception=exception)
