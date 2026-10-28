from __future__ import annotations

from contextlib import nullcontext
from typing import Any

from app.config import Settings


class Observability:
    def __init__(self, settings: Settings) -> None:
        self.capture_content = settings.observability_capture_content
        self.client = None
        if settings.observability_enabled:
            if not settings.langfuse_public_key or not settings.langfuse_secret_key:
                raise ValueError("Langfuse keys are required when observability is enabled")
            try:
                from langfuse import Langfuse
            except ImportError as exc:
                raise RuntimeError("Install langfuse for observability") from exc
            self.client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                base_url=settings.langfuse_base_url,
                environment=settings.environment,
            )

    def content(self, value: Any) -> Any:
        return value if self.capture_content else None

    def observe(self, name: str, *, kind: str = "span", **kwargs):
        if self.client is None:
            return nullcontext(NoopObservation())
        return self.client.start_as_current_observation(name=name, as_type=kind, **kwargs)

    def flush(self) -> None:
        if self.client is not None:
            self.client.flush()

    def record_feedback(self, trace_id: str, rating: bool, comment: str | None) -> bool:
        if self.client is None:
            return False
        self.client.create_score(
            trace_id=trace_id,
            name="user-feedback",
            value=1 if rating else 0,
            data_type="BOOLEAN",
            comment=comment,
        )
        return True


class NoopObservation:
    def update(self, **kwargs) -> None:
        pass
