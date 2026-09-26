"""Fake ModelAdapter for core/ unit tests -- no network access, ever."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from adapters.model import ModelRequest, ModelResult


@dataclass
class FakeModelAdapter:
    """Records every request it receives and returns a scripted result. Structurally
    satisfies the `ModelAdapter` Protocol without importing or subclassing anything from
    `adapters.model` beyond its plain data types."""

    result: ModelResult
    requests: List[ModelRequest] = field(default_factory=list)

    def complete(self, request: ModelRequest) -> ModelResult:
        self.requests.append(request)
        return self.result

    @property
    def last_request(self) -> Optional[ModelRequest]:
        return self.requests[-1] if self.requests else None
