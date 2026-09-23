from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class User:
    username: str
    password: str
    cash_balance: float = 100000.0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    _id: Optional[str] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.updated_at is None:
            self.updated_at = self.created_at
