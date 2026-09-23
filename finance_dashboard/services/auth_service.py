from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone

from finance_dashboard.user import User


class AuthenticationService:
    def __init__(self, repository):
        self.repository = repository

    @staticmethod
    def _is_valid_email(username: str) -> bool:
        return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", username))

    @staticmethod
    def _hash_password(password: str) -> str:
        return hashlib.sha256(password.encode("utf-8")).hexdigest()

    def register_user(self, username: str, password: str, confirm_password: str) -> User:
        if not self._is_valid_email(username):
            raise ValueError("Username must be a valid email address.")

        if password != confirm_password:
            raise ValueError("Passwords do not match.")

        if self.repository.find_by_username(username):
            raise ValueError("User with this email already exists.")

        user = User(
            username=username,
            password=self._hash_password(password),
            cash_balance=100000.0,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        return self.repository.create(user)
