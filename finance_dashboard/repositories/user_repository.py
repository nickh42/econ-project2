from __future__ import annotations

from typing import Optional

from finance_dashboard.user import User


class UserRepository:
    def __init__(self, collection):
        self.collection = collection

    def find_by_username(self, username: str) -> Optional[User]:
        doc = self.collection.find_one({"username": username})
        if not doc:
            return None
        return self._document_to_user(doc)

    def create(self, user: User) -> User:
        result = self.collection.insert_one(self._user_to_document(user))
        user._id = str(result.inserted_id)
        return user

    @staticmethod
    def _user_to_document(user: User) -> dict:
        return {
            "username": user.username,
            "password": user.password,
            "cash_balance": user.cash_balance,
            "created_at": user.created_at,
            "updated_at": user.updated_at,
        }

    @staticmethod
    def _document_to_user(doc: dict) -> User:
        user = User(
            username=doc["username"],
            password=doc["password"],
            cash_balance=float(doc.get("cash_balance", 100000.0)),
            created_at=doc.get("created_at"),
            updated_at=doc.get("updated_at"),
        )
        user._id = str(doc.get("_id"))
        return user
