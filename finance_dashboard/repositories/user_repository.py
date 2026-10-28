from __future__ import annotations

from datetime import datetime, timezone
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

    def update(self, user: User) -> User:
        user.updated_at = datetime.now(timezone.utc)
        result = self.collection.update_one(
            {"username": user.username},
            {"$set": self._user_state_to_document(user)},
        )
        if result.matched_count != 1:
            raise LookupError(f"Unable to find registered user {user.username} to update.")
        return user

    @staticmethod
    def _user_to_document(user: User) -> dict:
        return {
            "username": user.username,
            "password": user.password,
            **UserRepository._user_state_to_document(user),
            "created_at": user.created_at,
        }

    @staticmethod
    def _user_state_to_document(user: User) -> dict:
        return {
            "cash_balance": user.cash_balance,
            "holdings": user.holdings,
            "favorites": user.favorites,
            "trades": user.trades,
            "updated_at": user.updated_at,
        }

    @staticmethod
    def _document_to_user(doc: dict) -> User:
        user = User(
            username=doc["username"],
            password=doc["password"],
            cash_balance=float(doc.get("cash_balance", 100000.0)),
            holdings={
                str(ticker): {
                    "shares": float(position.get("shares", 0.0)),
                    "avg_cost": float(position.get("avg_cost", 0.0)),
                }
                for ticker, position in doc.get("holdings", {}).items()
            },
            favorites=[str(ticker).upper() for ticker in doc.get("favorites", [])],
            trades=list(doc.get("trades", [])),
            created_at=doc.get("created_at"),
            updated_at=doc.get("updated_at"),
        )
        user._id = str(doc.get("_id"))
        return user
