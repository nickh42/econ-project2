from __future__ import annotations

from pymongo import MongoClient

from finance_dashboard.repositories.user_repository import UserRepository


class MongoDB:
    def __init__(self, uri: str, database_name: str = "trading_simulator"):
        self.client = MongoClient(uri)
        self.database = self.client[database_name]

    @property
    def users(self):
        return self.database["users"]

    @property
    def user_repository(self) -> UserRepository:
        return UserRepository(self.users)
