import pytest

from finance_dashboard.repositories.user_repository import UserRepository
from finance_dashboard.services.auth_service import AuthenticationService
from finance_dashboard.user import User


class FakeUserRepository:
    def __init__(self):
        self.users = {}

    def find_by_username(self, username):
        return self.users.get(username)

    def create(self, user):
        self.users[user.username] = user
        return user


class FakeMongoClient:
    def __init__(self):
        self.user_repository = FakeUserRepository()


class FakeCollection:
    def __init__(self):
        self.documents = []

    def find_one(self, query):
        return next(
            (document for document in self.documents if document["username"] == query["username"]),
            None,
        )

    def insert_one(self, document):
        stored = {**document, "_id": "test-user-id"}
        self.documents.append(stored)
        return type("InsertResult", (), {"inserted_id": stored["_id"]})()

    def update_one(self, query, update):
        document = self.find_one(query)
        if document is None:
            return type("UpdateResult", (), {"matched_count": 0})()
        document.update(update["$set"])
        return type("UpdateResult", (), {"matched_count": 1})()


def test_user_repository_persists_cash_holdings_favorites_and_trades():
    repository = UserRepository(FakeCollection())
    user = User(username="user@example.com", password="hashed-password")
    repository.create(user)

    user.cash_balance = 97500.0
    user.holdings = {"AAPL": {"shares": 5.0, "avg_cost": 500.0}}
    user.favorites = ["AAPL"]
    user.trades = [
        {
            "side": "buy",
            "ticker": "AAPL",
            "shares": 5.0,
            "price": 500.0,
            "total": 2500.0,
        }
    ]
    repository.update(user)

    restored_user = repository.find_by_username("user@example.com")
    assert restored_user is not None
    assert restored_user.cash_balance == 97500.0
    assert restored_user.holdings == {"AAPL": {"shares": 5.0, "avg_cost": 500.0}}
    assert restored_user.favorites == ["AAPL"]
    assert restored_user.trades == user.trades


def test_register_user_with_valid_email_and_matching_passwords():
    service = AuthenticationService(repository=FakeMongoClient().user_repository)

    user = service.register_user("test@example.com", "StrongPass123!", "StrongPass123!")

    assert isinstance(user, User)
    assert user.username == "test@example.com"
    assert user.cash_balance == 100000
    assert user.password != "StrongPass123!"
    assert user.created_at is not None
    assert user.updated_at is not None


def test_register_user_rejects_invalid_email():
    service = AuthenticationService(repository=FakeMongoClient().user_repository)

    with pytest.raises(ValueError, match="valid email"):
        service.register_user("not-an-email", "StrongPass123!", "StrongPass123!")


def test_login_user_accepts_valid_credentials():
    repository = FakeUserRepository()
    service = AuthenticationService(repository=repository)
    service.register_user("user@example.com", "StrongPass123!", "StrongPass123!")

    user = service.login_user("user@example.com", "StrongPass123!")

    assert user.username == "user@example.com"


def test_login_user_rejects_invalid_password():
    repository = FakeUserRepository()
    service = AuthenticationService(repository=repository)
    service.register_user("user@example.com", "StrongPass123!", "StrongPass123!")

    with pytest.raises(ValueError, match="Incorrect password"):
        service.login_user("user@example.com", "WrongPass123!")


def test_register_user_rejects_non_matching_passwords():
    service = AuthenticationService(repository=FakeMongoClient().user_repository)

    with pytest.raises(ValueError, match="match"):
        service.register_user("user@example.com", "StrongPass123!", "DifferentPass123!")


def test_duplicate_user_registration_is_rejected():
    repository = FakeUserRepository()
    service = AuthenticationService(repository=repository)

    service.register_user("user@example.com", "StrongPass123!", "StrongPass123!")

    with pytest.raises(ValueError, match="already exists"):
        service.register_user("user@example.com", "AnotherPass123!", "AnotherPass123!")
