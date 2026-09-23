import pytest

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
