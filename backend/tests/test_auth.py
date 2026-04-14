"""Tests for authentication endpoints."""
import pytest
from fastapi import status


class TestRegister:
    """Test user registration."""

    def test_register_success(self, client):
        """Test successful user registration."""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "newuser@example.com",
                "password": "securepassword123",
                "full_name": "New User",
            },
        )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["email"] == "newuser@example.com"
        assert data["full_name"] == "New User"
        assert "id" in data
        assert "access_token" in data

    def test_register_duplicate_email(self, client, test_user):
        """Test registration with duplicate email."""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": test_user.email,
                "password": "differentpassword123",
                "full_name": "Different User",
            },
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "already exists" in data.get("detail", "").lower()

    def test_register_invalid_email(self, client):
        """Test registration with invalid email format."""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "not-an-email",
                "password": "securepassword123",
                "full_name": "Test User",
            },
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_register_weak_password(self, client):
        """Test registration with weak password."""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "user@example.com",
                "password": "weak",
                "full_name": "Test User",
            },
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST


class TestLogin:
    """Test user login."""

    def test_login_success(self, client, test_user):
        """Test successful login."""
        response = client.post(
            "/api/v1/auth/login",
            data={
                "username": test_user.email,
                "password": "testpassword123",
            },
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["access_token"]
        assert data["token_type"] == "bearer"

    def test_login_wrong_password(self, client, test_user):
        """Test login with wrong password."""
        response = client.post(
            "/api/v1/auth/login",
            data={
                "username": test_user.email,
                "password": "wrongpassword",
            },
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_login_nonexistent_user(self, client):
        """Test login with non-existent user."""
        response = client.post(
            "/api/v1/auth/login",
            data={
                "username": "nonexistent@example.com",
                "password": "anypassword",
            },
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_login_inactive_user(self, client, db_session, test_user):
        """Test login with inactive user."""
        # Deactivate the user
        test_user.is_active = False
        db_session.add(test_user)
        db_session.commit()

        response = client.post(
            "/api/v1/auth/login",
            data={
                "username": test_user.email,
                "password": "testpassword123",
            },
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestGetMe:
    """Test get current user endpoint."""

    def test_get_me_authenticated(self, client, auth_headers, test_user):
        """Test getting current user when authenticated."""
        response = client.get(
            "/api/v1/auth/me",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == str(test_user.id)
        assert data["email"] == test_user.email
        assert data["full_name"] == test_user.full_name

    def test_get_me_unauthenticated(self, client):
        """Test getting current user without authentication."""
        response = client.get("/api/v1/auth/me")

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_get_me_invalid_token(self, client):
        """Test getting current user with invalid token."""
        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer invalid_token"},
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestTokenRefresh:
    """Test token refresh."""

    def test_refresh_token_success(self, client, test_user):
        """Test successful token refresh."""
        # First login
        login_response = client.post(
            "/api/v1/auth/login",
            data={
                "username": test_user.email,
                "password": "testpassword123",
            },
        )

        assert login_response.status_code == status.HTTP_200_OK
        token = login_response.json()["access_token"]

        # Refresh token
        response = client.post(
            "/api/v1/auth/refresh",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_refresh_token_unauthenticated(self, client):
        """Test token refresh without authentication."""
        response = client.post("/api/v1/auth/refresh")

        assert response.status_code == status.HTTP_401_UNAUTHORIZED
