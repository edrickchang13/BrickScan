# BrickScan API Endpoints Reference

This document specifies the backend API endpoints that the mobile app expects.

## Base URL

All endpoints are relative to: `$EXPO_PUBLIC_API_URL` (default: `http://localhost:3000/api`)

## Authentication Endpoints

### POST /auth/login
Register user credentials and receive JWT token.

**Request:**
```json
{
  "email": "user@example.com",
  "password": "password123"
}
```

**Response (200):**
```json
{
  "token": "eyJhbGciOiJIUzI1NiIs...",
  "user": {
    "id": "user123",
    "email": "user@example.com"
  }
}
```

### POST /auth/register
Create a new user account.

**Request:**
```json
{
  "email": "user@example.com",
  "password": "password123"
}
```

**Response (201):**
```json
{
  "token": "eyJhbGciOiJIUzI1NiIs...",
  "user": {
    "id": "user123",
    "email": "user@example.com"
  }
}
```

## Parts Endpoints

### GET /parts/search
Search for LEGO parts by query string.

**Query Parameters:**
- `q` (string, required): Search query (part name or number)

**Response (200):**
```json
{
  "data": [
    {
      "id": "part123",
      "partNum": "3001",
      "name": "Brick 2x4",
      "category": "Bricks",
      "imageUrl": "https://..."
    }
  ]
}
```

### GET /parts/{partNum}
Get details for a specific part.

**Response (200):**
```json
{
  "data": {
    "id": "part123",
    "partNum": "3001",
    "name": "Brick 2x4",
    "category": "Bricks",
    "imageUrl": "https://..."
  }
}
```

## Sets Endpoints

### GET /sets/search
Search for LEGO sets.

**Query Parameters:**
- `q` (string): Search query (set name or number)
- `theme` (string, optional): Filter by theme

**Response (200):**
```json
{
  "data": [
    {
      "setNum": "75192",
      "name": "Millennium Falcon",
      "year": 2017,
      "theme": "Star Wars",
      "numParts": 7541,
      "imageUrl": "https://..."
    }
  ]
}
```

### GET /sets/{setNum}
Get full details for a specific set including all parts.

**Response (200):**
```json
{
  "data": {
    "setNum": "75192",
    "name": "Millennium Falcon",
    "year": 2017,
    "theme": "Star Wars",
    "numParts": 7541,
    "imageUrl": "https://...",
    "description": "...",
    "parts": [
      {
        "partNum": "3001",
        "partName": "Brick 2x4",
        "colorId": "1",
        "colorName": "Red",
        "colorHex": "#C40026",
        "quantity": 5,
        "imageUrl": "https://..."
      }
    ]
  }
}
```

### GET /sets/{setNum}/parts
Get all parts for a specific set.

**Response (200):**
```json
{
  "data": [
    {
      "partNum": "3001",
      "partName": "Brick 2x4",
      "colorId": "1",
      "colorName": "Red",
      "colorHex": "#C40026",
      "quantity": 5,
      "imageUrl": "https://..."
    }
  ]
}
```

## Scan Endpoint

### POST /scan
Scan a LEGO piece from a base64 image.

**Request:**
```json
{
  "image": "iVBORw0KGgoAAAANSUhEUgAAAA1AAAANCAYAAABycvieAAAABmIlDQ..."
}
```

**Response (200):**
```json
{
  "data": {
    "predictions": [
      {
        "partNum": "3001",
        "partName": "Brick 2x4",
        "colorId": "1",
        "colorName": "Red",
        "colorHex": "#C40026",
        "confidence": 0.92,
        "imageUrl": "https://..."
      },
      {
        "partNum": "3002",
        "partName": "Brick 2x3",
        "colorId": "1",
        "colorName": "Red",
        "colorHex": "#C40026",
        "confidence": 0.78,
        "imageUrl": "https://..."
      }
    ]
  }
}
```

## Inventory Endpoints

### GET /inventory
Get user's complete inventory (requires auth).

**Headers:** `Authorization: Bearer {token}`

**Response (200):**
```json
{
  "data": [
    {
      "id": "inv123",
      "partNum": "3001",
      "partName": "Brick 2x4",
      "colorId": "1",
      "colorName": "Red",
      "colorHex": "#C40026",
      "quantity": 10,
      "imageUrl": "https://...",
      "createdAt": "2024-01-15T10:30:00Z",
      "updatedAt": "2024-01-15T10:30:00Z"
    }
  ]
}
```

### POST /inventory
Add a piece to inventory (requires auth).

**Request:**
```json
{
  "partNum": "3001",
  "colorId": "1",
  "quantity": 5
}
```

**Response (201):**
```json
{
  "data": {
    "id": "inv123",
    "partNum": "3001",
    "partName": "Brick 2x4",
    "colorId": "1",
    "colorName": "Red",
    "colorHex": "#C40026",
    "quantity": 5,
    "imageUrl": "https://...",
    "createdAt": "2024-01-15T10:30:00Z",
    "updatedAt": "2024-01-15T10:30:00Z"
  }
}
```

### PATCH /inventory/{id}
Update quantity of an inventory item (requires auth).

**Request:**
```json
{
  "quantity": 15
}
```

**Response (200):**
```json
{
  "data": {
    "id": "inv123",
    "partNum": "3001",
    "partName": "Brick 2x4",
    "colorId": "1",
    "colorName": "Red",
    "colorHex": "#C40026",
    "quantity": 15,
    "imageUrl": "https://...",
    "createdAt": "2024-01-15T10:30:00Z",
    "updatedAt": "2024-01-15T10:30:00Z"
  }
}
```

### DELETE /inventory/{id}
Delete an inventory item (requires auth).

**Response (204):** No content

### GET /inventory/export/csv
Export inventory as CSV file (requires auth).

**Response (200):** CSV file content as text

## Build Check Endpoint

### POST /builds/check
Compare user's inventory to a set and get build progress.

**Request:**
```json
{
  "setNum": "75192"
}
```

**Response (200):**
```json
{
  "data": {
    "setNum": "75192",
    "setName": "Millennium Falcon",
    "percentComplete": 73,
    "have": 5487,
    "total": 7541,
    "missing": 2054,
    "haveParts": [
      {
        "partNum": "3001",
        "partName": "Brick 2x4",
        "colorId": "1",
        "colorName": "Red",
        "colorHex": "#C40026",
        "quantity": 10,
        "imageUrl": "https://..."
      }
    ],
    "missingParts": [
      {
        "partNum": "3002",
        "partName": "Brick 2x3",
        "colorId": "1",
        "colorName": "Red",
        "colorHex": "#C40026",
        "quantityNeeded": 5,
        "quantityHave": 2,
        "imageUrl": "https://..."
      }
    ]
  }
}
```

## BrickLink Endpoint

### POST /bricklink/wanted-list
Generate a BrickLink wanted list XML for missing parts.

**Request:**
```json
{
  "setNum": "75192",
  "condition": "N"
}
```

**Response (200):**
```xml
<?xml version="1.0" encoding="UTF-8"?>
<INVENTORY>
  <ITEM>
    <ITEMTYPE>P</ITEMTYPE>
    <ITEMID>3002</ITEMID>
    <COLOR>1</COLOR>
    <MINQTY>5</MINQTY>
    <CONDITION>N</CONDITION>
    <REMARKS>Missing parts from set 75192</REMARKS>
  </ITEM>
</INVENTORY>
```

## Error Responses

All errors follow this format:

**400 Bad Request:**
```json
{
  "success": false,
  "error": "Invalid request",
  "message": "Query parameter 'q' is required"
}
```

**401 Unauthorized:**
```json
{
  "success": false,
  "error": "Unauthorized",
  "message": "Invalid or expired token"
}
```

**404 Not Found:**
```json
{
  "success": false,
  "error": "Not found",
  "message": "Set with number 75192 not found"
}
```

**500 Internal Server Error:**
```json
{
  "success": false,
  "error": "Server error",
  "message": "An unexpected error occurred"
}
```

## Authentication

All endpoints requiring authentication expect the JWT token in the Authorization header:

```
Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
```

The mobile app automatically injects this header via axios interceptor.

If a 401 response is received, the app will:
1. Delete stored token
2. Clear user state
3. Redirect to login screen

## Rate Limiting

No specific rate limits are mentioned, but implement reasonable limits:
- 100 requests per minute per IP
- 10 requests per minute per token for auth endpoints

## Notes

- Image URLs should be absolute URLs (can be from Rebrickable CDN)
- Color codes should be in hex format with leading #
- Timestamps should be ISO 8601 format
- Quantities are integers
- Confidence values are decimals between 0 and 1
- All responses should include `data` wrapper for consistency
