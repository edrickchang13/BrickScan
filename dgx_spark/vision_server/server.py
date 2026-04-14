"""
BrickScan Local Vision Inference Server

Runs on DGX Spark and exposes a REST API for LEGO piece identification
using local models (LLaVA via Ollama) instead of cloud APIs.

Compatible with the BrickScan backend - set VISION_BACKEND=dgx in .env
to use this instead of Gemini (zero code changes needed, same API interface).

Usage:
    uvicorn server:app --host 0.0.0.0 --port 8001

Or with auto-reload during development:
    uvicorn server:app --host 0.0.0.0 --port 8001 --reload
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
import httpx
import base64
import json
import re
import logging
import time
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="BrickScan Vision Server",
    description="Local LEGO piece identification using Ollama and LLaVA",
    version="1.0.0"
)

# Configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
INFERENCE_TIMEOUT = float(os.getenv("INFERENCE_TIMEOUT", "120"))
MAX_INFERENCE_TIME = float(os.getenv("MAX_INFERENCE_TIME", "180"))  # 3 minutes hard limit

# Request/Response Models
class IdentifyRequest(BaseModel):
    """Request to identify a LEGO piece from image"""
    image_base64: str = Field(..., description="Base64-encoded image data")
    top_k: int = Field(3, ge=1, le=5, description="Number of predictions to return")
    model: Optional[str] = Field(None, description="Optional: specify Ollama model to use")

class PartPrediction(BaseModel):
    """Single prediction for a LEGO part"""
    part_num: str = Field(..., description="BrickLink part number")
    part_name: str = Field(..., description="Official part name")
    color_name: str = Field(..., description="Official LEGO color name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")

class IdentifyResponse(BaseModel):
    """Response with piece identification predictions"""
    predictions: list[PartPrediction]
    model_used: str = Field(..., description="Which Ollama model was used")
    processing_time_ms: float = Field(..., description="Total inference time in milliseconds")
    timestamp: str = Field(..., description="ISO 8601 timestamp")

class HealthResponse(BaseModel):
    """Server health status"""
    status: str
    available_models: list[str] = []
    ollama_url: str = ""
    error: Optional[str] = None
    timestamp: str = ""

class Model(BaseModel):
    """Ollama model info"""
    name: str
    digest: str
    size: int
    modified_at: str

# System prompt for vision identification
LEGO_IDENTIFICATION_PROMPT = """You are an expert LEGO parts identifier with encyclopedic knowledge of the BrickLink and Rebrickable catalogs.

Analyze this image of a LEGO piece and identify it precisely. Return ONLY a valid JSON array (no other text, no markdown, no code blocks) with your top 3 predictions ordered by confidence:

[
  {
    "part_num": "3001",
    "part_name": "Brick 2 x 4",
    "color_name": "Red",
    "confidence": 0.95
  },
  {
    "part_num": "3003",
    "part_name": "Brick 1 x 2",
    "color_name": "Red",
    "confidence": 0.03
  },
  {
    "part_num": "3004",
    "part_name": "Brick 1 x 2",
    "color_name": "Dark Red",
    "confidence": 0.02
  }
]

Rules:
- part_num MUST be a real BrickLink/Rebrickable part number (3-5 digits)
- part_name MUST be the official BrickLink name (e.g., "Brick 2 x 4", "Plate 1 x 2")
- color_name MUST be an official LEGO color (Red, Dark Red, Blue, White, Black, Dark Bluish Gray, Light Bluish Gray, Yellow, Green, Brown, Tan, Pink, etc.)
- confidence MUST be a decimal 0.0-1.0, ordered descending
- Return EXACTLY 3 predictions even if uncertain
- Return ONLY the JSON array, no other text or formatting
- If the image doesn't contain a LEGO piece, return low confidence scores anyway
- No markdown, no explanation, no "```json" blocks - ONLY the array"""

# Available models to try (in order of preference)
DEFAULT_MODELS = ["llava:13b", "llava:7b", "moondream", "llava"]


@app.on_event("startup")
async def startup_event():
    """Log startup information"""
    logger.info("BrickScan Vision Server starting...")
    logger.info(f"Ollama URL: {OLLAMA_URL}")

    # Try to connect to Ollama
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{OLLAMA_URL}/api/tags")
            models = [m["name"] for m in resp.json().get("models", [])]
            logger.info(f"Found {len(models)} Ollama models: {', '.join(models[:5])}")
    except Exception as e:
        logger.warning(f"Could not reach Ollama at startup: {e}")


@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Check server and Ollama health status.

    Returns available models and connection status.
    Used to verify vision server is running before making requests.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{OLLAMA_URL}/api/tags", timeout=5.0)
            data = resp.json()
            models = [m["name"] for m in data.get("models", [])]

        return HealthResponse(
            status="healthy",
            available_models=models,
            ollama_url=OLLAMA_URL,
            timestamp=datetime.utcnow().isoformat()
        )
    except httpx.ConnectError as e:
        logger.error(f"Cannot connect to Ollama: {e}")
        return HealthResponse(
            status="unhealthy",
            error=f"Cannot reach Ollama at {OLLAMA_URL}",
            ollama_url=OLLAMA_URL,
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="error",
            error=str(e),
            ollama_url=OLLAMA_URL,
            timestamp=datetime.utcnow().isoformat()
        )


@app.post("/identify", response_model=IdentifyResponse)
async def identify_piece(request: IdentifyRequest):
    """
    Identify a LEGO piece from a base64-encoded image.

    Uses local vision models (LLaVA) running via Ollama.
    Much faster than cloud APIs and respects privacy.

    Args:
        image_base64: Base64-encoded image data
        top_k: Number of predictions to return (1-5)
        model: Optional specific model to use

    Returns:
        IdentifyResponse with predictions and confidence scores
    """
    start_time = time.time()

    # Validate image
    try:
        image_bytes = base64.b64decode(request.image_base64)
        if len(image_bytes) < 100:
            raise HTTPException(status_code=400, detail="Invalid image data (too small)")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")

    # Determine which models to try
    if request.model:
        models_to_try = [request.model]
    else:
        models_to_try = DEFAULT_MODELS

    logger.info(f"Identifying piece, trying models: {models_to_try}")

    # Try each model until one succeeds
    last_error = None
    for model_name in models_to_try:
        try:
            result = await call_ollama_vision(
                request.image_base64,
                model_name,
                max_time=MAX_INFERENCE_TIME
            )

            if result:
                elapsed_ms = (time.time() - start_time) * 1000
                logger.info(
                    f"Success with {model_name}: "
                    f"confidence=[{result[0].confidence:.2f}, {result[1].confidence:.2f}, {result[2].confidence:.2f}] "
                    f"time={elapsed_ms:.0f}ms"
                )

                return IdentifyResponse(
                    predictions=result[:request.top_k],
                    model_used=model_name,
                    processing_time_ms=elapsed_ms,
                    timestamp=datetime.utcnow().isoformat()
                )
        except asyncio.TimeoutError:
            last_error = f"Model {model_name}: Timeout (>{MAX_INFERENCE_TIME}s)"
            logger.warning(last_error)
            continue
        except Exception as e:
            last_error = f"Model {model_name}: {str(e)[:100]}"
            logger.warning(last_error)
            continue

    # All models failed
    error_msg = f"All vision models failed. Last error: {last_error}"
    logger.error(error_msg)
    raise HTTPException(status_code=503, detail=error_msg)


@app.get("/models")
async def list_models():
    """
    List available Ollama models.

    Returns all models currently loaded in Ollama.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{OLLAMA_URL}/api/tags")
            resp.raise_for_status()

        models_raw = resp.json().get("models", [])
        models = [
            {
                "name": m["name"],
                "size_gb": round(m["size"] / 1e9, 2),
                "modified": m["modified_at"]
            }
            for m in models_raw
        ]

        logger.info(f"Listed {len(models)} models")
        return {"models": models, "count": len(models)}

    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=503, detail=f"Cannot reach Ollama: {e}")


@app.post("/pull-model")
async def pull_model(model_name: str, background_tasks: BackgroundTasks):
    """
    Pull (download) a model from Ollama registry.

    This happens asynchronously in the background.
    Monitor progress via /health or /models endpoints.

    Args:
        model_name: Name of model to pull (e.g., "llava:13b")
    """
    logger.info(f"User requested model pull: {model_name}")

    # Validate model name
    if not re.match(r'^[a-z0-9\-:]+$', model_name):
        raise HTTPException(status_code=400, detail="Invalid model name")

    # Add to background tasks
    background_tasks.add_task(pull_model_background, model_name)

    return {"status": "pulling", "model": model_name, "message": "Model pull started in background"}


async def pull_model_background(model_name: str):
    """Download a model from Ollama registry (background task)"""
    try:
        logger.info(f"Pulling model: {model_name}")
        async with httpx.AsyncClient(timeout=600.0) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_URL}/api/pull",
                json={"name": model_name}
            ) as resp:
                async for line in resp.aiter_lines():
                    if line:
                        data = json.loads(line)
                        if data.get("status"):
                            logger.info(f"  {model_name}: {data['status']}")

        logger.info(f"Successfully pulled {model_name}")
    except Exception as e:
        logger.error(f"Failed to pull {model_name}: {e}")


async def call_ollama_vision(
    image_base64: str,
    model: str,
    max_time: float = 180.0
) -> list[PartPrediction]:
    """
    Call Ollama vision API and parse LEGO identification response.

    Args:
        image_base64: Base64-encoded image
        model: Ollama model name to use
        max_time: Maximum time to wait (seconds)

    Returns:
        List of PartPrediction objects (up to 3)

    Raises:
        ValueError: If response cannot be parsed
        asyncio.TimeoutError: If inference takes too long
    """
    payload = {
        "model": model,
        "prompt": LEGO_IDENTIFICATION_PROMPT,
        "images": [image_base64],
        "stream": False,
        "options": {
            "temperature": 0.2,  # Low temp for consistent identification
            "top_p": 0.9,        # Focus on high-probability tokens
            "num_predict": 500,  # Max output tokens
        }
    }

    start_time = time.time()

    try:
        async with httpx.AsyncClient(timeout=max_time) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json=payload,
                timeout=max_time
            )
            resp.raise_for_status()

        # Check timeout
        if (time.time() - start_time) > max_time:
            raise asyncio.TimeoutError(f"Inference exceeded {max_time}s limit")

        response_text = resp.json().get("response", "")
        logger.debug(f"Model {model} response length: {len(response_text)} chars")

    except httpx.TimeoutException:
        raise asyncio.TimeoutError(f"Ollama request timeout after {max_time}s")
    except httpx.RequestError as e:
        raise ValueError(f"Failed to call Ollama: {e}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid response from Ollama: {e}")

    # Extract JSON from response
    # Model might add extra text, so we search for JSON array
    json_match = re.search(r'\[[\s\S]*\]', response_text)

    if not json_match:
        # Detailed error for debugging
        logger.warning(f"No JSON found in response. Response: {response_text[:500]}")
        raise ValueError(f"Model {model} did not return JSON array. Response: {response_text[:200]}")

    try:
        predictions_raw = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON: {e}")
        logger.warning(f"JSON string: {json_match.group()[:500]}")
        raise ValueError(f"Could not parse model response as JSON: {e}")

    # Validate and convert to PartPrediction objects
    if not isinstance(predictions_raw, list):
        raise ValueError(f"Expected JSON array, got {type(predictions_raw)}")

    if len(predictions_raw) == 0:
        raise ValueError("Model returned empty predictions array")

    predictions = []
    for i, p in enumerate(predictions_raw[:3]):  # Take top 3
        try:
            predictions.append(
                PartPrediction(
                    part_num=str(p.get("part_num", "")).strip(),
                    part_name=str(p.get("part_name", "")).strip(),
                    color_name=str(p.get("color_name", "")).strip(),
                    confidence=float(p.get("confidence", 0.0))
                )
            )
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not parse prediction {i}: {p}, error: {e}")
            continue

    if len(predictions) == 0:
        raise ValueError("No valid predictions could be parsed")

    # Pad to 3 predictions if needed
    while len(predictions) < 3:
        predictions.append(
            PartPrediction(
                part_num="0000",
                part_name="Unknown",
                color_name="Unknown",
                confidence=0.0
            )
        )

    return predictions


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    logger.error(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Catch-all exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# Root endpoint
@app.get("/")
async def root():
    """API root - returns documentation and health status"""
    return {
        "service": "BrickScan Vision Server",
        "version": "1.0.0",
        "endpoints": {
            "POST /identify": "Identify LEGO piece from image",
            "GET /health": "Server and Ollama health status",
            "GET /models": "List available Ollama models",
            "POST /pull-model": "Download a model from registry",
            "GET /docs": "Interactive API documentation (Swagger UI)"
        },
        "docs": "http://localhost:8001/docs",
        "ollama": OLLAMA_URL
    }


if __name__ == "__main__":
    import uvicorn
    import asyncio

    # Import here to avoid issues

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
