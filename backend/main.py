import asyncio
from contextlib import asynccontextmanager
import logging
import os
import time
import threading
from collections import defaultdict
from typing import List

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import PlainTextResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
import uvicorn
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry, multiprocess

from phonetic_engine import engine
from admin import router as admin_router

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IndicSync")

# --- Rate Limiting Infrastructure ---
class SlidingWindowRateLimiter:
    """Lightweight in-memory IP-based sliding window rate limiter."""
    def __init__(self, limit: int, window: int):
        self.limit = limit
        self.window = window
        self.requests = defaultdict(list)
        self.lock = threading.Lock()

    def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        with self.lock:
            # Keep only requests within the current sliding window
            times = [t for t in self.requests[client_ip] if now - t < self.window]
            if times:
                self.requests[client_ip] = times
            else:
                self.requests.pop(client_ip, None)
                times = []
                
            # Periodically prune old/inactive client IPs to avoid memory leaks
            if len(self.requests) > 1000:
                expired_ips = [ip for ip, req_times in list(self.requests.items()) if not req_times or now - req_times[-1] >= self.window]
                for ip in expired_ips:
                    self.requests.pop(ip, None)
                    
            if len(times) < self.limit:
                self.requests[client_ip].append(now)
                return True
            return False

# Limit to 100 requests per 60 seconds per IP
rate_limiter = SlidingWindowRateLimiter(limit=100, window=60)

async def rate_limit_dependency(request: Request):
    ip = request.client.host if request.client else "127.0.0.1"
    if not rate_limiter.is_allowed(ip):
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please slow down and try again later."
        )

# --- Lifespan Context Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logger.info("Initializing Indic Phonetic Similarity Service...")
    # Setup multiprocess environment for Prometheus if specified
    multiproc_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    if multiproc_dir:
        os.makedirs(multiproc_dir, exist_ok=True)
    yield
    # Shutdown logic
    logger.info("Shutting down Indic Phonetic Similarity Service...")
    if multiproc_dir and os.path.exists(multiproc_dir):
        import shutil
        try:
            shutil.rmtree(multiproc_dir)
        except Exception as e:
            logger.error(f"Failed to clear multiprocess metrics folder: {e}")

app = FastAPI(
    title="Indic Phonetic Similarity API",
    description="Production-grade phonetic similarity engine for Indic names and entities.",
    lifespan=lifespan
)

# Include Admin Router
app.include_router(admin_router)

# --- Prometheus Metrics Registration ---
# We use a dedicated registry to avoid duplicate metrics registrations on reload
metrics_registry = CollectorRegistry()

# Initialize multiprocess collector if configured
multiproc_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
if multiproc_dir:
    multiprocess.MultiProcessCollector(metrics_registry)

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total number of HTTP requests processed.",
    ["method", "endpoint", "status"],
    registry=metrics_registry
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "Total duration of HTTP requests in seconds.",
    ["method", "endpoint"],
    registry=metrics_registry
)

# --- Middleware ---
@app.middleware("http")
async def record_metrics(request: Request, call_next):
    # Skip endpoints we don't want to track in standard application metrics
    path = request.url.path
    if path in ["/metrics", "/health"] or path.startswith("/static") or "." in path.split("/")[-1]:
        return await call_next(request)
        
    start_time = time.perf_counter()
    try:
        response = await call_next(request)
        duration = time.perf_counter() - start_time
        HTTP_REQUESTS_TOTAL.labels(
            method=request.method,
            endpoint=path,
            status=response.status_code
        ).inc()
        HTTP_REQUEST_DURATION_SECONDS.labels(
            method=request.method,
            endpoint=path
        ).observe(duration)
        return response
    except Exception as e:
        duration = time.perf_counter() - start_time
        HTTP_REQUESTS_TOTAL.labels(
            method=request.method,
            endpoint=path,
            status=500
        ).inc()
        HTTP_REQUEST_DURATION_SECONDS.labels(
            method=request.method,
            endpoint=path
        ).observe(duration)
        raise e

# CORS Configuration
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "")
if allowed_origins_env:
    origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
    logger.info(f"CORS: Allowed origins configured: {origins}")
else:
    origins = []
    logger.warning("CORS: ALLOWED_ORIGINS not set. CORS requests will be blocked.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True if origins else False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Models ---
class ComparisonRequest(BaseModel):
    name1: str = Field(..., min_length=1, max_length=100, description="First name or place entity")
    name2: str = Field(..., min_length=1, max_length=100, description="Second name or place entity")
    enable_aliases: bool = Field(True, description="Enable administrative/historical alias synonym matching")
    threshold: float = Field(default=None, description="Optional custom similarity threshold override")

class BatchRequest(BaseModel):
    pairs: List[ComparisonRequest] = Field(..., max_items=1000, description="List of comparison pairs")
    enable_aliases: bool = Field(True, description="Enable administrative/historical alias synonym matching globally")
    threshold: float = Field(default=None, description="Optional custom similarity threshold override globally")

def validate_input(name: str, identifier: str):
    """Performs validation checks on input names with sanitized error messages to prevent injection reflection."""
    trimmed = name.strip()
    if not trimmed:
        raise HTTPException(
            status_code=400,
            detail=f"Input for {identifier} must contain non-whitespace characters."
        )
    
    normalized = engine.normalize(trimmed)
    if not normalized:
        raise HTTPException(
            status_code=400,
            detail=f"Input for {identifier} contains no valid Latin characters. Please use English transliterations."
        )

# --- Public Endpoints ---
@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/metrics")
async def get_metrics():
    # If multiprocess collector directory is configured, regenerate latest dynamically
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        reg = CollectorRegistry()
        multiprocess.MultiProcessCollector(reg)
        data = generate_latest(reg)
    else:
        data = generate_latest(metrics_registry)
    return PlainTextResponse(data, media_type=CONTENT_TYPE_LATEST)

@app.post("/compare", dependencies=[Depends(rate_limit_dependency)])
async def compare_names(request: ComparisonRequest):
    validate_input(request.name1, "First Name / Place")
    validate_input(request.name2, "Second Name / Place")
    
    start_time = time.perf_counter()
    try:
        result = await run_in_threadpool(
            engine.compare,
            request.name1,
            request.name2,
            request.enable_aliases,
            request.threshold
        )
        duration_ms = (time.perf_counter() - start_time) * 1000
        result["processing_time_ms"] = round(duration_ms, 3)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error during comparison: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during processing")

async def process_pair(pair: ComparisonRequest, global_enable_aliases: bool, global_threshold: float, index: int):
    """Worker helper to validate and run comparisons in a threadpool concurrently."""
    try:
        # Validate input values (sanitized)
        validate_input(pair.name1, f"Pair {index+1} Name 1")
        validate_input(pair.name2, f"Pair {index+1} Name 2")
        
        # Determine threshold & alias enablement preference
        aliases_flag = pair.enable_aliases if pair.enable_aliases is not None else global_enable_aliases
        threshold_val = pair.threshold if pair.threshold is not None else global_threshold
        
        res = await run_in_threadpool(
            engine.compare,
            pair.name1,
            pair.name2,
            aliases_flag,
            threshold_val
        )
        return {"status": "success", "data": res}
    except HTTPException as e:
        return {"status": "error", "error": e.detail}
    except Exception as e:
        return {"status": "error", "error": "Internal error during batch processing"}

@app.post("/compare-batch", dependencies=[Depends(rate_limit_dependency)])
async def compare_batch(request: BatchRequest):
    if not request.pairs:
        raise HTTPException(status_code=400, detail="Batch request must contain at least one comparison pair.")

    start_time = time.perf_counter()
    
    # Offload CPU-bound batch computation tasks concurrently to avoid blocking
    tasks = [
        process_pair(pair, request.enable_aliases, request.threshold, i)
        for i, pair in enumerate(request.pairs)
    ]
    results = await asyncio.gather(*tasks)
    
    duration_ms = (time.perf_counter() - start_time) * 1000
    logger.info(f"Batch of {len(request.pairs)} comparisons completed concurrently in {duration_ms:.2f}ms")
    return {
        "results": results,
        "processing_time_ms": round(duration_ms, 3)
    }

# Mount static frontend
try:
    frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../frontend"))
    if os.path.exists(frontend_dir):
        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="static")
        logger.info(f"Mounted static frontend from {frontend_dir}")
    else:
        logger.warning(f"Frontend directory not found at {frontend_dir}. API mode only.")
except Exception as e:
    logger.error(f"Failed to mount frontend: {str(e)}")

if __name__ == "__main__":
    dev_mode = os.getenv("DEV_MODE", "false").lower() == "true"
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=dev_mode)
