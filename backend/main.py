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
from concurrent.futures import ProcessPoolExecutor
import redis.asyncio as redis

from phonetic_engine import engine
from admin import router as admin_router

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IndicSync")

redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)

# --- Prometheus Metrics Initialization ---
metrics_registry = CollectorRegistry()

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

REDIS_CONNECTION_ERRORS_TOTAL = Counter(
    "redis_connection_errors_total",
    "Total number of Redis connection failures.",
    registry=metrics_registry
)

RATE_LIMITER_BYPASSED_TOTAL = Counter(
    "rate_limiter_bypassed_total",
    "Total number of rate limiter checks that failed open/fallback.",
    registry=metrics_registry
)

# --- Rate Limiting Infrastructure ---
class DummyRequests:
    def clear(self):
        try:
            import redis
            r = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
            r.flushdb()
        except:
            pass

class AsyncRedisCircuitBreaker:
    def __init__(self, failure_threshold=3, recovery_time=60):
        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time
        self.failure_count = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF-OPEN
        self.last_state_change = 0

    def record_success(self):
        self.failure_count = 0
        self.state = "CLOSED"

    def record_failure(self):
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            self.last_state_change = time.time()
            logger.error("Redis (Async) Circuit Breaker tripped to OPEN.")

    def is_allowed(self):
        if self.state == "OPEN":
            if time.time() - self.last_state_change > self.recovery_time:
                self.state = "HALF-OPEN"
                return True
            return False
        return True

class TokenBucketLimiter:
    def __init__(self, limit: int, window: int):
        self.limit = limit
        self.window = window
        self.buckets = {}  # ip -> (tokens, last_update)
        self.lock = threading.Lock()

    def is_allowed(self, ip: str) -> bool:
        with self.lock:
            now = time.time()
            if ip not in self.buckets:
                self.buckets[ip] = (self.limit - 1, now)
                return True
            tokens, last_update = self.buckets[ip]
            elapsed = now - last_update
            new_tokens = tokens + elapsed * (self.limit / self.window)
            if new_tokens > self.limit:
                new_tokens = self.limit
            if new_tokens >= 1:
                self.buckets[ip] = (new_tokens - 1, now)
                return True
            self.buckets[ip] = (new_tokens, now)
            return False

class RedisRateLimiter:
    """Lightweight Redis-based sliding window rate limiter with local fallback & circuit breaker."""
    def __init__(self, limit: int, window: int):
        self.limit = limit
        self.window = window
        self.requests = DummyRequests()
        self.breaker = AsyncRedisCircuitBreaker()
        self.fallback_limiter = TokenBucketLimiter(limit, window)

    async def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        key = f"rate_limit:{client_ip}"
        
        if not self.breaker.is_allowed():
            RATE_LIMITER_BYPASSED_TOTAL.inc()
            return self.fallback_limiter.is_allowed(client_ip)
            
        try:
            async with redis_client.pipeline(transaction=True) as pipe:
                pipe.zremrangebyscore(key, 0, now - self.window)
                pipe.zcard(key)
                pipe.zadd(key, {str(now): now})
                pipe.expire(key, self.window)
                results = await pipe.execute()
                
            current_requests = results[1]
            self.breaker.record_success()
            return current_requests < self.limit
        except Exception as e:
            logger.warning(f"Rate limiter redis error: {e}")
            self.breaker.record_failure()
            REDIS_CONNECTION_ERRORS_TOTAL.inc()
            RATE_LIMITER_BYPASSED_TOTAL.inc()
            return self.fallback_limiter.is_allowed(client_ip)

# Limit to 100 requests per 60 seconds per IP
rate_limiter = RedisRateLimiter(limit=100, window=60)

async def rate_limit_dependency(request: Request):
    ip = request.client.host if request.client else "127.0.0.1"
    if not await rate_limiter.is_allowed(ip):
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
        from prometheus_client import multiprocess
        multiprocess.MultiProcessCollector(metrics_registry)
        
    # Start ProcessPoolExecutor with cgroups-aware limits
    max_workers_env = os.environ.get("MAX_WORKERS")
    if max_workers_env:
        max_workers = int(max_workers_env)
    else:
        try:
            with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
                quota = int(f.read().strip())
            with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
                period = int(f.read().strip())
            if quota > 0 and period > 0:
                max_workers = max(1, int(quota / period))
            else:
                max_workers = os.cpu_count()
        except Exception:
            max_workers = os.cpu_count()
            
    logger.info(f"Starting ProcessPoolExecutor with {max_workers} workers.")
    app.state.executor = ProcessPoolExecutor(max_workers=max_workers)
    
    # Start background Pub/Sub listener for config updates
    async def listen_config_updates():
        pubsub = redis_client.pubsub()
        await pubsub.subscribe("config_updates")
        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    logger.info("Redis Pub/Sub signal received. Reloading configuration...")
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, engine.sync_config, True)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in config update Pub/Sub listener: {e}")
        finally:
            await pubsub.unsubscribe("config_updates")

    config_listener_task = asyncio.create_task(listen_config_updates())
    
    yield
    
    # Shutdown logic
    config_listener_task.cancel()
    try:
        await config_listener_task
    except asyncio.CancelledError:
        pass
        
    app.state.executor.shutdown(wait=True)
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

# --- Prometheus Metrics Registration (Placeholder, defined above) ---

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
async def compare_names(request_data: ComparisonRequest, request: Request):
    validate_input(request_data.name1, "First Name / Place")
    validate_input(request_data.name2, "Second Name / Place")
    
    start_time = time.perf_counter()
    try:
        loop = asyncio.get_running_loop()
        executor = getattr(request.app.state, "executor", None)
        if executor:
            result = await loop.run_in_executor(
                executor,
                engine.compare,
                request_data.name1,
                request_data.name2,
                request_data.enable_aliases,
                request_data.threshold
            )
        else:
            result = await run_in_threadpool(
                engine.compare,
                request_data.name1,
                request_data.name2,
                request_data.enable_aliases,
                request_data.threshold
            )
        duration_ms = (time.perf_counter() - start_time) * 1000
        result["processing_time_ms"] = round(duration_ms, 3)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error during comparison: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during processing")

async def process_pair(pair: ComparisonRequest, global_enable_aliases: bool, global_threshold: float, index: int, executor):
    """Worker helper to validate and run comparisons in a threadpool concurrently."""
    try:
        # Validate input values (sanitized)
        validate_input(pair.name1, f"Pair {index+1} Name 1")
        validate_input(pair.name2, f"Pair {index+1} Name 2")
        
        # Determine threshold & alias enablement preference
        aliases_flag = pair.enable_aliases if pair.enable_aliases is not None else global_enable_aliases
        threshold_val = pair.threshold if pair.threshold is not None else global_threshold
        
        loop = asyncio.get_running_loop()
        if executor:
            res = await loop.run_in_executor(
                executor,
                engine.compare,
                pair.name1,
                pair.name2,
                aliases_flag,
                threshold_val
            )
        else:
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

from fastapi import Response

@app.post("/compare-batch", dependencies=[Depends(rate_limit_dependency)])
async def compare_batch(batch_request: BatchRequest, request: Request, response: Response):
    if not batch_request.pairs:
        raise HTTPException(status_code=400, detail="Batch request must contain at least one comparison pair.")

    # Enforce payload limit
    if len(batch_request.pairs) > 1000:
        raise HTTPException(status_code=400, detail="Batch request size cannot exceed 1000 pairs.")

    start_time = time.perf_counter()
    
    # Process tasks in chunks of 50 to yield control to the event loop
    results = []
    chunk_size = 50
    executor = getattr(request.app.state, "executor", None)
    
    for i in range(0, len(batch_request.pairs), chunk_size):
        chunk = batch_request.pairs[i : i + chunk_size]
        chunk_tasks = [
            process_pair(pair, batch_request.enable_aliases, batch_request.threshold, i + j, executor)
            for j, pair in enumerate(chunk)
        ]
        chunk_results = await asyncio.gather(*chunk_tasks)
        results.extend(chunk_results)
    
    # Check for mixed errors and return 207 Multi-Status
    has_success = any(r["status"] == "success" for r in results)
    has_error = any(r["status"] == "error" for r in results)
    
    if has_error:
        if not has_success:
            response.status_code = 400
        else:
            response.status_code = 207  # Mixed success/failure
            
    duration_ms = (time.perf_counter() - start_time) * 1000
    logger.info(f"Batch of {len(batch_request.pairs)} comparisons completed in {duration_ms:.2f}ms")
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
