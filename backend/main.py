import asyncio
import os
import logging
import time
import sqlite3
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List
import uvicorn
from phonetic_engine import engine

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IndicSync")

app = FastAPI(
    title="Indic Phonetic Similarity API",
    description="Production-grade phonetic similarity engine for Indic names and entities."
)

# SQLite metrics tracker to support multi-process (multi-worker) aggregation
# ponytail: SQLite database for zero-dependency shared metrics storage
METRICS_DB = os.path.join(os.path.dirname(__file__), "metrics.db")

def init_metrics_db():
    try:
        with sqlite3.connect(METRICS_DB, timeout=5.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("CREATE TABLE IF NOT EXISTS metrics (key TEXT PRIMARY KEY, value REAL)")
            conn.execute("INSERT OR IGNORE INTO metrics VALUES ('http_requests_total', 0.0)")
            conn.execute("INSERT OR IGNORE INTO metrics VALUES ('http_request_duration_seconds_sum', 0.0)")
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to initialize metrics database: {e}")

init_metrics_db()

def increment_metrics(requests: int, duration: float):
    try:
        with sqlite3.connect(METRICS_DB, timeout=2.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("UPDATE metrics SET value = value + ? WHERE key = 'http_requests_total'", (requests,))
            conn.execute("UPDATE metrics SET value = value + ? WHERE key = 'http_request_duration_seconds_sum'", (duration,))
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to increment metrics: {e}")

def get_aggregated_metrics():
    try:
        with sqlite3.connect(METRICS_DB, timeout=2.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            cursor = conn.cursor()
            cursor.execute("SELECT key, value FROM metrics")
            return dict(cursor.fetchall())
    except Exception as e:
        logger.error(f"Failed to read metrics: {e}")
        return {"http_requests_total": 0.0, "http_request_duration_seconds_sum": 0.0}

import threading

# In-memory metric buffers to avoid SQLite write bottlenecks
# ponytail: threading.Lock guarantees thread-safe increments on multi-threaded ASGI servers
local_requests = 0
local_duration = 0.0
metrics_lock = threading.Lock()

async def flush_metrics_loop():
    global local_requests, local_duration
    while True:
        try:
            await asyncio.sleep(5.0)
            with metrics_lock:
                reqs, dur = local_requests, local_duration
                local_requests = 0
                local_duration = 0.0
            if reqs > 0:
                increment_metrics(reqs, dur)
        except asyncio.CancelledError:
            break
        except Exception:
            pass

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(flush_metrics_loop())

@app.on_event("shutdown")
def shutdown_event():
    global local_requests, local_duration
    with metrics_lock:
        reqs, dur = local_requests, local_duration
        local_requests = 0
        local_duration = 0.0
    if reqs > 0:
        increment_metrics(reqs, dur)

@app.middleware("http")
async def record_metrics(request, call_next):
    global local_requests, local_duration
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    with metrics_lock:
        local_requests += 1
        local_duration += duration
    return response

@app.get("/metrics")
async def get_metrics():
    from fastapi.responses import PlainTextResponse
    m = get_aggregated_metrics()
    with metrics_lock:
        reqs = local_requests
        dur = local_duration
    total_reqs = int(m.get("http_requests_total", 0.0)) + reqs
    total_dur = m.get("http_request_duration_seconds_sum", 0.0) + dur
    return PlainTextResponse(
        f'# HELP http_requests_total Total number of HTTP requests processed.\n'
        f'# TYPE http_requests_total counter\n'
        f'http_requests_total {total_reqs}\n'
        f'# HELP http_request_duration_seconds_sum Total duration of HTTP requests in seconds.\n'
        f'# TYPE http_request_duration_seconds_sum counter\n'
        f'http_request_duration_seconds_sum {round(total_dur, 6)}\n'
    )

# CORS configuration
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "")
if allowed_origins_env:
    origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
    logger.info(f"CORS: Allowed origins configured: {origins}")
else:
    # ponytail: secure-by-default fallback to empty list instead of wildcard *
    origins = []
    logger.warning("CORS: ALLOWED_ORIGINS not set. CORS requests will be blocked.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True if origins else False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ComparisonRequest(BaseModel):
    name1: str = Field(..., min_length=1, max_length=100, description="First name or place entity")
    name2: str = Field(..., min_length=1, max_length=100, description="Second name or place entity")
    enable_aliases: bool = Field(True, description="Enable administrative/historical alias synonym matching")

class BatchRequest(BaseModel):
    pairs: List[ComparisonRequest] = Field(..., max_items=1000, description="List of comparison pairs")

def validate_input(name: str, identifier: str):
    """Performs validation checks on input names."""
    trimmed = name.strip()
    if not trimmed:
        raise HTTPException(status_code=400, detail=f"{identifier} cannot be empty or contain only whitespaces.")
    
    # Check if the string contains only non-Latin characters (which normalize to empty string)
    normalized = engine.normalize(trimmed)
    if not normalized:
        raise HTTPException(
            status_code=400,
            detail=f"Input '{name}' for {identifier} contains no valid Latin characters. Please use English transliterations."
        )

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/compare")
async def compare_names(request: ComparisonRequest):
    # 1. Input Validation
    validate_input(request.name1, "First Name / Place")
    validate_input(request.name2, "Second Name / Place")
    
    start_time = time.perf_counter()
    try:
        # 2. Concurrency: run directly in the event loop since compare is sub-millisecond CPU execution
        result = engine.compare(
            request.name1,
            request.name2,
            request.enable_aliases
        )
        duration_ms = (time.perf_counter() - start_time) * 1000
        result["processing_time_ms"] = round(duration_ms, 3)
        logger.info(f"Comparison of '{request.name1}' and '{request.name2}' completed in {duration_ms:.2f}ms")
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error comparing '{request.name1}' and '{request.name2}': {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during processing")

@app.post("/compare-batch")
async def compare_batch(request: BatchRequest):
    if not request.pairs:
        raise HTTPException(status_code=400, detail="Batch request must contain at least one comparison pair.")

    start_time = time.perf_counter()
    
    # ponytail: process sequentially since each compare is sub-millisecond, avoiding thread pool scheduling overhead
    results = []
    for i, pair in enumerate(request.pairs):
        try:
            validate_input(pair.name1, f"Pair {i+1} Name 1")
            validate_input(pair.name2, f"Pair {i+1} Name 2")
            res = engine.compare(pair.name1, pair.name2, pair.enable_aliases)
            results.append({"status": "success", "data": res})
        except HTTPException as e:
            results.append({"status": "error", "error": e.detail})
        except Exception as e:
            results.append({"status": "error", "error": f"Internal error: {str(e)}"})
            
    duration_ms = (time.perf_counter() - start_time) * 1000
    logger.info(f"Batch of {len(request.pairs)} comparisons completed in {duration_ms:.2f}ms")
    return {
        "results": results,
        "processing_time_ms": round(duration_ms, 3)
    }

# Mount frontend static directory
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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
