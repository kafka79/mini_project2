from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from typing import List
from phonetic_engine import engine
import os
import secrets
import logging
import json
import redis.asyncio as redis

logger = logging.getLogger("IndicSync")
redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)
router = APIRouter(prefix="/admin", tags=["admin"])

# Secure Admin Key Generation Pattern
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")
if not ADMIN_API_KEY or ADMIN_API_KEY == "admin-secret-key-change-me":
    # DO NOT auto-generate a random key for distributed deployments.
    # Default to a highly secure/disabled state and warn the user.
    ADMIN_API_KEY = "DISABLED_SECURE_DEFAULT"
    logger.critical("ADMIN_API_KEY is missing or set to default! Admin API endpoints are disabled. Please set the ADMIN_API_KEY environment variable for production.")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_admin_key(api_key: str = Security(api_key_header)):
    if not api_key or not ADMIN_API_KEY or ADMIN_API_KEY == "DISABLED_SECURE_DEFAULT":
        raise HTTPException(status_code=403, detail="Admin API is disabled. Configure ADMIN_API_KEY.")
    if not secrets.compare_digest(api_key, ADMIN_API_KEY):
        raise HTTPException(status_code=403, detail="Invalid or missing X-API-Key header.")
    return api_key

async def safe_publish_update():
    try:
        await redis_client.publish("config_updates", "reload")
    except Exception as e:
        logger.warning(f"Failed to publish config update to Redis: {e}")

@router.post("/reload-aliases")
async def reload_aliases(api_key: str = Depends(verify_admin_key)):
    """Triggers hot-reloading of the aliases JSON configuration."""
    success = engine.reload_aliases()
    if not success:
        raise HTTPException(status_code=500, detail="Failed to reload aliases config.")
    await safe_publish_update()
    return {"status": "success", "message": "Aliases configuration reloaded successfully."}

@router.post("/aliases")
async def add_alias_group(group: List[str], api_key: str = Depends(verify_admin_key)):
    """Dynamically adds/merges a group of synonymous names to the active alias index."""
    if len(group) < 2:
        raise HTTPException(status_code=400, detail="An alias group must contain at least 2 synonyms.")
    
    # Normalize each synonym
    normalized_group = {engine.normalize(x).lower() for x in group if engine.normalize(x).strip()}
    if len(normalized_group) < 2:
        raise HTTPException(status_code=400, detail="Must have at least 2 unique valid normalized names.")
        
    # Merge into the engine's current alias mapping
    full_group = set(normalized_group)
    
    with engine._alias_lock:
        for term in normalized_group:
            if term in engine.aliases:
                full_group.update(engine.aliases[term])
                
        # Re-map every member of this merged group to group - {member}
        for member in full_group:
            engine.aliases[member] = list(full_group - {member})
            
        await redis_client.set("aliases", json.dumps(engine.aliases))
        await safe_publish_update()
        
    return {"status": "success", "message": f"Added/merged alias group: {list(full_group)}"}

class WeightsUpdate(BaseModel):
    default_threshold: float = Field(None, ge=0.0, le=100.0)
    fuzzy_weight: float = Field(None, ge=0.0, le=1.0)
    boost_short_word: float = Field(None, ge=0.0)
    boost_long_word: float = Field(None, ge=0.0)
    min_short_word: float = Field(None, ge=0.0, le=100.0)
    min_long_word: float = Field(None, ge=0.0, le=100.0)

@router.get("/weights")
async def get_weights(api_key: str = Depends(verify_admin_key)):
    """Retrieves current tuning weights and scoring thresholds."""
    return {
        "DEFAULT_THRESHOLD": engine.DEFAULT_THRESHOLD,
        "MAX_CODE_LEN": engine.MAX_CODE_LEN,
        "FUZZY_WEIGHT": engine.FUZZY_WEIGHT,
        "BOOST_SHORT_WORD": engine.BOOST_SHORT_WORD,
        "BOOST_LONG_WORD": engine.BOOST_LONG_WORD,
        "MIN_SHORT_WORD": engine.MIN_SHORT_WORD,
        "MIN_LONG_WORD": engine.MIN_LONG_WORD
    }

@router.post("/weights")
async def update_weights(payload: WeightsUpdate, api_key: str = Depends(verify_admin_key)):
    """Dynamically calibrates scoring weights and thresholds."""
    updates = {}
    if payload.default_threshold is not None:
        updates["DEFAULT_THRESHOLD"] = payload.default_threshold
    if payload.fuzzy_weight is not None:
        updates["FUZZY_WEIGHT"] = payload.fuzzy_weight
    if payload.boost_short_word is not None:
        updates["BOOST_SHORT_WORD"] = payload.boost_short_word
    if payload.boost_long_word is not None:
        updates["BOOST_LONG_WORD"] = payload.boost_long_word
    if payload.min_short_word is not None:
        updates["MIN_SHORT_WORD"] = payload.min_short_word
    if payload.min_long_word is not None:
        updates["MIN_LONG_WORD"] = payload.min_long_word
        
    engine.update_weights(updates)
    await redis_client.set("weights", json.dumps(updates))
    await safe_publish_update()
    return {"status": "success", "message": "Weights updated successfully.", "current_weights": updates}
