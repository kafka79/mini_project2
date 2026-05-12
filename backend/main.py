from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
import uvicorn
from phonetic_engine import engine

app = FastAPI(title="Indic Phonetic Similarity API")

# Enable CORS for frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ComparisonRequest(BaseModel):
    name1: str
    name2: str

class BatchRequest(BaseModel):
    pairs: List[ComparisonRequest]

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/compare")
def compare_names(request: ComparisonRequest):
    try:
        result = engine.compare(request.name1, request.name2)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compare-batch")
def compare_batch(request: BatchRequest):
    results = []
    for pair in request.pairs:
        results.append(engine.compare(pair.name1, pair.name2))
    return results

# Mount frontend
# Note: In production, you'd serve the frontend separately or build it
# For this mini-project, we'll assume index.html is in the ../frontend directory
try:
    app.mount("/", StaticFiles(directory="../frontend", html=True), name="static")
except:
    print("Frontend directory not found, serving API only.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
