from fastapi import FastAPI
import json
from pathlib import Path
from fastapi.responses import JSONResponse
from app.qdrant import qdrant_client
from app import properties, requirements
from app.routes.qdrant import router as qdrant_router

app = FastAPI()

app.include_router(qdrant_router, prefix="/qdrant", tags=["qdrant"])

@app.get("/properties")
def get_properties():
    if properties:
        return JSONResponse({"message": "Properties fetched successfully"})
    else:
        return JSONResponse({"message": "No properties found"}, status_code=404)


@app.get("/requirements")
def get_requirements():
    if requirements:
        return JSONResponse({"message": "Requirements fetched successfully"})
    else:
        return JSONResponse({"message": "No requirements found"}, status_code=404)