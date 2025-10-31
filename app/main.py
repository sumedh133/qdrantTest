from fastapi import FastAPI
import json
from pathlib import Path
from fastapi.responses import JSONResponse
from app.qdrant import qdrant_client

app = FastAPI()

# Check if the Qdrant client is initialized
if qdrant_client:
    print("✅ Qdrant client is available in main.py")
# Global variable to hold data
properties = None
requirements = None

@app.on_event("startup")
def load_json_data():
    global properties
    global requirements
    json_path = Path("app/data/properties.json")  # path to your file
    with open(json_path, "r", encoding="utf-8") as f:
        properties = json.load(f)
    
    json_path = Path("app/data/requirements.json")  # path to your file
    with open(json_path, "r", encoding="utf-8") as f:
        requirements = json.load(f)
    print("✅ JSON data loaded successfully!")

@app.get("/properties")
def get_properties():
    print(properties[0])
    return JSONResponse(content=properties[0])


@app.get("/requirements")
def get_requirements():
    print(requirements[0])
    return JSONResponse(content=requirements[0])