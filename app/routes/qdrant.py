from fastapi import APIRouter, HTTPException, Body, Request
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel
from app.qdrant import add_property_to_qdrant
from app.qdrant.propertyVectorize import PropertyVectorizer


router = APIRouter()


class PropertyPoint(BaseModel):
    id: Union[int, str]
    payload: Dict[str, Any]
    vector: Optional[List[float]] = None
    collection: Optional[str] = "properties"


@router.post("/properties")
def add_property(point: PropertyPoint):
    if point.vector is None:
        raise HTTPException(status_code=400, detail="vector is required to upsert into Qdrant")

    try:
        add_property_to_qdrant(
            id=point.id,
            vector=point.vector,
            payload=point.payload,
            collection=point.collection,
        )
        return {"message": "Property upserted successfully", "collection": point.collection, "id": point.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upsert property: {e}")


@router.post("/properties/vectorize")
async def vectorize_and_add_property(request: Request):
    try:
        try:
            print("request: ", await request.body())
            body = await request.json()
        except Exception:
            raw = await request.body()
            try:
                import json as _json
                body = _json.loads(raw.decode("utf-8"))
            except Exception as _e:
                print("/qdrant/properties/vectorize invalid body:", raw[:2000])
                raise HTTPException(status_code=400, detail="Invalid JSON body") from _e
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Invalid JSON body (must be an object)")

        prop_id: Union[int, str] = body.get("id")
        # Accept both 'property' and 'property_' keys
        prop_payload: Dict[str, Any] = body.get("property") or body.get("property_") or {}
        collection: str = body.get("collection") or "properties_index"

        if prop_id is None:
            raise HTTPException(status_code=422, detail="Field 'id' is required")
        if not isinstance(prop_payload, dict) or not prop_payload:
            raise HTTPException(status_code=422, detail="Field 'property' must be a non-empty object")

        vectorizer = PropertyVectorizer()
        vectors = vectorizer.fit_transform([prop_payload])
        vector_list = vectors[0].tolist()

        add_property_to_qdrant(
            id=prop_id,
            vector=vector_list,
            payload=prop_payload,
            collection=collection,
        )
        return {"message": "Property vectorized and upserted successfully", "collection": collection, "id": prop_id}
    except HTTPException:
        raise
    except Exception as e:
        # Surface error details for debugging
        raise HTTPException(status_code=500, detail={
            "error": "Failed to vectorize/upsert property",
            "message": str(e),
        })


@router.get("/properties/search")
def search_properties():
    print("found")