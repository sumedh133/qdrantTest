from fastapi import APIRouter, HTTPException, Body, Request
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel
from app.qdrant import add_property_to_qdrant
from app.qdrant.propertyVectorize import PropertyVectorizer
from app import properties, requirements


router = APIRouter()


class PropertyPoint(BaseModel):
    id: Union[int, str]
    payload: Dict[str, Any]
    vector: Optional[List[float]] = None
    collection: Optional[str] = "properties"


@router.get("/update-all-properties-from-json")
def update_all_properties_from_json():
    import hashlib

    properties_data = properties[-50:]
    print("Processing", len(properties_data), "properties from JSON")
    
    # ✅ 1. Vectorize ALL properties TOGETHER (this is the crucial fix)
    vectorizer = PropertyVectorizer()
    vectors = vectorizer.fit_transform(properties_data)

    success_count = 0
    error_count = 0

    # ✅ 2. Loop and insert using the precomputed vectors
    for i, prop in enumerate(properties_data):
        try:
            vector_list = vectors[i].tolist()

            # ✅ Convert string ID to integer hash
            prop_id = prop.get("id")
            if isinstance(prop_id, str):
                hash_obj = hashlib.md5(prop_id.encode())
                prop_id = int(hash_obj.hexdigest()[:8], 16)
            elif not isinstance(prop_id, int):
                hash_obj = hashlib.md5(str(prop).encode())
                prop_id = int(hash_obj.hexdigest()[:8], 16)

            add_property_to_qdrant(
                id=prop_id,
                vector=vector_list,
                payload=prop,
                collection="properties_index3",
            )
            success_count += 1

        except Exception as e:
            print(f"Error processing property {prop.get('id')}: {str(e)}")
            error_count += 1
            continue

    return {
        "message": "Property update completed",
        "success": success_count,
        "errors": error_count,
        "total": len(properties_data)
    }


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
        raw = await request.body()
        
        try:
            import json as _json
            body = _json.loads(raw.decode("utf-8"))
        except Exception as e:
            print("JSON parse error:", str(e))
            raise HTTPException(status_code=400, detail="Invalid JSON body") from e
        
        if not isinstance(body, dict):
            print(f"Body is not a dict, it's: {type(body)}")
            raise HTTPException(status_code=400, detail="Invalid JSON body (must be an object)")

        prop_id: Union[int, str] = body.get("id")
        prop_payload: Dict[str, Any] = body.get("property")
        collection: str = body.get("collection")

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
        raise HTTPException(status_code=500, detail={
            "error": "Failed to vectorize/upsert property",
            "message": str(e),
        })

@router.get("/properties/search")
def search_properties():
    print("found")
    return {"message": "Search endpoint not yet implemented"}