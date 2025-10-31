from typing import Dict, Any, List, Union, Optional
from qdrant_client.models import PointStruct, VectorParams, Distance
from app.qdrant import qdrant_client


def add_property_to_qdrant(
    *,
    id: Union[int, str],
    vector: List[float],
    payload: Dict[str, Any],
    collection: Optional[str] = "properties_index",
) -> None:
    # Ensure collection exists with correct vector size
    try:
        qdrant_client.get_collection(collection)
    except Exception:
        qdrant_client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=len(vector), distance=Distance.COSINE),
        )

    qdrant_client.upsert(
        collection_name=collection,
        points=[
            PointStruct(
                id=id,
                vector=vector,
                payload=payload,
            )
        ],
    )
