from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Cloud-only configuration
api_key = os.getenv("QDRANT_API_KEY")
url = os.getenv("QDRANT_URL") or os.getenv("QDRANT_ENDPOINT")

if not url:
    raise ValueError("QDRANT_ENDPOINT or QDRANT_URL must be set for Qdrant Cloud.")

# Initialize Qdrant client for Cloud (expects full https URL)
client = QdrantClient(url=url, api_key=api_key)

print("✅ Qdrant client initialized successfully!")

point = client.retrieve(
    collection_name="properties_index2",
    ids=[105532699],
    with_vectors=True,
    with_payload=True
)

print(point)
