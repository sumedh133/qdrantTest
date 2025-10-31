from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Optionally get API key from environment variable
api_key = os.getenv("QDRANT_API_KEY")
endpoint = os.getenv("QDRANT_ENDPOINT")

# Initialize Qdrant client
qdrant_client = QdrantClient(
    host=endpoint,
    api_key=api_key, 
)
