import { QdrantClient } from "@qdrant/js-client-rest";
import dotenv from "dotenv";
dotenv.config();

export const qdrant = new QdrantClient({
  url: process.env.QDRANT_ENDPOINT,
  apiKey: process.env.QDRANT_API_KEY,
});
