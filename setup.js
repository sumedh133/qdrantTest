import { qdrant } from "./qdrant.js";

async function setup() {

  console.log("Creating properties_index...");
  await qdrant.createCollection("properties_index", {
    vectors: {
      size: 128,
      distance: "Cosine",
    },
  });

  console.log("Creating requirements_index...");
  await qdrant.createCollection("requirements_index", {
    vectors: {
      size: 128,
      distance: "Cosine",
    },
  });

  console.log("✅ Collections created successfully!");
}

setup();
