import { ollamaEmbeddingService } from "./services/ollamaEmbedding.service";
import { vectorDBService } from "./services/vectorDB.service";

async function testRAGPipeline() {
  console.log("🚀 Starting RAG Integration Test...");

  try {
    // 1. Check Ollama Connection & Model
    const isOllamaReady = await ollamaEmbeddingService.checkConnection();
    if (!isOllamaReady) {
      console.error("❌ Ollama not responding or embedding model missing.");
      return;
    }
    console.log("✅ Ollama Connection: OK");

    // 2. Generate a test embedding
    const testText = "Pragya is an offline educational platform.";
    const embedding = await ollamaEmbeddingService.generateEmbedding(testText);
    console.log(`✅ Embedding Generated (Dimensions: ${embedding.length})`);

    // 3. Store in ChromaDB
    const testChunk = {
      id: "test-chunk-1",
      content: testText,
      metadata: {
        fileId: "test-pdf-001",
        fileName: "test.pdf",
        page: 1,
      },
    };

    await vectorDBService.storeChunks(
      [testChunk],
      [embedding],
      "test_collection",
    );
    console.log("✅ Data stored in ChromaDB");

    // 4. Query it back
    const searchResults = await vectorDBService.queryChunks(
      embedding,
      1,
      "test_collection",
    );
    console.log(
      "✅ Query successful. Found document:",
      searchResults.documents[0],
    );

    console.log(
      "\n✨ ALL SYSTEMS GO! Your local AI stack is fully integrated.",
    );
  } catch (error) {
    console.error("❌ Pipeline Test Failed:", error);
  }
}

testRAGPipeline();
