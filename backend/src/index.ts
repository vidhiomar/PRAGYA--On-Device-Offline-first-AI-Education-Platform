import express, { Request, Response } from "express";
import cors from "cors";
import { fileURLToPath } from "url";
import path from "path";
import env from "./src/config/env";
import { connectDatabase } from "./src/config/database";
import { errorHandler } from "./src/middleware/error.middleware";
import uploadRoutes from "./src/routes/upload.routes";
import queryRoutes from "./src/routes/query.routes";
import chatRoutes from "./src/routes/chat.routes";
import browseRoutes from "./src/routes/browse.routes";
import posterRoutes from "./src/routes/poster.routes";
import lmrRoutes from "./src/routes/lmr.routes";
import stitchRoutes from "./src/routes/stitch.routes";
import filesRoutes from "./src/routes/files.routes";
import speechRoutes from "./src/routes/speech.routes";
import analyzeRoutes from "./src/routes/analyze.routes";
import boardRoutes from "./src/routes/board.routes";
import documentTreeRoutes from "./src/routes/documentTree.routes";
import planRoutes from "./src/routes/plan.routes";

const app = express();

// ESM workaround for __dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Middleware
// CORS - Allow all origins
app.use(
  cors({
    origin: true, // Allow all origins
    credentials: true,
    methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allowedHeaders: [
      "Content-Type",
      "Authorization",
      "X-Requested-With",
      "Accept",
    ],
    exposedHeaders: ["Content-Type"],
  }),
);
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Routes
app.get("/", (req: Request, res: Response) => {
  res.json({
    success: true,
    message: "EduRAG Assistant API with Multilingual Support",
    version: "3.1.0",
    features: [
      "PDF page-wise processing",
      "Multi-file type support (PDF, TXT, DOCX, PPT, Images)",
      "Document preview with file streaming",
      "Stateful chat history (MongoDB)",
      "Multi-document support",
      "Multilingual support (22 Indian languages)",
      "Autonomous language detection",
      "Chat-based isolation",
      "Source citations (PDF name, page number)",
    ],
    endpoints: {
      upload: "/api/upload",
      query: "/api/query",
      files: "/api/files/:fileId",
      chats: "/api/chats",
      browse: "/api/browse",
      posters: "/api/posters",
      lmr: "/api/lmr",
      stitch: "/api/stitch",
      speech: "/api/speech/transcribe",
      health: "/api/query/health",
      stats: "/api/upload/stats",
    },
  });
});

app.use("/api/upload", uploadRoutes);
app.use("/api/query", queryRoutes);
app.use("/api/chats", chatRoutes);
app.use("/api/browse", browseRoutes);
app.use("/api/posters", posterRoutes);
app.use("/api/lmr", lmrRoutes);
app.use("/api/stitch", stitchRoutes);
app.use("/api/files", filesRoutes);
app.use("/api/speech", speechRoutes);
app.use("/api/analyze", analyzeRoutes);
app.use("/api/board", boardRoutes);
app.use("/api/document-tree", documentTreeRoutes);
app.use("/api/plan", planRoutes);

// Error handling middleware (must be last)
app.use(errorHandler);

// Initialize and start server
async function startServer() {
  // Connect to MongoDB (optional - app works without it)
  await connectDatabase();

  // Check ChromaDB connection
  try {
    const chromaResponse = await fetch(`${env.CHROMA_URL}/api/v1/heartbeat`);
    if (chromaResponse.ok) {
      console.log("ChromaDB connected");
    }
  } catch (error) {
    // ChromaDB not available - silent
  }

  // Check Ollama connection
  try {
    const ollamaUrl = env.OLLAMA_URL || "http://localhost:11434";
    const ollamaResponse = await fetch(`${ollamaUrl}/api/tags`);
    if (ollamaResponse.ok) {
      console.log("Ollama connected");
    }
  } catch (error) {
    // Ollama not available - silent
  }

  // Check Whisper availability
  const fs = await import("fs");
  const path = await import("path");
  const whisperPath = path.join(
    __dirname,
    "..",
    "bin",
    "whisper",
    "whisper-cli.exe",
  );
  if (fs.existsSync(whisperPath)) {
    console.log("Whisper is running");
  }

  // Start server
  app.listen(env.PORT, () => {
    console.log(`Server running on port ${env.PORT}`);
  });
}

startServer().catch((error) => {
  console.error("Failed to start server:", error);
  process.exit(1);
});
 