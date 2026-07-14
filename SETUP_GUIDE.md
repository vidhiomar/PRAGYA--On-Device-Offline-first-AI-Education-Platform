# Pragya Project Setup Guide

## 📋 Project Overview

Pragya is a **an On Device, lightweight multilingual educational AI system** designed for rural Indian students. It uses:

- **DeepSeek R1 1.5B** - Language model (reasoning & content generation)
- **NLLB 600M** - Translation across 22+ Indian languages
- **ChromaDB** - Vector embeddings (local)
- **MongoDB** - State management (local)
- **Ollama** - Local LLM inference engine

### 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Pragya Stack                          │
├─────────────────────────────────────────────────────────┤
│  Frontend (React + Vite)    │  Mobile (React Native)    │
│  - Port: 5173              │  - Expo                   │
│  - Routes: /pitch, /       │  - iOS/Android            │
├─────────────────────────────────────────────────────────┤
│             Backend API (Express + TypeScript)          │
│             - Port: 5001                                │
│             - ESM Modules                               │
├─────────────────────────────────────────────────────────┤
│  Ollama (LLM)  │ ChromaDB │ MongoDB  │  NLLB Service   │
│  Port: 11434   │ 8000     │  27017   │  Python proxy   │
└─────────────────────────────────────────────────────────┘
```

---

## 📦 Prerequisites

✅ Verified on your system:

- **Node.js**: v20.17.0
- **npm**: 10.8.2
- **Python**: 3.12.5

### Additional Requirements

1. **Ollama** (for running LLMs locally)
   - Download: https://ollama.ai
   - Required models: `deepseek-r1:1.5b`, `embeddinggemma:latest`

2. **MongoDB** (for state management)
   - Download: https://www.mongodb.com/try/download/community
   - Or use: `docker run -d -p 27017:27017 --name mongodb mongo:latest`

3. **ChromaDB** (for vector embeddings)
   - Runs as Python service on port 8000

4. **Git** (already have it)

---

## 🚀 Quick Start (5 Steps)

### Step 1: Verify All Dependencies Are Installed

```bash
# Backend
cd D:\Pragya\backend
npm install

# Frontend
cd D:\Pragya\frontend
npm install

# Mobile App
cd D:\Pragya\app
npm install
```

**Status**: ✅ Already done

---

### Step 2: Start External Services

These need to run BEFORE starting the backend:

#### A. Start MongoDB

```bash
# Option 1: Docker (recommended)
docker run -d -p 27017:27017 --name pragya-mongo mongo:latest

# Option 2: Local MongoDB installation
mongod --dbpath C:\data\mongodb
```

#### B. Start Ollama

```bash
# 1. Install Ollama: https://ollama.ai
# 2. Pull required models
ollama pull deepseek-r1:1.5b
ollama pull embeddinggemma:latest

# 3. Start Ollama service
ollama serve
# Ollama will start on port 11434
```

#### C. Start ChromaDB (Vector Database)

```bash
cd D:\Pragya\backend
python3 -m pip install chromadb

# Start ChromaDB server
chroma run --host localhost --port 8000
```

#### D. Setup NLLB Translation Service (Optional but Recommended)

```bash
# Install Python dependencies
cd D:\Pragya\backend
python3 -m pip install -r proxy/requirements.txt

# Note: Models are auto-downloaded on first run (~2GB)
```

---

### Step 3: Start the Backend

```bash
cd D:\Pragya\backend
npm run dev
```

**Expected Output:**

```
[nodemon] starting `tsx src/index.ts src/index.ts`
Server running on port 5001
ChromaDB connected
Ollama connected
```

**Backend API available at**: `http://localhost:5001`

---

### Step 4: Start the Frontend

```bash
cd D:\Pragya\frontend
npm run dev
```

**Expected Output:**

```
VITE v7.3.0  ready in 500 ms

➜  Local:   http://localhost:5173/
➜  Pitch:   http://localhost:5173/pitch
```

**Access in browser**: `http://localhost:5173`

---

### Step 5: (Optional) Start Mobile App

```bash
cd D:\Pragya\app
npm start
# Select 'a' for Android emulator or 'i' for iOS simulator
```

---

## 🔧 Configuration

### Backend Environment Variables

Create `.env` file in `backend/` directory (currently hardcoded in `src/config/env.ts`):

```env
# Server
PORT=5001
NODE_ENV=development

# Database
MONGODB_URI=mongodb://localhost:27017/masterg

# Vector DB
CHROMA_URL=http://localhost:8000
CHROMA_COLLECTION_NAME=edu_notes

# LLM Services
OLLAMA_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=deepseek-r1:1.5b
OLLAMA_EMBED_MODEL=embeddinggemma:latest

# Translation
NLLB_ENABLED=true

# File Upload
UPLOAD_DIR=./uploads
MAX_FILE_SIZE=31457280
```

---

## 📡 API Endpoints

### Main Endpoints

| Method | Endpoint                 | Purpose                        |
| ------ | ------------------------ | ------------------------------ |
| GET    | `/`                      | API info & health check        |
| POST   | `/api/upload`            | Upload files (PDF, DOCX, etc.) |
| POST   | `/api/query`             | Query documents with AI        |
| POST   | `/api/chats`             | Chat with context              |
| POST   | `/api/speech/transcribe` | Speech to text                 |
| GET    | `/api/files/:fileId`     | Download file                  |
| POST   | `/api/analyze`           | Analyze document content       |

### Example: Query the API

```bash
curl -X POST http://localhost:5001/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is photosynthesis?",
    "language": "en"
  }'
```

---

## 🌐 Frontend Pages

| Route         | Purpose                             |
| ------------- | ----------------------------------- |
| `/`           | Home page                           |
| `/pitch`      | Project pitch (animations, journey) |
| `/benchmarks` | Performance benchmarks              |
| Other routes  | Loaded dynamically                  |

---

## 📱 Mobile App

Built with **React Native + Expo**

```bash
# Start development
cd app
npm start

# Run on Android
npm run android

# Run on iOS
npm run ios
```

---

## 🛠️ Development Workflow

### Backend Development

```bash
cd D:\Pragya\backend
npm run dev      # Watch mode with nodemon + tsx
npm run build    # Compile TypeScript
npm run typecheck # Check types without emitting
```

### Frontend Development

```bash
cd D:\Pragya\frontend
npm run dev      # Dev server with HMR
npm run build    # Production build
npm run preview  # Preview production build
npm run lint     # Check code quality
```

### Type Checking

```bash
# Backend
cd D:\Pragya\backend
npm run typecheck

# Frontend
cd D:\Pragya\frontend
npm run typecheck
```

---

## 🐛 Troubleshooting

### Backend Won't Start

**Error**: `Error [ERR_REQUIRE_ESM]: require() of ES Module`

- **Solution**: Ensure `"type": "module"` is in `package.json` and `tsconfig.json` has `"module": "esnext"`

**Error**: `ECONNREFUSED 127.0.0.1:11434` (Ollama not running)

- **Solution**: Start Ollama: `ollama serve`

**Error**: `ECONNREFUSED 127.0.0.1:27017` (MongoDB not running)

- **Solution**: Start MongoDB or Docker: `docker run -d -p 27017:27017 mongo`

### Frontend Port 5173 Already in Use

```bash
# Kill the process on port 5173
netstat -ano | findstr :5173
taskkill /PID <PID> /F
```

### Python Service Issues

```bash
# Check if Python 3.12 is available
python3 --version

# Install missing dependencies
cd D:\Pragya\backend
python3 -m pip install -r proxy/requirements.txt
```

---

## 📊 Project Structure

```
D:\Pragya/
├── backend/                 # Node.js Express API
│   ├── src/
│   │   ├── index.ts        # Entry point
│   │   ├── config/         # Configuration
│   │   ├── services/       # Business logic
│   │   ├── routes/         # API routes
│   │   ├── middleware/     # Express middleware
│   │   └── types/          # TypeScript types
│   ├── proxy/              # Python services
│   │   ├── nllb_server.py  # Translation service
│   │   └── setup_models.py
│   ├── package.json
│   ├── tsconfig.json
│   └── nodemon.json
│
├── frontend/               # React + Vite
│   ├── src/
│   │   ├── App.tsx
│   │   ├── pages/          # Page components
│   │   ├── components/     # Reusable components
│   │   ├── services/       # API clients
│   │   └── hooks/          # Custom hooks
│   ├── public/
│   ├── package.json
│   ├── vite.config.ts
│   └── tailwind.config.ts
│
├── app/                    # React Native (Expo)
│   ├── app/                # Navigation structure
│   ├── components/         # React Native components
│   ├── hooks/              # Custom hooks
│   ├── services/           # API integration
│   ├── app.json            # Expo config
│   └── package.json
│
├── docs/                   # Documentation
├── SETUP_GUIDE.md         # This file
└── README.md              # Main README
```

---

## 🔌 Port Reference

| Service      | Port  | URL                       |
| ------------ | ----- | ------------------------- |
| Backend API  | 5001  | http://localhost:5001     |
| Frontend     | 5173  | http://localhost:5173     |
| Ollama       | 11434 | http://localhost:11434    |
| ChromaDB     | 8000  | http://localhost:8000     |
| MongoDB      | 27017 | mongodb://localhost:27017 |
| NLLB Service | 5555  | (Python subprocess)       |

---

## 🚢 Deployment Notes

### Production Build

```bash
# Frontend
cd D:\Pragya\frontend
npm run build
# Output: dist/

# Backend
cd D:\Pragya\backend
npm run build
# Output: dist/
```

### Deployment Platforms

- **Frontend**: Vercel, Netlify, AWS S3 + CloudFront
- **Backend**: Heroku, AWS EC2, DigitalOcean, Azure
- **Mobile**: Apple App Store, Google Play Store (via Expo)

### Environment Variables for Production

Update in deployment platform's secrets:

```
MONGODB_URI=<production-mongodb-url>
OLLAMA_URL=<your-ollama-server>
CHROMA_URL=<your-chroma-server>
```

---

## 📚 Technology Stack Summary

| Layer           | Technology                  | Purpose                |
| --------------- | --------------------------- | ---------------------- |
| **Frontend**    | React 19 + Vite             | Web UI                 |
| **Mobile**      | React Native + Expo         | Mobile apps            |
| **Backend**     | Express + TypeScript (ESM)  | API server             |
| **LLM**         | DeepSeek R1 1.5B via Ollama | Content generation     |
| **Translation** | NLLB 600M                   | Multi-language support |
| **Vector DB**   | ChromaDB                    | Semantic search        |
| **State DB**    | MongoDB                     | Data persistence       |
| **Styling**     | Tailwind CSS                | UI design              |
| **Icons**       | Lucide React                | Icon library           |
| **Charts**      | Recharts                    | Data visualization     |

---

## ✅ Verification Checklist

- [ ] Node.js v20+ installed
- [ ] npm 10+ installed
- [ ] Python 3.12+ installed
- [ ] MongoDB running
- [ ] Ollama running with required models
- [ ] ChromaDB service running
- [ ] Backend dependencies installed
- [ ] Frontend dependencies installed
- [ ] Backend starts: `npm run dev` (backend/)
- [ ] Frontend starts: `npm run dev` (frontend/)
- [ ] Can access http://localhost:5173
- [ ] Can access http://localhost:5001

---

## 🆘 Need Help?

1. **Check logs**: Look at terminal output for error messages
2. **Verify services**: Make sure MongoDB, Ollama, ChromaDB are running
3. **Port conflicts**: Ensure ports 5001, 5173, 8000, 11434 are free
4. **Dependencies**: Run `npm install` again in each directory
5. **Check README**: See project root [README.md](README.md) for more details

---

**Last Updated**: July 15, 2026
**Status**: ✅ Setup Complete
