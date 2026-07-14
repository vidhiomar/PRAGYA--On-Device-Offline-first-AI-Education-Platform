# Pragya - Where Knowing Begins 

Welcome to **Pragya** 

## Project Overview

**Pragya** is a lightweight, multilingual on device AI-powered educational platform that generates curriculum-aligned content in 22+ Indian languages. Designed for **100% offline operation** with **zero reliance on cloud services**, Pragya runs efficiently on low-resource devices (4-8GB RAM) without requiring internet connectivity.

**Pragya Won Second Place at Technovate: A National Level Hackathon**

## Tech Stack

### **Frontend**

- **React 19** with TypeScript
- **Vite**
- **Tailwind CSS**
- **React Router DOM**
- **Recharts** (Data Visualization)
- **KaTeX** (Math Rendering)
- **HTML5 Canvas API**

### **Backend**

- **Node.js + Express** with TypeScript
- **MongoDB** (Local session management)
- **ChromaDB** (Local vector embeddings)
- **Tesseract.js** (OCR)
- **Ollama** (Local AI inference)

### **Mobile**

- **Expo + React Native**
- **Expo Router**
- **Expo Camera**
- **Expo SQLite** (Local storage)
- **llama.rn** (On-device AI processing)

### **AI & Machine Learning**

- **DeepSeek-R1 1.5B** (Local reasoning model)
- **NLLB-200 600M** (Local multilingual translation)
- **ChromaDB** (Local vector database)
- **Tesseract.js** (Local OCR)

## Features

- **AI Whiteboard (AI-Board)** - Canvas-based drawing with AI-powered card operations
- **AI Chat Interface** - Document upload with RAG-powered multilingual chat
- **Learning Material Resource (LMR)** - Generate summaries, quizzes, and revision notes
- **Educational Poster Generation** - Culturally relevant educational posters
- **Offline Content Generator (Stitch)** - Real-time multilingual content generation
- **Mobile Application** - Cross-platform offline-capable mobile app

## Setup and Installation

1. **Clone this repository:**
   ```bash
   git clone https://github.com/Samarth2190/Pragya.git
   ```
2. **Install frontend dependencies:**
   ```bash
   cd frontend
   npm install
   ```
3. **Install backend dependencies:**
   ```bash
   cd ../backend
   npm install
   ```
4. **Setup Ollama (Required for AI features):**
   ```bash
   # Install Ollama from https://ollama.ai
   # Pull required models
   ollama pull deepseek-r1:1.5b
   ```
5. **Setup ChromaDB (Required for vector storage):**
   ```bash
   # ChromaDB runs locally, no cloud setup needed
   # Follow ChromaDB installation guide
   ```
6. **Start backend server:**
   ```bash
   cd backend
   npm run dev
   ```
7. **Start frontend development server:**
   ```bash
   cd frontend
   npm run dev
   ```

**Note:** Pragya is **100% offline** and requires **no external API keys** or cloud services. All AI processing happens locally using Ollama and local models.

## Contributing

**Contributions are welcomed!**  
Let's keep the spirit of open source alive!

---
