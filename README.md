# Course Materials RAG System

A Retrieval-Augmented Generation (RAG) system designed to answer questions about course materials using semantic search and AI-powered responses.

## Overview

This application is a full-stack web application that enables users to query course materials and receive intelligent, context-aware responses. It uses ChromaDB for vector storage, Ollama for local AI generation, and provides a web interface for interaction.


## Prerequisites

- Python 3.13 or higher
- Mamba (or Conda) package manager
- Ollama installed and running locally
- **For Windows**: Use Git Bash to run the application commands - [Download Git for Windows](https://git-scm.com/downloads/win)

## Installation

1. **Install Mamba** (if not already installed)
   ```bash
   # Install Miniforge which includes mamba
   # See: https://github.com/conda-forge/miniforge
   ```

2. **Install Python dependencies**
   ```bash
   mamba install --file requirements.txt
   ```

3. **Install Ollama and pull a model**
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ollama pull qwen2.5:7b
   ```

4. **Set up environment variables** (optional - defaults work out of the box)

   Create a `.env` file in the root directory:
   ```bash
   # Optional overrides (defaults shown)
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=qwen2.5:7b
   ```

## Running the Application

### Quick Start

Use the provided shell script:
```bash
chmod +x run.sh
./run.sh
```

### Manual Start

```bash
cd backend
mamba run uvicorn app:app --reload --port 8000
```

The application will be available at:
- Web Interface: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`
