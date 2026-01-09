# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Run Commands

**Always use uv to manage all dependencies and run the server. Do not use mamba or pip directly.**

```bash
# Install dependencies
uv sync

# Run the application (from project root)
./run.sh

# Or run manually (from backend directory)
cd backend && uv run uvicorn app:app --reload --port 8000
```

The web interface runs at http://localhost:8000. API docs available at http://localhost:8000/docs.

## Architecture

This is a RAG (Retrieval-Augmented Generation) chatbot for querying course materials.

### Request Flow

```
Frontend (vanilla JS) → FastAPI → RAGSystem → AIGenerator → Claude API
                                      ↓
                               ToolManager → CourseSearchTool → VectorStore (ChromaDB)
```

1. User submits query via chat UI (`frontend/script.js`)
2. FastAPI endpoint receives POST `/api/query` (`backend/app.py`)
3. `RAGSystem.query()` orchestrates the process (`backend/rag_system.py`)
4. `AIGenerator` calls Claude with available tools (`backend/ai_generator.py`)
5. If Claude decides to search, `CourseSearchTool` executes against ChromaDB (`backend/search_tools.py`)
6. Claude synthesizes final answer from search results
7. Response returned with sources to frontend

### Key Components

- **RAGSystem** (`backend/rag_system.py`): Main orchestrator that coordinates document processing, vector storage, AI generation, and session management
- **AIGenerator** (`backend/ai_generator.py`): Wraps Claude API with tool-calling support. Uses `claude-sonnet-4-20250514` model
- **VectorStore** (`backend/vector_store.py`): ChromaDB wrapper with two collections: `course_catalog` (metadata) and `course_content` (chunks). Uses `all-MiniLM-L6-v2` embeddings
- **DocumentProcessor** (`backend/document_processor.py`): Parses course documents and chunks text (800 chars, 100 overlap)
- **ToolManager/CourseSearchTool** (`backend/search_tools.py`): Implements Anthropic tool-calling interface for semantic search

### Document Format

Course documents in `docs/` follow this structure:
- Line 1: Course Title
- Line 2: Course Link (optional)
- Line 3: Instructor (optional)
- Remaining: Lesson markers (`Lesson X:`) and content

### Configuration

All settings in `backend/config.py`: API keys, model names, chunking params, ChromaDB path.

## Environment Setup

Requires `.env` file in project root:
```
ANTHROPIC_API_KEY=your-key-here
```
