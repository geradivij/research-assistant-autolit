# Project Overview: Research Assistant Agent

## Purpose
An agentic AI research assistant for ingesting academic PDFs, extracting structured summaries, and producing comparative surveys across multiple papers. The system is designed for local-first operation using a local LLM (e.g., Ollama/llama) and supports vector retrieval (FAISS) for RAG-style extraction.

## Data Flow
- PDFs (data/pdfs) → PDF loader (`src/autolit/pdf_loader.py`)
- Text extraction → Chunking (`src/autolit/chunking.py`)
- Embeddings + FAISS index (`src/autolit/vector_store.py` → `data/indexes/*`)
- Retrieval → RAG query (`src/autolit/rag_query.py` / `src/autolit/extraction_agent.py`)
- Per-paper Phase-2 JSON summaries (`outputs/summaries/*.json`)
- Multi-paper agents: selection, comparison, writer (`src/autolit/multi_paper_agents.py`) → survey outputs (`outputs/surveys/`)

## Major Components
- `src/autolit/pdf_loader.py`: Loads PDFs and extracts text pages.
- `src/autolit/chunking.py`: Breaks long text into overlapping chunks used for indexing.
- `src/autolit/vector_store.py`: Builds/loads FAISS index and manages mapping between chunks and index entries.
- `src/autolit/extraction_agent.py`: Performs field-wise RAG extraction and produces structured JSON outputs.
- `src/autolit/multi_paper_agents.py`: Orchestrates selection of papers, comparison, and survey writing.
- `src/llm/llama_client.py` and `src/autolit/llm/client.py`: Local LLM client wrappers used for prompting.

## Quick Usage
1. Install dependencies from `requirements.txt` into a virtual environment.
2. Prepare PDFs under `data/pdfs/`.
3. Build chunks and index using the pipeline scripts in `scripts/` or via the Python modules.
4. Run extraction and survey generation:

```bash
python scripts/summarize_paper.py --pdf data/pdfs/your_paper.pdf
python scripts/generate_survey.py --topic "your topic" --top_k 3
```

## Development Notes
- Tests in the repo are script-style and may invoke LLMs and FAISS. Prefer adding pytest-style unit tests with mocks for CI.
- Configuration (model name, data dirs) is currently hard-coded in places — consider centralizing in a `config.py` or environment-driven settings.
- Avoid heavy imports at module import time; prefer lazy initialization for LLMs and embedding models.

## Next Steps
- Add a concise `README.md` (this repo) and a project overview (this file) — completed.
- Consider unifying LLM wrappers and adding CI-friendly unit tests that mock external services.

For more details, inspect the `src/` folder and `scripts/` directory for example workflows.