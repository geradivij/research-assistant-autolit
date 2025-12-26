# Research Assistant Agent

An agentic research assistant for ingesting academic PDFs, extracting structured summaries, and producing comparative surveys across multiple papers.

This repository provides tools to:
- Load and extract text from PDFs
- Chunk documents and build FAISS vector indexes
- Run retrieval-augmented generation (RAG) extractions to produce structured JSON summaries per paper
- Orchestrate multi-paper selection, comparison, and survey generation

Quickstart
1. Create and activate a Python virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Add PDFs to `data/pdfs/`.

3. Build chunks and indexes (use provided scripts or call modules directly):

```powershell
# summarize a single paper
python scripts/summarize_paper.py --pdf data/pdfs/your_paper.pdf

# generate a multi-paper survey
python scripts/generate_survey.py --topic "your topic" --top_k 3
```

Development notes
- Configuration for model names, data directories, and embedding settings is currently in-code; consider centralizing settings.
- Tests in the repo are script-style and may invoke LLMs/FAISS; prefer adding pytest tests with mocks for CI.
- Avoid running tests that require a local LLM or FAISS unless those systems are installed and configured.

See `docs/PROJECT_OVERVIEW.md` for a more detailed architecture and component map.
# Research-AutoLit-Llama

An agentic AI research assistant built with Llama + LangChain.

Current status:
- [x] Local Llama via Ollama
- [x] Basic LangChain client test
- [x] PDF ingestion + chunking
- [x] Vector store + RAG
- [ ] Multi-agent orchestration
