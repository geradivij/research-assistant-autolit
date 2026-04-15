# autolit

A web-first, multi-agent pipeline for academic literature review. Drop in PDFs, let autolit index and summarize them, choose which papers to include, then generate a structured Markdown survey.

```
PDF upload -> per-paper FAISS indexes -> structured summaries -> multi-paper survey
```

The app uses FAISS plus a local sentence-transformers embedding model for retrieval, and Groq-hosted chat models for extraction and survey writing. PDF contents are stored locally in `data/` and generated outputs are written to `outputs/`.

---

## What It Does

1. **Ingests PDFs** - extracts text, chunks it with overlap, and builds one FAISS vector index per paper.
2. **Extracts structured summaries** - retrieves targeted context for 7 fields: task, approach, datasets, metrics, key results, limitations, and overview.
3. **Generates a survey** - selector, comparator, and writer agents choose relevant papers, compare them, and produce a Markdown literature survey.
4. **Provides a browser UI** - drag-and-drop PDFs, live paper status, visible ingestion errors, paper include/exclude controls, streamed pipeline logs, and rendered survey output.

---

## Requirements

- Python 3.10+
- A Groq API key in `GROQ_API_KEY`
- Internet access for Groq model calls
- Around 400 MB disk for the `all-MiniLM-L6-v2` sentence-transformers embedding model, downloaded on first use

---

## Setup

```bash
git clone https://github.com/your-username/research-assistant-agent
cd research-assistant-agent

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## Run The Web App

```bash
python run.py
```

Open:

```text
http://localhost:8000
```

From the UI you can:

- Drop or browse for PDFs.
- Watch each paper move through `queued`, `processing`, `indexed`, or `error`.
- Exclude indexed papers from survey consideration with the `x` button.
- Re-include excluded papers with the `+` button.
- Enter a topic and run the multi-agent survey pipeline.
- Export the generated Markdown survey.

Errored papers are hidden from the left panel so they do not clutter the active paper list.

---

## CLI Usage

The browser UI is the main path, but the original scripts are still useful for direct runs.

### Summarize One Paper

Put a PDF in `data/pdfs/`. The filename without `.pdf` is the `paper_id`.

```bash
python scripts/summarize_paper.py --paper_id attention_is_all_you_need
```

This builds the FAISS index if needed and writes:

```text
outputs/summaries/attention_is_all_you_need.json
```

The JSON contains:

```text
task, approach, datasets, metrics, key_results, limitations, notes
```

You can pass a Groq model name:

```bash
python scripts/summarize_paper.py --paper_id bert --model llama-3.3-70b-versatile
```

### Generate A Survey

Once summaries exist:

```bash
python scripts/generate_survey.py --topic "transformer architectures" --top_k 3
```

Output:

```text
outputs/surveys/transformer_architectures_survey.md
```

Options:

```text
--topic          Research topic (required)
--top_k          Number of papers to select (default: 3)
--summaries_dir  Where to read summaries (default: outputs/summaries)
--surveys_dir    Where to write the survey (default: outputs/surveys)
```

---

## How It Works

**Ingestion** - PyMuPDF extracts page text. Text is split into 1000-character chunks with 200-character overlap. Chunks are embedded with `all-MiniLM-L6-v2` and stored in a FAISS flat index under `data/indexes/`.

**Extraction** - Each summary field runs independently. The app retrieves field-specific chunks and asks the LLM a small targeted question. This is more reliable than asking for one large JSON object in a single prompt.

**Survey generation** - `src/autolit/phase3/pipeline.py` wires three agents together. The selector picks relevant papers, the comparator creates a structured table and critique, and the writer turns those materials into a Markdown survey.

**Upload handling** - The API queues uploads and processes them one at a time. This avoids overlapping embedding/model work and keeps paper status visible in the UI.

For the full technical picture, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Project Layout

```text
data/pdfs/          Uploaded or manually added PDFs
data/indexes/       FAISS indexes and chunks, safe to rebuild
outputs/summaries/  Per-paper JSON summaries
outputs/surveys/    Generated Markdown surveys
src/autolit/api.py   FastAPI backend and static web serving
src/autolit/web/     Single-file React UI served by FastAPI
src/autolit/         Pipeline source code
  pdf_loader.py       PDF text extraction
  chunking.py         Overlapping text chunking
  vector_store.py     FAISS index build/load/search
  extraction_agent.py Field-by-field RAG extraction
  phase3/             Multi-agent survey pipeline
scripts/            CLI entry points
```

---

## Known Limitations

**Requires Groq API access.** The current LLM client uses `langchain-groq` and reads `GROQ_API_KEY` from the environment.

**Extraction quality varies.** Retrieval helps, but papers with unusual formatting, missing abstracts, or results buried in appendices may still produce weak summaries.

**No pytest suite yet.** There are script-style smoke tests, but no full unit/integration test suite.

**Paths are project-root relative.** Run the app and scripts from the repository root.

---

## Roadmap

- [ ] pytest suite with mocked LLM tests plus optional integration tests
- [ ] Model selection in the web UI
- [ ] Better retry/reprocess controls for failed papers
- [ ] Parallel field extraction with controlled concurrency
- [ ] `python -m autolit` entry point

See [DESIGN.md](DESIGN.md) for additional design notes.

---

## License

MIT
