# autolit

A local, multi-agent pipeline for academic literature review. Drop in PDFs, get back a structured survey — no API keys, no cloud, no data leaving your machine.

```
data/pdfs/*.pdf  →  per-paper summaries  →  multi-paper survey (Markdown)
```

Everything runs on a local Ollama instance + FAISS. The full pipeline for 3 papers takes roughly 5–10 minutes depending on your hardware.

---

## What It Does

1. **Ingests PDFs** — extracts text, chunks it with overlap, builds a FAISS vector index per paper
2. **Extracts structured summaries** — for each paper, independently retrieves and extracts 7 fields (task, approach, datasets, metrics, key results, limitations, overview) using RAG + a local LLM
3. **Generates a survey** — three agents work in sequence: a selector picks the most relevant papers for your topic, a comparator produces a comparison table and critique, a writer produces a full Markdown survey

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally with `llama3` pulled: `ollama pull llama3`
- ~400MB disk for the sentence-transformers embedding model (downloaded automatically on first use)

---

## Setup

```bash
git clone https://github.com/your-username/research-assistant-agent
cd research-assistant-agent

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Make sure Ollama is running:
```bash
ollama serve   # if not already running as a service
```

---

## Usage

### Step 1 — Add PDFs

Put your PDFs in `data/pdfs/`:
```
data/pdfs/
├── attention_is_all_you_need.pdf
├── bert.pdf
└── gpt4.pdf
```

The filename without `.pdf` becomes the `paper_id` used throughout the pipeline.

### Step 2 — Summarize papers

Run this for each paper you want to include:
```bash
python scripts/summarize_paper.py --paper_id attention_is_all_you_need
```

This builds the FAISS index (if it doesn't exist) and runs field-by-field RAG extraction. Output:
```
outputs/summaries/attention_is_all_you_need.json
```

The JSON has 7 fields: `task`, `approach`, `datasets`, `metrics`, `key_results`, `limitations`, `notes`.

You can pass a different Ollama model:
```bash
python scripts/summarize_paper.py --paper_id bert --model llama3.2
```

### Step 3 — Generate a survey

Once you've summarized at least 2 papers:
```bash
python scripts/generate_survey.py --topic "transformer architectures" --top_k 3
```

Three agents run in sequence:
1. **Selector** — picks the `top_k` most relevant papers for your topic
2. **Comparator** — produces a structured comparison table + critique
3. **Writer** — writes a full Markdown survey with Introduction, Individual Papers, Comparison, Limitations, and Future Work sections

Output: `outputs/surveys/transformer_architectures_survey.md`

Options:
```
--topic          The research topic (required)
--top_k          Number of papers to select (default: 3)
--summaries_dir  Where to read Phase 2 summaries (default: outputs/summaries)
--surveys_dir    Where to write the survey (default: outputs/surveys)
```

---

## How It Works

**Ingestion** — PyMuPDF extracts text page by page. Text is split into 1000-character chunks with 200-character overlap (so content near boundaries appears in two chunks, improving retrieval). Chunks are embedded with `all-MiniLM-L6-v2` (sentence-transformers) and stored in a FAISS flat index on disk.

**Extraction** — 7 independent RAG queries, one per output field. Each field has a targeted search query (e.g., the `datasets` query looks for "dataset names like CIFAR, ImageNet, WikiText"). Top-k retrieved chunks become the context for a single, strict LLM prompt. Separate calls per field are more reliable than asking for all 7 in one prompt — local models frequently produce malformed JSON on large output schemas.

**Survey generation** — three agents wired by `phase3/pipeline.py`. The selector sends all paper summaries to the LLM and asks for ranked paper IDs as JSON. The comparator receives the selected papers and produces a comparison table (JSON) and prose critique in delimited blocks. The writer takes all inputs and produces structured Markdown. All agents use a repair loop that re-prompts the LLM if JSON parsing fails.

For the full technical picture, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Project Layout

```
data/pdfs/          Put your PDFs here
data/indexes/       FAISS indexes (auto-built, safe to delete)
outputs/summaries/  Per-paper JSON summaries
outputs/surveys/    Generated Markdown surveys
src/autolit/        Pipeline source code
  pdf_loader.py       PDF text extraction
  chunking.py         Overlapping text chunking
  vector_store.py     FAISS index build/load/search
  extraction_agent.py Field-by-field RAG extraction
  phase3/             Multi-agent survey pipeline
    contracts.py        JSON parsing + LLM repair loop
    selection_agent.py  Agent 1: pick relevant papers
    comparator_agent.py Agent 2: compare + critique
    writer_agent.py     Agent 3: write Markdown survey
    pipeline.py         Orchestrator
scripts/            CLI entry points
```

---

## Known Limitations

**Extraction quality varies.** Local 8B models sometimes misread a paper's topic, especially if the relevant sections are buried or the paper has unusual structure. The per-field retrieval approach helps but doesn't fully solve this.

**No tests yet.** Correctness is verified by reading outputs. A pytest suite (unit + integration) is the immediate next priority — see [DESIGN.md](DESIGN.md).

**CLI-only.** A web UI with live pipeline streaming is in design — see [DESIGN.md](DESIGN.md).

**Phase 3 model is hard-coded to `llama3`.** Phase 2 accepts `--model`; Phase 3 doesn't yet.

**Runs from project root only.** Paths are relative to the working directory. Running scripts from other directories will fail.

---

## Roadmap

- [ ] pytest suite — unit tests (mocked, no Ollama) + integration tests
- [ ] FastAPI + React web UI — drag-and-drop PDFs, live agent log, rendered survey
- [ ] `python -m autolit` entry point
- [ ] Parallel field extraction in Phase 2
- [ ] Model selection in Phase 3

See [DESIGN.md](DESIGN.md) for the full plan with API contracts, component specs, and build order.

---

## License

MIT
