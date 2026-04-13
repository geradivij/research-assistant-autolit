# autolit — Architecture & Codebase Guide

A deep-dive into everything that's been built: what each piece does, why it was designed that way, and how the pieces fit together.

---

## What This Is

autolit is a **local-first, multi-agent pipeline** for academic literature review. You give it a folder of PDFs and a research topic. It reads every paper, extracts structured summaries, selects the most relevant ones, compares them across dimensions, and writes a Markdown survey — all without sending a byte to the cloud.

The whole thing runs on:
- **Ollama** (local LLM server, typically `llama3`)
- **FAISS** (in-process vector similarity search)
- **sentence-transformers** (local embedding model)

No OpenAI API key. No cloud. No rate limits.

---

## High-Level Data Flow

```
data/pdfs/paper.pdf
        │
        ▼  Phase 1: Ingestion
┌─────────────────────┐
│  pdf_loader.py      │  Extract raw text, page by page
│  chunking.py        │  Split into overlapping 1000-char chunks
│  vector_store.py    │  Embed chunks → FAISS index
└─────────────────────┘
        │
        ▼  Stored in:
   data/indexes/<paper_id>/
     ├── index.faiss      ← FAISS flat index
     ├── metadata.json    ← chunk→page mapping
     └── chunks.json      ← full chunk text

        │
        ▼  Phase 2: Per-Paper Extraction
┌─────────────────────────┐
│  extraction_agent.py    │  For each of 7 fields:
│                         │    1. Query the FAISS index with field-specific question
│                         │    2. Retrieve top-k chunks
│                         │    3. Ask the LLM one tight question
│                         │    4. Parse the answer
└─────────────────────────┘
        │
        ▼  Stored in:
   outputs/summaries/<paper_id>.json
     {task, approach, datasets, metrics, key_results, limitations, notes}

        │
        ▼  Phase 3: Multi-Agent Survey
┌────────────────────────────────────┐
│  phase3/selection_agent.py         │  LLM picks top-k relevant papers
│  phase3/comparator_agent.py        │  LLM produces comparison table + critique
│  phase3/writer_agent.py            │  LLM writes full Markdown survey
│  phase3/pipeline.py                │  Orchestrates the three agents
└────────────────────────────────────┘
        │
        ▼  Stored in:
   outputs/surveys/<topic>_survey.md
```

---

## Phase 1 — Ingestion & Indexing

### `src/autolit/pdf_loader.py`

Uses **PyMuPDF** (`fitz`) to open a PDF and extract text page by page. Returns a list of dicts:

```python
[
    {"page_num": 1, "text": "Introduction. Large language models..."},
    {"page_num": 2, "text": "Related work. Prior approaches..."},
    ...
]
```

Page numbers are 1-based. Empty pages are included (empty string text). The extraction is purely textual — figures, equations rendered as images, and multi-column layouts may not extract cleanly.

### `src/autolit/chunking.py`

Takes the page list and splits it into fixed-size overlapping chunks. Default: **1000 characters per chunk, 200-character overlap**.

```
Page text: |----1000----|
                |----1000----|    ← overlapping by 200 chars
                        |----1000----|
```

Overlap is critical for RAG: it ensures that content near chunk boundaries appears in at least two chunks, so a retrieval query that's relevant to boundary-straddling content has a chance of matching.

Each chunk carries:
```python
{
    "chunk_id": 0,        # global sequential ID
    "page_num": 1,        # which page this chunk came from
    "text": "chunk text"
}
```

Chunks are **per-page**: a new page starts a new sequence of chunks, so chunk boundaries never cross page boundaries. This preserves page attribution in the retrieved context.

### `src/autolit/vector_store.py`

This is where the FAISS index lives. The class is `FaissVectorStore`.

**Building an index** (`FaissVectorStore.from_chunks`):

1. Extract the `text` field from each chunk
2. Embed all texts using `sentence-transformers` model `all-MiniLM-L6-v2`
   - 384-dimensional embeddings
   - ~22M parameters, runs on CPU in a few seconds per paper
3. L2-normalize the embeddings (so inner product = cosine similarity)
4. Build a `faiss.IndexFlatIP` (flat inner product index — exact search, no approximation)
5. Store chunk metadata (chunk_id, page_num) in a parallel list

**Saving** writes two files:
- `index.faiss` — the serialized FAISS index
- `metadata.json` — embedding dimension, model name, list of ChunkMetadata

**Searching** (`FaissVectorStore.search`):
1. Embed the query string using the same model
2. L2-normalize
3. Run `index.search(query_vec, k)` — returns top-k cosine scores and indices
4. Map indices back to ChunkMetadata objects

The design uses `IndexFlatIP` (exact search) not `IndexIVFFlat` (approximate). For paper-sized corpora (100–500 chunks), exact search is fast enough and avoids the ANN accuracy tradeoff.

### `src/autolit/rag_query.py`

Orchestrates Phase 1: given a PDF path, it calls `pdf_loader` → `chunking` → `vector_store.from_chunks` → `store.save`. Also handles the "build if missing, load if exists" caching pattern used throughout the project.

Exports constants `PDF_DIR` and `INDEX_DIR` (both relative to the working directory) that are imported by `extraction_agent.py`.

---

## Phase 2 — Per-Paper Extraction

### `src/autolit/extraction_agent.py`

This is the most important design decision in the whole project: **extract one field at a time, not all fields in one prompt.**

The naive approach is to send all of a paper's text to an LLM and ask it to return a JSON with 7 fields. This fails for several reasons:
- Context window pressure causes the LLM to hallucinate missing fields
- The LLM fills fields from the wrong parts of the paper
- JSON generation failures cascade: one bad field breaks the whole object

The autolit approach is different. For each of the 7 output fields, it:

1. **Queries FAISS with a field-specific question** — each field has a tailored search query that targets the most relevant section of the paper. For example, `datasets` searches for "Which datasets are used in the experiments? Look for dataset names like CIFAR, ImageNet..."

2. **Retrieves the top 5–8 chunks** — a small, tight context window (~8 chunks ≈ ~8000 chars) focused on the field

3. **Asks a single, strict question** — a short prompt that asks for only one thing, forbids meta-commentary ("Do not say 'in this context...'"), and explicitly says what format to use

4. **Parses the result independently** — list fields (`datasets`, `metrics`) are parsed line-by-line; text fields are taken as-is

The field queries are defined in `FIELD_QUERIES`:

| Field | What the search looks for |
|---|---|
| `task` | Research goal, problem statement |
| `approach` | "We propose", "our method", method description |
| `datasets` | Dataset names (CIFAR, ImageNet, WikiText...) |
| `metrics` | Metric names (accuracy, F1, BLEU, perplexity...) |
| `key_results` | Percentages, scores, improvement figures |
| `limitations` | Future work, drawbacks, open questions |
| `notes` | Whole-paper summary (uses broad query) |

**Output schema** — validated by Pydantic:

```python
class PaperSummary(BaseModel):
    task: str          # 3–5 sentences
    approach: str      # 4–6 sentences
    datasets: List[str]
    metrics: List[str]
    key_results: str   # 4–6 sentences
    limitations: str   # 3–5 sentences
    notes: str         # 150–250 word overview
```

The LLM used here is `langchain_ollama.ChatOllama` invoked directly — not the shared `chat()` wrapper. Temperature is 0.2 for determinism.

**Known limitation:** Because chunks are extracted per-page and queries are field-specific, papers with unusual structures (no abstract, results buried in appendices) can return poor context. The fallback when FAISS returns nothing is to use the first page's chunks — which helps for abstracts but misses results sections.

---

## Phase 3 — Multi-Agent Survey

Phase 3 takes the per-paper JSON summaries from Phase 2 and produces a multi-paper survey. It has four files: `pipeline.py` (orchestrator), `selection_agent.py`, `comparator_agent.py`, `writer_agent.py`, and `contracts.py` (shared parsing utilities).

### `src/autolit/phase3/pipeline.py`

The top-level function `run_survey_pipeline(topic, summaries_dir, surveys_dir, top_k)` runs the full sequence:

```
load_all_summaries()
    → select_papers_for_topic()
        → compare_papers()
            → write_survey_markdown()
                → save_survey_markdown()
```

Each step feeds its output to the next. If any step returns empty results, the pipeline raises and stops — there's no silent degradation.

### `src/autolit/phase3/io.py`

Handles reading Phase 2 summaries off disk and writing the final survey.

`PaperSummary` here is a plain dataclass (not Pydantic):
```python
@dataclass
class PaperSummary:
    paper_id: str
    summary: Dict[str, Any]  # the raw JSON dict from outputs/summaries/
```

Note: there are *two* `PaperSummary` types in the codebase:
- `extraction_agent.PaperSummary` — a Pydantic model, used during Phase 2 extraction
- `phase3/io.PaperSummary` — a plain dataclass, used during Phase 3 as a container

They serve different purposes: the Pydantic one validates field extraction output; the dataclass one is a lightweight carrier for already-validated JSON.

`load_all_summaries` scans `outputs/summaries/*.json` and loads each into a `PaperSummary`. `save_survey_markdown` slugifies the topic name and writes `outputs/surveys/<slug>_survey.md`.

### `src/autolit/phase3/selection_agent.py`

Takes all loaded `PaperSummary` objects and a topic string. Asks the LLM to pick the top-k most relevant papers.

**Prompt design:** The system prompt is kept deliberately short and strict. Long examples in the prompt were found to cause "explain the JSON" behavior — the model would write prose about what it would do rather than just outputting JSON. The current prompt is:

```
You are a Paper Selection Agent.
Given a topic and a list of papers (each has paper_id + short summary fields),
return the most relevant paper_ids.

Return ONLY valid JSON with exactly one key:
{"selected_papers": ["paper_id1", "paper_id2", "..."]}

Constraints:
- Only use these paper_id values: [...]
- Sort from most relevant to least relevant.
- Return at least {top_k} ids.
- No extra keys. No explanations. No markdown.
```

The user message is a JSON payload of all paper descriptions: `{topic, top_k, papers: [{paper_id, task, approach, datasets, metrics, key_results, notes}]}`.

**Fallback ranking:** If the LLM returns malformed JSON (or valid JSON but with invalid paper IDs), the agent falls back to `_fallback_rank_by_overlap` — a deterministic keyword-overlap ranker. It tokenizes the topic and each paper's concatenated fields, computes intersection size, and returns the top-k by overlap count. This ensures the pipeline never silently dies due to a bad LLM response.

**JSON parsing:** Uses `contracts.parse_or_repair_json` with `max_attempts=2` (see Contracts section below).

### `src/autolit/phase3/comparator_agent.py`

Receives the selected papers and asks the LLM to produce two things: a **structured comparison table** (JSON) and a **natural-language critique** (prose).

To get two distinct output types from one LLM call, the agent uses delimiter-based output formatting:

```
BEGIN_JSON
{ ... comparison table ... }
END_JSON

BEGIN_CRITIQUE
(critique text here)
END_CRITIQUE
```

The system prompt instructs the model to output EXACTLY these two blocks in this order with nothing outside them. In practice, models occasionally add preamble or swap the order — `contracts._extract_between` handles this by scanning for the start/end tags rather than assuming fixed positions.

The JSON block is parsed through the full `parse_or_repair_json` pipeline. The critique is extracted as plain text.

### `src/autolit/phase3/writer_agent.py`

The simplest of the three agents. Takes topic + selected papers + comparison table + critique, and asks the LLM to write a Markdown mini-survey with a fixed section structure:

```markdown
# Introduction
## Individual Papers
## Comparison
## Limitations
## Future Work
```

The user message is a JSON payload containing all inputs. The system prompt specifies the section structure explicitly. Output is plain Markdown — no delimiter parsing needed since the model is just asked to write prose.

Temperature is 0.3 (slightly higher than the other agents) to allow more natural-sounding prose.

### `src/autolit/phase3/contracts.py`

The most defensively engineered module in the codebase. LLM JSON output is notoriously fragile — models add prose preambles, wrap JSON in markdown code fences, introduce trailing commas, use single quotes, leave strings unescaped. `contracts.py` handles all of this.

**`_strip_code_fences(text)`** — removes ` ```json ` and ` ``` ` wrappers. Applied before every parse attempt.

**`_extract_json_object(text)`** — finds the first `{` and walks forward tracking brace depth and string escapes to find the matching `}`. This means even if the model outputs "Here is the JSON: {...} Hope that helps!", the object is extracted cleanly.

**`parse_or_repair_json(raw, max_attempts=2)`** — the main entry point:
1. Try `_parse_json_from_model(raw)` (strip fences → extract object → `json.loads`)
2. If that fails, ask the LLM to repair it: send the raw text to a "strict JSON repair assistant" and try parsing again
3. Repeat up to `max_attempts` times
4. If all attempts fail, raise `ContractRepairError` with full diagnostics

**`_extract_between(text, start_tag, end_tag)`** — extracts the substring between two delimiter tags. Supports `allow_missing_end=True` for cases where the model cuts off before writing `END_CRITIQUE`.

---

## The LLM Stack

There are two LLM client layers, reflecting the project's evolution:

### `src/llm/llama_client.py` (original)

The first LLM integration. Returns a `langchain_community.ChatOllama` instance. This was the initial client before the project restructured.

### `src/autolit/llm/client.py` (current)

The shared client used by all Phase 3 agents. Wraps `get_llm()` from the original client and exposes a single function:

```python
def chat(messages: List[Dict[str, str]], temperature: float = 0.2) -> str
```

All Phase 3 agents call `chat(messages, temperature=...)`. This is the seam that makes the agents testable — mocking `src.autolit.llm.client.chat` mocks all LLM calls in Phase 3.

**Note:** `extraction_agent.py` doesn't use `chat()` — it uses `langchain_ollama.ChatOllama` directly. This is an inconsistency from Phase 2 being written before the shared client was extracted. It means Phase 2 and Phase 3 use different call paths to reach the same Ollama server.

---

## The Compatibility Shim

`src/autolit/multi_paper_agents.py` is not a real module — it's a re-export shim:

```python
from src.autolit.phase3.io import load_all_summaries, save_survey_markdown, PaperSummary
from src.autolit.phase3.selection_agent import select_papers_for_topic
from src.autolit.phase3.comparator_agent import compare_papers
from src.autolit.phase3.writer_agent import write_survey_markdown
from src.autolit.phase3.pipeline import run_survey_pipeline
```

It exists because the original multi-agent code was all in one file (`multi_paper_agents.py`), which was then refactored into the `phase3/` package. Scripts that import from `multi_paper_agents` still work without modification.

Once `scripts/generate_survey.py` is updated to import directly from `phase3`, this shim can be deleted.

---

## Directory Structure

```
research-assistant-agent/
├── data/
│   ├── pdfs/                    ← Input: put PDFs here
│   │   ├── memorization.pdf
│   │   ├── dpscaling.pdf
│   │   └── TOFUpaper.pdf
│   └── indexes/                 ← Phase 1 output: FAISS indexes
│       └── <paper_id>/
│           ├── index.faiss
│           ├── metadata.json
│           └── chunks.json
│
├── outputs/
│   ├── summaries/               ← Phase 2 output: per-paper JSON
│   │   └── <paper_id>.json
│   └── surveys/                 ← Phase 3 output: Markdown surveys
│       └── <topic>_survey.md
│
├── src/
│   ├── llm/
│   │   └── llama_client.py      ← Original Ollama client (LangChain wrapper)
│   └── autolit/
│       ├── __init__.py
│       ├── pdf_loader.py        ← Phase 1: PDF text extraction
│       ├── chunking.py          ← Phase 1: Overlapping text chunking
│       ├── vector_store.py      ← Phase 1: FAISS index build/load/search
│       ├── rag_query.py         ← Phase 1: Orchestrator + constants
│       ├── extraction_agent.py  ← Phase 2: Field-by-field RAG extraction
│       ├── multi_paper_agents.py ← Compat shim (re-exports phase3)
│       ├── llm/
│       │   └── client.py        ← Shared chat() wrapper for Phase 3
│       └── phase3/
│           ├── __init__.py
│           ├── contracts.py     ← JSON parsing + repair
│           ├── io.py            ← Load summaries, save survey
│           ├── selection_agent.py ← Agent 1: pick relevant papers
│           ├── comparator_agent.py ← Agent 2: compare + critique
│           ├── writer_agent.py  ← Agent 3: write survey Markdown
│           └── pipeline.py      ← Orchestrate agents 1→2→3
│
├── scripts/
│   ├── summarize_paper.py       ← CLI: run Phase 2 on one paper
│   ├── generate_survey.py       ← CLI: run Phase 3 (full pipeline)
│   ├── phase3_compare.py        ← Dev/debug: run comparator alone
│   ├── phase3_select.py         ← Dev/debug: run selector alone
│   ├── phase3_write.py          ← Dev/debug: run writer alone
│   ├── test_comparator_agent.py ← Script-style smoke test
│   └── test_selection_agent.py  ← Script-style smoke test
│
├── docs/
│   └── PROJECT_OVERVIEW.md      ← Brief project overview
│
├── DESIGN.md                    ← Frontend + test suite design plan
├── ARCHITECTURE.md              ← This file
├── README.md                    ← Getting started
└── requirements.txt             ← Python dependencies
```

---

## Key Design Decisions

### Why field-by-field extraction?

The alternative — one big prompt asking for all 7 fields as JSON — fails in practice with local models. Problems:
1. **Context confusion**: The model mixes information from different sections when asked everything at once
2. **JSON generation instability**: Smaller models (~8B parameters) frequently produce malformed JSON when the object has 7 keys and text values
3. **Field interdependence**: Asking "what are the datasets?" in the same prompt as "summarize the paper" causes the summary to anchor on the datasets answer

Field-by-field extraction trades API call count (7 calls instead of 1) for reliability. Each call is small, simple, and independently retry-able.

### Why FAISS IndexFlatIP over approximate search?

Academic papers are typically 5,000–15,000 words, which produces 50–200 chunks. FAISS's flat index is exact and takes under 10ms to search even at 500 chunks. The additional complexity of `IndexIVFFlat` or `HNSW` is not worth it until the corpus is in the thousands of papers.

### Why delimiters (BEGIN_JSON/END_JSON) in the comparator?

Getting a local 8B model to produce two distinct output types (structured JSON + prose) reliably is hard. Options considered:
- **Two separate calls**: Clean, but doubles latency. The comparison and critique need the same context window
- **Single JSON object with a critique field**: Works poorly — the model tries to make prose JSON-safe
- **Delimiter blocks**: The model learns "output structured thing, then prose thing" reliably with this format

### Why the `contracts.py` repair loop?

Even with careful prompting, local LLMs produce malformed JSON ~20% of the time in practice. The two-attempt repair loop recovers from most common failures (code fences, trailing commas, single quotes) without surfacing errors to the user.

### Why a compatibility shim instead of just updating the imports?

The `multi_paper_agents.py` shim was the pragmatic choice during a live refactor. The original monolithic file was split into the `phase3/` package across multiple commits. The shim let `scripts/generate_survey.py` keep working during the refactor without requiring a simultaneous update to every caller.

---

## What the Output Actually Looks Like

**Phase 2 output** (`outputs/summaries/<paper_id>.json`):
```json
{
  "task": "This paper studies whether neural language models memorize training data...",
  "approach": "The authors propose a membership inference attack that extracts verbatim sequences...",
  "datasets": ["WikiText-103", "The Pile", "C4"],
  "metrics": ["extraction rate", "perplexity", "k-eidetic memorization score"],
  "key_results": "GPT-2 XL memorizes 3x more sequences than GPT-2 small...",
  "limitations": "Results are limited to autoregressive models; bidirectional models behave differently...",
  "notes": "150–250 word overview of the paper..."
}
```

**Phase 3 output** (`outputs/surveys/<topic>_survey.md`):
```markdown
# Introduction
This survey examines three recent works on memorization in language models...

## Individual Papers
### memorization
...
### dpscaling
...

## Comparison
| Paper | Task | Approach | Key metric | Limitation |
|---|---|---|---|---|
...

## Limitations
...

## Future Work
...
```

---

## Known Issues and Rough Edges

1. **Extraction quality is variable.** The `memorization.json` in the repo shows an example where the model misidentified the paper's topic (summarizing it as a PII extraction paper rather than a memorization-in-LMs paper). This happens when the retrieved chunks happen to come from an unrelated section of the paper. The field-by-field approach helps but doesn't eliminate this.

2. **No tests.** There are script-style smoke tests in `scripts/test_*.py` but no pytest suite. A broken `contracts.py` repair path or the selection fallback ranker would be invisible until the full pipeline runs against Ollama.

3. **LLM client inconsistency.** Phase 2 (`extraction_agent.py`) uses `langchain_ollama.ChatOllama` directly. Phase 3 uses the shared `chat()` wrapper in `src/autolit/llm/client.py`. Mocking for testing needs to target both paths.

4. **Hard-coded paths.** `PDF_DIR`, `INDEX_DIR`, and the summaries/surveys output paths are hard-coded relative to the working directory. Running scripts from anywhere other than the project root fails silently.

5. **`chat()` always uses `format="json"`** in `src/autolit/llm/client.py`. This tells Ollama to force JSON-mode output, which is good for structured responses but can interfere with the writer agent (which should produce Markdown, not JSON). The writer agent works despite this because it's a well-structured format that Ollama's JSON mode doesn't fully constrain.

6. **No concurrency.** Phase 2 extraction runs 7 sequential LLM calls per paper. With 10 papers, that's 70 sequential calls. Fields are independent and could be parallelized with `asyncio.gather`.

---

## What's Planned Next

See `DESIGN.md` for the full plan. The two main tracks:

**Testing** — a proper pytest suite with:
- Unit tests for `contracts.py`, the selection fallback ranker, and extraction field parsing (no Ollama required)
- Integration tests against real PDFs (`@pytest.mark.integration`)

**Frontend** — a FastAPI + React (via CDN) web UI with:
- Drag-and-drop PDF upload
- Live pipeline progress (SSE streaming per agent)
- Rendered survey output with dark mode
- No Node.js build step required
