"""
api.py — FastAPI backend for autolit

Exposes four routes:
  GET  /api/papers              list indexed papers from manifest
  POST /api/ingest              upload a PDF, runs Phase 1 + 2 in background
  GET  /api/summaries/{paper_id} return Phase 2 JSON summary
  POST /api/survey/stream       SSE stream of Phase 3 pipeline events

Static files (the React frontend) are served from src/autolit/web/.
"""

from __future__ import annotations

import json
import queue
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import asyncio

# ---------------------------------------------------------------------------
# Paths — all relative to project root (set by __main__.py or run.py)
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")
PDF_DIR = DATA_DIR / "pdfs"
INDEX_DIR = DATA_DIR / "indexes"
SUMMARIES_DIR = Path("outputs") / "summaries"
SURVEYS_DIR = Path("outputs") / "surveys"
MANIFEST_PATH = DATA_DIR / "papers_manifest.json"
WEB_DIR = Path(__file__).parent / "web"

MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB

_manifest_lock = threading.RLock()
_ingest_queue: queue.Queue[Tuple[str, str]] = queue.Queue()
_ingest_worker_started = False
_ingest_worker_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


def _load_manifest() -> Dict[str, Any]:
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_manifest(manifest: Dict[str, Any]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _mark_paper(
    paper_id: str,
    status: str,
    error: Optional[str] = None,
    pdf_filename: Optional[str] = None,
) -> None:
    with _manifest_lock:
        manifest = _load_manifest()
        previous = manifest.get(paper_id, {})
        manifest[paper_id] = {
            "status": status,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "error": error,
            "pdf_filename": pdf_filename or previous.get("pdf_filename"),
        }
        _save_manifest(manifest)


def _ingest_paper(paper_id: str, pdf_filename: str) -> None:
    pdf_path = PDF_DIR / pdf_filename
    try:
        _mark_paper(paper_id, "processing", pdf_filename=pdf_filename)

        # Phase 1: build FAISS index
        from src.autolit.rag_query import build_index_for_pdf
        index_dir = INDEX_DIR / paper_id
        build_index_for_pdf(pdf_path, index_dir)

        # Phase 2: field-by-field extraction
        from src.autolit.extraction_agent import summarize_paper_structured
        summary = summarize_paper_structured(paper_id)

        # Persist summary JSON
        SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
        summary_path = SUMMARIES_DIR / f"{paper_id}.json"
        summary_path.write_text(
            json.dumps(summary.model_dump(), indent=2), encoding="utf-8"
        )

        _mark_paper(paper_id, "indexed", pdf_filename=pdf_filename)
    except Exception as exc:
        _mark_paper(paper_id, "error", error=str(exc), pdf_filename=pdf_filename)


def _ingest_worker() -> None:
    while True:
        paper_id, pdf_filename = _ingest_queue.get()
        try:
            _ingest_paper(paper_id, pdf_filename)
        finally:
            _ingest_queue.task_done()


def _ensure_ingest_worker() -> None:
    global _ingest_worker_started
    with _ingest_worker_lock:
        if _ingest_worker_started:
            return
        threading.Thread(target=_ingest_worker, daemon=True, name="autolit-ingest").start()
        _ingest_worker_started = True


# ---------------------------------------------------------------------------
# App startup — sync manifest with files already on disk
# ---------------------------------------------------------------------------

app = FastAPI(title="autolit API")


@app.on_event("startup")
async def _sync_manifest() -> None:
    """Populate the manifest from summaries on disk and reset orphaned processing entries."""
    _ensure_ingest_worker()
    manifest = _load_manifest()
    changed = False

    # Add any summaries on disk that aren't in the manifest
    if SUMMARIES_DIR.exists():
        for f in SUMMARIES_DIR.glob("*.json"):
            pid = f.stem
            if pid not in manifest:
                manifest[pid] = {
                    "status": "indexed",
                    "ingested_at": None,
                    "error": None,
                }
                changed = True

    # Reset stuck "processing" entries — threads don't survive restarts
    to_requeue: List[Tuple[str, str]] = []
    for pid, meta in list(manifest.items()):
        recoverable_interruption = (
            meta.get("status") == "error"
            and str(meta.get("error") or "").startswith("Ingestion interrupted")
        )
        if meta.get("status") not in {"queued", "processing"} and not recoverable_interruption:
            continue

        has_summary = (SUMMARIES_DIR / f"{pid}.json").exists()
        pdf_filename = meta.get("pdf_filename") or f"{pid}.pdf"
        pdf_path = PDF_DIR / pdf_filename
        if has_summary:
            manifest[pid] = {
                "status": "indexed",
                "ingested_at": meta.get("ingested_at"),
                "error": None,
                "pdf_filename": pdf_filename,
            }
        elif pdf_path.exists():
            manifest[pid] = {
                "status": "queued",
                "ingested_at": meta.get("ingested_at"),
                "error": None,
                "pdf_filename": pdf_filename,
            }
            to_requeue.append((pid, pdf_path.name))
        else:
            manifest[pid] = {
                "status": "error",
                "ingested_at": meta.get("ingested_at"),
                "error": "Ingestion interrupted and the uploaded PDF is missing.",
                "pdf_filename": pdf_filename,
            }
        changed = True

    for pid, meta in ():
        if meta.get("status") == "processing":
            has_summary = (SUMMARIES_DIR / f"{pid}.json").exists()
            manifest[pid] = {
                "status": "indexed" if has_summary else "error",
                "ingested_at": meta.get("ingested_at"),
                "error": None if has_summary else "Ingestion interrupted — please re-upload.",
            }
            changed = True

    if changed:
        _save_manifest(manifest)

    for item in to_requeue:
        _ingest_queue.put(item)


# ---------------------------------------------------------------------------
# GET /api/papers
# ---------------------------------------------------------------------------


@app.get("/api/papers")
async def get_papers() -> List[Dict[str, Any]]:
    """Return all known papers with their current status."""
    with _manifest_lock:
        manifest = _load_manifest()
    result = []
    for paper_id, meta in manifest.items():
        has_summary = (SUMMARIES_DIR / f"{paper_id}.json").exists()
        result.append(
            {
                "paper_id": paper_id,
                "status": meta.get("status", "unknown"),
                "has_summary": has_summary,
                "ingested_at": meta.get("ingested_at"),
                "error": meta.get("error"),
                "pdf_filename": meta.get("pdf_filename"),
            }
        )
    return result


# ---------------------------------------------------------------------------
# POST /api/ingest
# ---------------------------------------------------------------------------


@app.post("/api/ingest")
async def ingest_pdf(file: UploadFile = File(...)) -> Dict[str, str]:
    """Upload a PDF and kick off Phase 1 + 2 in a background thread."""
    filename = Path(file.filename or "").name
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted.")

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "PDF exceeds 100 MB limit.")

    paper_id = Path(filename).stem
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = PDF_DIR / filename
    pdf_path.write_bytes(content)

    _ensure_ingest_worker()
    _mark_paper(paper_id, "queued", pdf_filename=filename)
    _ingest_queue.put((paper_id, filename))
    return {"paper_id": paper_id, "status": "queued"}


# ---------------------------------------------------------------------------
# GET /api/summaries/{paper_id}
# ---------------------------------------------------------------------------


@app.get("/api/summaries/{paper_id}")
async def get_summary(paper_id: str) -> Dict[str, Any]:
    """Return the Phase 2 JSON summary for a paper."""
    manifest = _load_manifest()
    meta = manifest.get(paper_id)
    if meta is None:
        raise HTTPException(404, f"Unknown paper: {paper_id}")

    summary_path = SUMMARIES_DIR / f"{paper_id}.json"
    if not summary_path.exists():
        return {"status": meta.get("status", "processing"), "summary": None}

    return {
        "status": "indexed",
        "summary": json.loads(summary_path.read_text(encoding="utf-8")),
    }


# ---------------------------------------------------------------------------
# POST /api/survey/stream  (SSE)
# ---------------------------------------------------------------------------


class SurveyRequest(BaseModel):
    topic: str
    paper_ids: Optional[List[str]] = None
    top_k: int = 3
    model: str = "llama3"


@app.post("/api/survey/stream")
async def survey_stream(request: SurveyRequest) -> StreamingResponse:
    """
    Run the Phase 3 pipeline and stream progress as Server-Sent Events.

    Each event is a JSON object:
      {"type": "log"|"result"|"error",
       "agent": "pipeline"|"selector"|"comparator"|"writer",
       "message": str,
       "timestamp": ISO8601}

    The final success event has type="result" and includes "markdown".
    """
    if not request.topic.strip():
        raise HTTPException(400, "topic must not be empty.")

    queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _emit(type_: str, agent: str, message: str, **extra: Any) -> None:
        payload: Dict[str, Any] = {
            "type": type_,
            "agent": agent,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **extra,
        }
        loop.call_soon_threadsafe(queue.put_nowait, json.dumps(payload))

    def _progress(agent: str, message: str) -> None:
        _emit("log", agent, message)

    def _run() -> None:
        try:
            from src.autolit.phase3.pipeline import run_survey_pipeline

            result_path = run_survey_pipeline(
                topic=request.topic,
                summaries_dir=str(SUMMARIES_DIR),
                surveys_dir=str(SURVEYS_DIR),
                top_k=request.top_k,
                paper_ids=request.paper_ids,
                progress_callback=_progress,
            )

            markdown = Path(result_path).read_text(encoding="utf-8")
            _emit("result", "pipeline", "Survey complete.", markdown=markdown)
        except Exception as exc:
            _emit("error", "pipeline", str(exc))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

    threading.Thread(target=_run, daemon=True).start()

    async def _event_generator():
        event_id = 0
        while True:
            item = await queue.get()
            if item is None:
                break
            yield f"id: {event_id}\ndata: {item}\n\n"
            event_id += 1

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# Static files — React frontend (must be last)
# ---------------------------------------------------------------------------

if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="static")
