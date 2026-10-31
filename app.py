"""Run:  uvicorn app:app --reload   then open http://127.0.0.1:8000"""
import json, re, shutil, threading, uuid
from contextlib import asynccontextmanager
from pathlib import Path

import chromadb
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from groq import Groq
from pydantic import BaseModel, Field
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

import config

BASE = Path(__file__).parent
WS_DIR = BASE / "workspaces"
ID_RE = re.compile(r"^[a-f0-9]{12}$")
NOT_FOUND = "I couldn't find that in your documents."
SYSTEM_PROMPT = f"""You are a helpful assistant that answers questions about the user's uploaded documents.
Answer using ONLY the provided context.
If the answer is not in the context, reply exactly: "{NOT_FOUND}" and do not cite any source.
Be clear and concise. Mention the file name (and page, if given) you used."""

state, jobs = {}, {}
embed_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not config.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is missing. Copy .env.example to .env and add your key.")
    WS_DIR.mkdir(exist_ok=True)
    state["db"] = chromadb.PersistentClient(path=str(BASE / "chroma_db"))
    state["model"] = SentenceTransformer(config.EMBED_MODEL)
    state["groq"] = Groq(api_key=config.GROQ_API_KEY)
    yield


app = FastAPI(title="Ask Your Documents", lifespan=lifespan)


# ---------- reading + chunking ----------
def clean(text):
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def read_file(path: Path):
    """Yield (page, text). page is 0 for formats without pages."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        for i, p in enumerate(PdfReader(str(path)).pages, start=1):
            t = clean(p.extract_text() or "")
            if t:
                yield i, t
    elif ext == ".docx":
        from docx import Document
        t = clean("\n".join(p.text for p in Document(str(path)).paragraphs))
        if t:
            yield 0, t
    else:
        t = clean(path.read_text(encoding="utf-8", errors="replace"))
        if t:
            yield 0, t


def chunk(source, page, text):
    start = 0
    while start < len(text):
        piece = text[start:start + config.CHUNK_SIZE].strip()
        if piece:
            yield {"id": f"{source}::p{page}::{start}", "text": piece, "source": source, "page": page}
        start += config.CHUNK_SIZE - config.OVERLAP


# ---------- workspace state ----------
def job_for(ws: str):
    if not ID_RE.match(ws):
        raise HTTPException(404, "Workspace not found. Upload your files again.")
    if ws in jobs:
        return jobs[ws]
    meta_path = WS_DIR / ws / "meta.json"
    if not meta_path.exists():
        raise HTTPException(404, "Workspace not found. Upload your files again.")
    jobs[ws] = {"state": "ready", "stage": "Ready", "progress": 1.0, "error": None,
                "warning": None, "skipped": [], **json.loads(meta_path.read_text())}
    return jobs[ws]


def public(job):
    keys = ("state", "stage", "progress", "error", "warning", "skipped", "files", "chunks")
    return {k: job[k] for k in keys}


def ingest_job(ws: str, names: list):
    job, up = jobs[ws], WS_DIR / ws / "uploads"
    had_data = job["chunks"] > 0
    try:
        chunks, skipped = [], []
        for n, name in enumerate(names):
            job.update(stage=f"Reading {name}", progress=0.05 + 0.25 * n / len(names))
            before = len(chunks)
            try:
                for page, text in read_file(up / name):
                    chunks += list(chunk(name, page, text))
            except Exception:
                skipped.append(f"{name}: could not be read")
                continue
            if len(chunks) == before:
                skipped.append(f"{name}: no readable text (scanned or image-only?)")
        if not chunks:
            raise ValueError("No readable text found in the uploaded files. "
                             "Scanned or image-only PDFs are not supported.")

        col = state["db"].get_or_create_collection(f"ws_{ws}", metadata={"hnsw:space": "cosine"})
        total = len(chunks)
        for i in range(0, total, 64):
            b = chunks[i:i + 64]
            job.update(stage=f"Indexing passages ({min(i + 64, total)} of {total})",
                       progress=0.3 + 0.65 * i / total)
            with embed_lock:
                emb = state["model"].encode([c["text"] for c in b], normalize_embeddings=True,
                                            convert_to_numpy=True)
            col.upsert(ids=[c["id"] for c in b], documents=[c["text"] for c in b],
                       embeddings=emb.tolist(),
                       metadatas=[{"source": c["source"], "page": c["page"]} for c in b])

        files = list(dict.fromkeys(job["files"] + [c["source"] for c in chunks]))
        meta = {"files": files, "chunks": col.count()}
        (WS_DIR / ws / "meta.json").write_text(json.dumps(meta))
        job.update(state="ready", stage="Ready", progress=1.0, skipped=skipped, **meta)
    except Exception as e:
        if had_data:
            job.update(state="ready", stage="Ready", progress=1.0, warning=str(e))
        else:
            job.update(state="error", error=str(e))
    finally:
        shutil.rmtree(up, ignore_errors=True)


# ---------- API ----------
@app.post("/api/upload")
async def upload(background: BackgroundTasks, files: list[UploadFile] = File(...),
                 workspace: str | None = Form(None)):
    if not files or len(files) > config.MAX_FILES:
        raise HTTPException(400, f"Upload between 1 and {config.MAX_FILES} files at a time.")
    ws = workspace or uuid.uuid4().hex[:12]
    job = job_for(ws) if workspace else None
    if job and job["state"] == "processing":
        raise HTTPException(409, "Your files are still being indexed. Wait until it finishes.")

    incoming = []
    for f in files:
        name = Path(f.filename or "").name
        if Path(name).suffix.lower() not in config.ALLOWED:
            raise HTTPException(400, f"{name or 'File'}: unsupported type. "
                                     f"Use {', '.join(sorted(config.ALLOWED))}.")
        data = await f.read()
        if not data:
            raise HTTPException(400, f"{name} is empty.")
        if len(data) > config.MAX_MB * 1_000_000:
            raise HTTPException(413, f"{name} is larger than {config.MAX_MB} MB.")
        incoming.append((name, data))

    up = WS_DIR / ws / "uploads"
    up.mkdir(parents=True, exist_ok=True)
    for name, data in incoming:
        (up / name).write_bytes(data)

    if not job:
        job = jobs[ws] = {"files": [], "chunks": 0}
    job.update(state="processing", stage="Starting", progress=0.02, error=None, warning=None, skipped=[])
    background.add_task(ingest_job, ws, [n for n, _ in incoming])
    return {"workspace": ws}


@app.get("/api/status/{ws}")
def status(ws: str):
    return public(job_for(ws))


@app.delete("/api/workspace/{ws}")
def delete_workspace(ws: str):
    if ID_RE.match(ws):
        try:
            state["db"].delete_collection(f"ws_{ws}")
        except Exception:
            pass
        shutil.rmtree(WS_DIR / ws, ignore_errors=True)
        jobs.pop(ws, None)
    return {"deleted": True}


class AskIn(BaseModel):
    workspace: str
    question: str = Field(min_length=2, max_length=1000)
    k: int = Field(5, ge=1, le=10)


@app.post("/api/ask")
def ask(body: AskIn):
    job = job_for(body.workspace)
    if job["state"] != "ready":
        raise HTTPException(409, "Your files are not ready yet.")
    question = body.question.strip()

    with embed_lock:
        q = state["model"].encode([question], normalize_embeddings=True, convert_to_numpy=True)
    col = state["db"].get_collection(f"ws_{body.workspace}")
    res = col.query(query_embeddings=q.tolist(), n_results=body.k)
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        if 1 - dist >= config.MIN_SCORE:
            hits.append({"score": round(1 - dist, 3), "text": doc, **meta})
    if not hits:
        return {"answer": NOT_FOUND, "sources": []}

    context = "\n\n".join(
        f"[{h['source']}{' p' + str(h['page']) if h['page'] else ''}]\n{h['text']}" for h in hits)
    try:
        resp = state["groq"].chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
            temperature=0.2, max_tokens=2000)
    except Exception as e:
        raise HTTPException(502, f"Groq request failed: {e}")

    answer = (resp.choices[0].message.content or "").strip()
    if not answer:
        raise HTTPException(502, "The model returned an empty answer. Try again.")
    if answer.startswith(NOT_FOUND):
        return {"answer": NOT_FOUND, "sources": []}

    seen, sources = set(), []
    for h in hits:
        key = (h["source"], h["page"])
        if key not in seen:
            seen.add(key)
            sources.append({"source": h["source"], "page": h["page"], "score": h["score"],
                            "snippet": h["text"][:220]})
    return {"answer": answer, "sources": sources}


app.mount("/", StaticFiles(directory=BASE / "static", html=True), name="static")
