import streamlit as st
import os, re, json, time, hashlib, sqlite3, logging, warnings, tempfile, shutil, subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Any
from functools import lru_cache
from datetime import datetime

import requests
import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from pypdf import PdfReader
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.config import Settings as _ChromaSettings

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CHROMA_TELEMETRY", "False")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
logging.basicConfig(level=logging.WARNING)

CACHE_DIR = Path(".streamlit_rag_cache")
CACHE_DIR.mkdir(exist_ok=True)

EMBED_MODEL              = "BAAI/bge-large-en-v1.5"
CHUNK_SIZE               = 900
CHUNK_OVERLAP            = 200
TOP_K_BM25               = 12
TOP_K_CHROMA             = 12
RRF_K                    = 20
TOP_K_FINAL              = 20
CONFIDENCE_THRESHOLD     = 0.45
EVIDENCE_MATCH_THRESHOLD = 70
FALLBACK_TOKEN           = "Insufficient evidence found"
LLM_TEMPERATURE          = 0.0
LLM_NUM_PREDICT          = 2048
API_CALL_DELAY           = 2.0

SUBMISSION_COLUMNS = [
    "Filename", "Brand", "Age", "Step_Therapy_Requirements",
    "Number_of_Steps_through_Brands", "Number_of_Steps_through_Generic",
    "Step_Through_Phototherapy", "TB_Test_Required",
    "Initial_Authorization_Duration", "Reauthorization_Duration",
    "Reauthorization_Required", "Reauthorization_Requirements",
    "Specialist_Types", "Quantity_Limits", "Access_Score",
]

# ── Embedding model ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )

# ── LLM cache + call wrapper ──────────────────────────────────────────────────
@st.cache_resource
def get_llm_cache():
    db_path = CACHE_DIR / "llm.sqlite"
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS llm_cache "
        "(key TEXT PRIMARY KEY, response TEXT, created_at REAL)"
    )
    conn.commit()
    return conn

def _cache_key(model, system, prompt):
    h = hashlib.sha256()
    h.update(model.encode()); h.update(b"\x00")
    h.update((system or "").encode()); h.update(b"\x00")
    h.update(prompt.encode())
    return h.hexdigest()

def call_llm_json(prompt, system_prompt, api_key, model, max_tokens=None, retries=3):
    conn = get_llm_cache()
    key = _cache_key(model, system_prompt or "", prompt)
    row = conn.execute("SELECT response FROM llm_cache WHERE key=?", (key,)).fetchone()
    if row:
        return json.loads(row[0])
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "temperature": LLM_TEMPERATURE,
        "max_tokens": max_tokens or LLM_NUM_PREDICT,
        "top_p": 0.9,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for attempt in range(retries):
        try:
            time.sleep(API_CALL_DELAY)
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers, json=payload, timeout=600,
            )
            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"].strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            parsed = json.loads(raw)
            conn.execute(
                "INSERT OR REPLACE INTO llm_cache(key,response,created_at) VALUES(?,?,?)",
                (key, json.dumps(parsed), time.time()),
            )
            conn.commit()
            return parsed
        except json.JSONDecodeError:
            pass
        except Exception:
            time.sleep(max(2 ** attempt, API_CALL_DELAY))
    return None

# ── PDF extraction ────────────────────────────────────────────────────────────
@dataclass
class PDFDoc:
    filename: str
    pages: list
    full_text: str
    effective_date: Optional[str]
    page_count: int

_DATE_RE = re.compile(
    r"(?im)(effective|revision|revised|policy\s+date|approved|last\s+updated|"
    r"reviewed|next\s+review|date\s+of\s+origin|origination|implementation|"
    r"adopted|publish(?:ed)?|date)\s*[:\-]?\s*"
    r"((?:\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})|"
    r"(?:[A-Z][a-z]+\.?\s+\d{1,2},?\s+\d{4})|"
    r"(?:\d{4}-\d{2}-\d{2}))"
)

def _normalize_date(raw):
    raw = raw.strip().rstrip(",.;:")
    for fmt in ("%m/%d/%Y", "%m/%d/%y", "%m-%d-%Y", "%m-%d-%y",
                "%B %d, %Y", "%B %d %Y", "%b %d, %Y", "%b %d %Y"):
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    return None

def _detect_col_boundaries(text_blocks, page_width):
    if len(text_blocks) < 3:
        return []
    xs = sorted(set(round(b[0] / 10) * 10 for b in text_blocks))
    if len(xs) < 2:
        return []
    gaps = [(xs[i+1] - xs[i], xs[i], xs[i+1]) for i in range(len(xs) - 1)]
    gap_sizes = [g[0] for g in gaps]
    median_gap = sorted(gap_sizes)[len(gap_sizes) // 2]
    threshold = max(page_width * 0.08, median_gap * 2.5)
    boundaries = []
    for size, left, right in gaps:
        if size >= threshold:
            mid = left + size / 2
            if page_width * 0.05 < mid < page_width * 0.95:
                boundaries.append(mid)
    return boundaries

def _col_index(x, boundaries):
    for i, b in enumerate(boundaries):
        if x < b:
            return i
    return len(boundaries)

_CHECKBOX_CHECKED   = {"☑", "☒", "✔", "✓", "⧫", "■"}
_CHECKBOX_UNCHECKED = {"☐", "□", "◻", "⬜"}
_WINGDINGS_MAP = {
    "": "[ ]", "": "[ ]", "": "[ ]",
    "": "[X]", "": "[X]", "": "[X]", "": "[X]",
    "": "[ ]", "": "[X]",
}

def _extract_form_widgets(page):
    widgets = []
    try:
        for w in (page.widgets() or []):
            ft = getattr(w, "field_type_string", None)
            if ft not in ("CheckBox", "RadioButton"):
                continue
            val = w.field_value
            checked = bool(val) and str(val).strip().lower() not in ("off", "no", "false", "")
            state = "[X]" if checked else "[ ]"
            label = (getattr(w, "field_label", None) or w.field_name or "").replace("/", " ").strip()
            text = f"{state} {label}" if label else state
            r = w.rect
            widgets.append((r.x0, r.y0, r.x1, r.y1, text, -1, 0))
    except Exception:
        pass
    return widgets

def clean_text(raw: str) -> str:
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", " ", raw)
    for ch in _CHECKBOX_CHECKED:
        text = text.replace(ch, "[X]")
    for ch in _CHECKBOX_UNCHECKED:
        text = text.replace(ch, "[ ]")
    for ch, repl in _WINGDINGS_MAP.items():
        text = text.replace(ch, repl)
    for src, tgt in [
        ("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'),
        ("–", "-"), ("—", "-"), ("®", ""), ("•", "-"),
        ("≥", ">="), ("≤", "<="),
    ]:
        text = text.replace(src, tgt)
    text = re.sub(
        r"(?im)^(page\s+\d+(\s+of\s+\d+)?|confidential|proprietary|"
        r"copyright|all rights reserved)\s*$", "", text,
    )
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [re.sub(r"[ \t]{2,}", " ", ln).strip() for ln in text.splitlines()]
    return "\n".join(lines).strip()

def extract_pdf(pdf_bytes: bytes, filename: str) -> PDFDoc:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = Path(tmp.name)
    try:
        pages = []
        doc = fitz.open(str(tmp_path))
        try:
            for page in doc:
                blocks = page.get_text("blocks") or []
                text_blocks = [b for b in blocks if len(b) >= 7 and b[6] == 0 and b[4].strip()]
                text_blocks += _extract_form_widgets(page)
                if not text_blocks:
                    pages.append(page.get_text("text") or "")
                    continue
                page_width = page.rect.width or 612.0
                boundaries = _detect_col_boundaries(text_blocks, page_width)
                if boundaries:
                    text_blocks.sort(key=lambda b: (_col_index(b[0], boundaries), b[1], b[0]))
                else:
                    text_blocks.sort(key=lambda b: (b[1], b[0]))
                pages.append("\n".join(b[4].strip() for b in text_blocks))
        finally:
            doc.close()
    except Exception:
        reader = PdfReader(str(tmp_path))
        pages = [(p.extract_text() or "") for p in reader.pages]
    finally:
        tmp_path.unlink(missing_ok=True)

    full_text = "\n\n".join(f"[PAGE {i+1}]\n{t}" for i, t in enumerate(pages))
    window = (pages[0][:1500] if pages else "") + "\n" + (pages[-1][-1500:] if pages else "")
    candidates = [_normalize_date(raw) for _, raw in _DATE_RE.findall(window)]
    candidates = [d for d in candidates if d]
    return PDFDoc(
        filename=filename, pages=pages, full_text=full_text,
        effective_date=max(candidates) if candidates else None,
        page_count=len(pages),
    )

# ── Chunking ──────────────────────────────────────────────────────────────────
@dataclass
class Chunk:
    idx: int
    text: str
    page: int
    rel_position: float

    def __hash__(self):
        return hash((self.idx, self.page))

_PAGE_HEAD_RE = re.compile(r"\[PAGE (\d+)\]")

def chunk_document(cleaned: str, page_count: int) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", ", ", " "],
    )
    raw_chunks = splitter.split_text(cleaned)
    chunks, total, running_offset, current_page = [], len(cleaned) or 1, 0, 1
    for i, txt in enumerate(raw_chunks):
        m = _PAGE_HEAD_RE.search(txt)
        if m:
            current_page = int(m.group(1))
        running_offset = cleaned.find(txt, running_offset)
        rel = running_offset / total if running_offset >= 0 else i / max(len(raw_chunks), 1)
        chunks.append(Chunk(idx=i, text=txt, page=current_page, rel_position=rel))
        running_offset = max(running_offset, 0) + len(txt)
    return chunks

# ── Retrieval ─────────────────────────────────────────────────────────────────
_TOKEN_RE = re.compile(r"[a-z0-9]+")

def _tokenize(text):
    return _TOKEN_RE.findall(text.lower())

def build_bm25_index(chunks):
    return BM25Okapi([_tokenize(c.text) for c in chunks])

def build_chroma_collection(chunks, embed_model):
    client = chromadb.EphemeralClient(settings=_ChromaSettings(anonymized_telemetry=False))
    coll = client.create_collection(name="doc", metadata={"hnsw:space": "cosine"})
    texts = [c.text for c in chunks]
    embeddings = embed_model.embed_documents(texts)
    coll.add(
        ids=[str(c.idx) for c in chunks],
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"idx": c.idx, "page": c.page, "rel_position": c.rel_position} for c in chunks],
    )
    return coll, client

@dataclass
class RetrievalResult:
    chunks: list
    bm25_ranks: dict
    chroma_ranks: dict
    rrf_scores: dict

    @property
    def faiss_ranks(self):
        return self.chroma_ranks

def hybrid_retrieve(query, chunks, bm25, chroma_coll, embed_model):
    scores = bm25.get_scores(_tokenize(query))
    bm25_order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:TOP_K_BM25]
    bm25_ranks = {idx: r + 1 for r, idx in enumerate(bm25_order)}

    q_emb = embed_model.embed_query(query)
    res = chroma_coll.query(
        query_embeddings=[q_emb],
        n_results=min(TOP_K_CHROMA, len(chunks)),
        include=["metadatas"],
    )
    chroma_order = [int(m["idx"]) for m in (res["metadatas"][0] if res["metadatas"] else [])]
    chroma_ranks = {idx: r + 1 for r, idx in enumerate(chroma_order)}

    rrf: dict = {}
    for rank, idx in enumerate(bm25_order, 1):
        rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (RRF_K + rank)
    for rank, idx in enumerate(chroma_order, 1):
        rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (RRF_K + rank)

    fused = sorted(rrf.keys(), key=lambda i: rrf[i], reverse=True)[:TOP_K_FINAL]
    return RetrievalResult(
        chunks=[chunks[i] for i in fused],
        bm25_ranks=bm25_ranks,
        chroma_ranks=chroma_ranks,
        rrf_scores=rrf,
    )

# ── Source confidence ─────────────────────────────────────────────────────────
def compute_freshness_score(effective_date):
    if not effective_date:
        return 0.7
    try:
        d = datetime.fromisoformat(effective_date)
        years = max(0.0, (datetime.now() - d).days / 365.25)
        return float(max(0.05, min(1.0, np.exp(-years / 3.0))))
    except ValueError:
        return 0.7

def compute_trust_score(result, page_count):
    if not result.chunks:
        return 0.0
    positions = np.array([c.rel_position for c in result.chunks])
    body_mass = float(np.mean((positions > 0.05) & (positions < 0.85)))
    page_spread = len({c.page for c in result.chunks}) / max(1, min(page_count, len(result.chunks)))
    return float(0.6 * body_mass + 0.4 * min(1.0, page_spread))

def compute_retrieval_consistency(result, top_k=10):
    bm25_top = {i for i, r in result.bm25_ranks.items() if r <= top_k}
    faiss_top = {i for i, r in result.faiss_ranks.items() if r <= top_k}
    if not bm25_top or not faiss_top:
        return 0.0
    jacc = len(bm25_top & faiss_top) / len(bm25_top | faiss_top)
    both = sum(1 for c in result.chunks if c.idx in result.bm25_ranks and c.idx in result.faiss_ranks)
    return float(0.5 * jacc + 0.5 * both / max(1, len(result.chunks)))

def aggregate_source_confidence(freshness, trust, consistency):
    eps = 1e-3
    return float(np.exp(
        0.10 * np.log(max(freshness, 0.5) + eps)
        + 0.45 * np.log(max(trust, 0.3) + eps)
        + 0.45 * np.log(max(consistency, 0.3) + eps)
    ))

# ── Brand detection ───────────────────────────────────────────────────────────
TARGET_BRANDS = {"TREMFYA", "STELARA"}
BRAND_ALIASES = {
    "TREMFYA": ["TREMFYA", "GUSELKUMAB"],
    "STELARA": ["STELARA", "USTEKINUMAB"],
}
BRAND_GENERIC = {"TREMFYA": "guselkumab", "STELARA": "ustekinumab"}

BRAND_DETECTION_SYSTEM = (
    "You are a medical policy parser. Use ONLY the provided text. "
    "Do NOT use external knowledge. Output strict JSON only."
)

INDICATION_CHECK_PROMPT = """Does the PA policy text below have its OWN approval criteria
(not just mentioned as a step-therapy alternative) for {brand} ({generic})
under the PSORIASIS (PsO) indication?

Reply with ONLY: {{"pso": true}} or {{"pso": false}}

TEXT:
{text}"""

PSA_CHECK_PROMPT = """Does the PA policy text below have its OWN approval criteria
(not just mentioned as a step-therapy alternative) for {brand} ({generic})
under the PSORIATIC ARTHRITIS (PsA) indication?

Reply with ONLY: {{"psa": true}} or {{"psa": false}}

TEXT:
{text}"""

DISCOVER_BRANDS_PROMPT = """You are reading a Prior Authorization (PA) policy document.
List every biologic or specialty drug brand name that has its OWN distinct approval
criteria in this policy for PSORIASIS (PsO) or PSORIATIC ARTHRITIS (PsA).
Do NOT include drugs mentioned only as step-therapy alternatives or exclusions.

Return ONLY valid JSON in this exact format:
{{"brands": ["BRAND_A", "BRAND_B"]}}

If none are found, return: {{"brands": []}}

TEXT:
{text}"""

PA_RELEVANCE_PROMPT = """Does the PA policy text below contain actual Prior Authorization (PA)
approval criteria specifically for {brand}?

Answer YES if the text includes ANY of:
- Clinical eligibility criteria or diagnosis requirements for {brand}
- Step therapy requirements that must be met before {brand} is approved
- Authorization duration or renewal criteria for {brand}
- Prescriber or specialist requirements for {brand}
- Quantity limits or dosing restrictions for {brand}
- Lab test or safety screening requirements for {brand}

Answer NO if {brand} appears ONLY in:
- An "applicable drug list", "covered drug list", or formulary table with no PA criteria
- A step-therapy alternative list (mentioned only as a drug another patient must try first)
- An exclusion list or non-covered drugs section
- An incidental mention with no coverage criteria attached

Reply with ONLY: {{"pa_relevant": true}} or {{"pa_relevant": false}}

TEXT:
{text}"""

def _brands_in_text(text):
    text_upper = text.upper()
    return {brand for brand, aliases in BRAND_ALIASES.items()
            if any(alias in text_upper for alias in aliases)}

def _check_brand_indication(brand, generic, sample, api_key, model):
    result = call_llm_json(
        INDICATION_CHECK_PROMPT.format(brand=brand, generic=generic, text=sample),
        BRAND_DETECTION_SYSTEM, api_key, model, max_tokens=32,
    )
    if result and result.get("pso"):
        return "PSO"
    result_psa = call_llm_json(
        PSA_CHECK_PROMPT.format(brand=brand, generic=generic, text=sample),
        BRAND_DETECTION_SYSTEM, api_key, model, max_tokens=32,
    )
    if result_psa and result_psa.get("psa"):
        return "PSA"
    return None

def _validate_pa_relevance(brands, cleaned, api_key, model):
    if not brands:
        return brands
    sample = cleaned[:20000]
    validated = []
    for brand in brands:
        result = call_llm_json(
            PA_RELEVANCE_PROMPT.format(brand=brand, text=sample),
            BRAND_DETECTION_SYSTEM, api_key, model, max_tokens=32,
        )
        if result is None or result.get("pa_relevant"):
            validated.append(brand)
    return validated or brands

def detect_brands(cleaned, api_key, model):
    present = _brands_in_text(cleaned)
    sample = cleaned[:15000]

    if not present:
        result = call_llm_json(
            DISCOVER_BRANDS_PROMPT.format(text=sample),
            BRAND_DETECTION_SYSTEM, api_key, model, max_tokens=256,
        )
        discovered = [str(b).strip().upper() for b in (result or {}).get("brands", []) if b]
        if not discovered:
            return []
        by_ind: dict = {"PSO": set(), "PSA": set()}
        for brand in discovered:
            ind = _check_brand_indication(brand, brand.lower(), sample, api_key, model)
            by_ind[ind or "PSO"].add(brand)
        pso = sorted(by_ind["PSO"])
        result_brands = pso or sorted(by_ind["PSA"]) or discovered
        return _validate_pa_relevance(result_brands, cleaned, api_key, model)

    by_ind = {"PSO": set(), "PSA": set()}
    for brand in sorted(present):
        generic = BRAND_GENERIC[brand]
        ind = _check_brand_indication(brand, generic, sample, api_key, model)
        if ind:
            by_ind[ind].add(brand)
            continue
        brand_pos = cleaned.upper().find(brand)
        if brand_pos >= 0:
            window = cleaned[max(0, brand_pos - 300): brand_pos + 2000].upper()
            if any(kw in window for kw in ["PSORIASIS", "PLAQUE", " PSO", "PSOR"]):
                by_ind["PSO"].add(brand)
            else:
                by_ind["PSA"].add(brand)

    pso = sorted(by_ind["PSO"])
    result_brands = pso or sorted(by_ind["PSA"]) or sorted(present)
    return _validate_pa_relevance(result_brands, cleaned, api_key, model)

# ── Parameters ────────────────────────────────────────────────────────────────
PARAMETERS = {
    "Age": {
        "query": "{brand} age eligibility minimum age requirement years old pediatric adult adolescent threshold FDA labelled",
        "definition": (
            "Age eligibility criteria for {brand} for PsO. If no numerical threshold but refers to "
            "FDA-indicated use, output 'FDA labelled age'. OUTPUT FORMAT: '>=18', '>6 years', '>=12', "
            "'FDA labelled age'. Capture YOUNGEST if two age groups listed. NOT_FOUND if not mentioned."
        ),
        "null_default": "Not specified",
    },
    "Step_Therapy_Requirements": {
        "query": "{brand} step therapy prior failure trial inadequate response intolerance contraindication conventional systemic biologic universal criteria all indications",
        "definition": (
            "ALL step therapy language for {brand} for PsO — both brand-specific AND universal criteria. "
            "Include phototherapy/PUVA if within step statements. Moderate-to-severe PsO only. "
            "Write each requirement as a numbered step on its own line. NOT_FOUND if absent."
        ),
        "null_default": "None required",
    },
    "Number_of_Steps_through_Brands": {
        "query": "{brand} branded biologic biosimilar TNF IL-17 IL-23 preferred adalimumab ustekinumab etanercept secukinumab ixekizumab tried failed step prior drug class",
        "definition": (
            "Count BRANDED/BIOLOGIC step therapy steps before {brand} can be approved for PsO. "
            "Union of universal + brand-specific (AND logic). OR statements: take fewer steps path. "
            "Exclude phototherapy. Output a single number or 'NA'."
        ),
        "null_default": "NA",
    },
    "Number_of_Steps_through_Generic": {
        "query": "{brand} topical corticosteroid methotrexate cyclosporine acitretin retinoid apremilast non-biologic conventional systemic generic step prior",
        "definition": (
            "Count GENERIC/NON-BIOLOGIC step therapy steps before {brand} for PsO. Topicals count as generic. "
            "Same union AND/OR logic. Exclude phototherapy. Output a single number or 'NA'."
        ),
        "null_default": "NA",
    },
    "Step_Through_Phototherapy": {
        "query": "{brand} phototherapy PUVA UVB narrow-band light therapy psoralen ultraviolet step prior failure required mandatory",
        "definition": (
            "Is phototherapy (including PUVA) a mandatory step before {brand} for PsO? "
            "'Yes' if mandatory and not in OR statement. 'No' if not required. 'N/A' if no criteria at all."
        ),
        "null_default": "No",
    },
    "TB_Test_Required": {
        "query": "{brand} tuberculosis TB test required screening baseline PPD tuberculin skin test TST interferon-gamma release assay IGRA QuantiFERON-TB Gold latent active TB prior to initiating pre-treatment safety screening",
        "definition": (
            "Is a TB test required before {brand} for PsO? TB testing may be called: TB test, PPD, "
            "tuberculin skin test (TST), IGRA, QuantiFERON-TB Gold, latent tuberculosis screening, LTBI test. "
            "'Yes' = required. 'No' = explicitly not required. 'Not specified' = not mentioned."
        ),
        "null_default": "Not specified",
    },
    "Initial_Authorization_Duration": {
        "query": "{brand} initial authorization duration period months coverage approved first PA approval length granted",
        "definition": (
            "Initial PA coverage duration for {brand} for PsO. "
            "Output '6 Months', '12 Months' (include unit). 'Unspecified' if PA required but duration not stated."
        ),
        "null_default": "Unspecified",
    },
    "Reauthorization_Duration": {
        "query": "{brand} reauthorization renewal continuation re-approval subsequent months duration after initial period reassessment",
        "definition": (
            "Reauthorization duration for {brand} for PsO. "
            "Output '12 Months', '6 Months' (include unit). 'Unspecified' if required but not stated. 'NA' if not required."
        ),
        "null_default": "Unspecified",
    },
    "Reauthorization_Required": {
        "query": "{brand} reauthorization required renewal continued coverage reassessment documentation after initial authorization",
        "definition": (
            "Is reauthorization required for {brand} for PsO? "
            "RULE: if Reauthorization Duration OR Requirements are present, this MUST be 'Yes'. "
            "'No' = explicitly not needed. 'Not specified' = no reauth language found."
        ),
        "null_default": "Not specified",
    },
    "Reauthorization_Requirements": {
        "query": "{brand} reauthorization continued clinical benefit response improvement PASI BSA stable maintained therapy continuation criteria documentation lab values physician attestation",
        "definition": (
            "Explicit continuation criteria for {brand} for PsO reauthorization. "
            "Write each criterion as a numbered point. NOT_FOUND if not specified."
        ),
        "null_default": "Not specified",
    },
    "Specialist_Types": {
        "query": "{brand} prescriber specialist dermatologist rheumatologist gastroenterologist physician prescribed by managed by initiating specialist restriction required",
        "definition": (
            "Medical specialties explicitly required to prescribe {brand} for PsO. "
            "List as comma-separated values, e.g. 'Dermatologist, Rheumatologist'. NOT_FOUND if no restriction."
        ),
        "null_default": "No restriction specified",
    },
    "Quantity_Limits": {
        "query": "{brand} quantity limit units per month supply allowable maximum vials syringes pens dispense days supply dispensing frequency pack size",
        "definition": (
            "Describe the quantity limit for {brand} for PsO in plain English. Include dosage form, "
            "numeric quantity, and time period. E.g. '2 syringes (100 mg/mL) per 28 days', "
            "'1 carton of 2 pens per month'. Look under headings: quantity limit, dispensing limit, supply limit. "
            "If no restriction exists, output 'No quantity limit specified'."
        ),
        "null_default": "No quantity limit specified",
    },
}

# ── Extraction ────────────────────────────────────────────────────────────────
EXTRACTION_SYSTEM = """You are an expert pharmaceutical policy extraction assistant.

A Payer Prior Authorization (PA) policy is a document issued by a health insurance payer
that defines the clinical criteria a patient must meet before a specific drug can be covered.
These policies govern which patients qualify for a drug, what prerequisite treatments they
must have tried first (step therapy), how long the approval lasts, and what documentation
a prescriber must provide.

Your task is to extract structured information from retrieved text chunks of these payer PA
policy documents for a specific drug brand.

EXTRACTION RULES:
1. Extract ONLY from the document context provided. NEVER use your training knowledge.
2. If a field is not explicitly present in the context, return "NOT_FOUND" with confidence 0.0.
3. Do NOT infer, assume, or extrapolate beyond what is written.
4. Respond with ONLY a valid JSON object. No markdown, no explanation, no prose.
5. Evidence MUST be a verbatim phrase copied from the document, under 30 words.
6. TB MAPPING: "latent tuberculosis screening", "TB screening", "tuberculosis test",
   "LTBI test", "QuantiFERON", "tuberculin skin test" all map to TB_Test_Required = "Yes".
7. STEP THERAPY: mark Yes only if the policy explicitly requires prior treatment failure.
8. QUANTITY LIMITS: describe the actual restriction in plain English with form + quantity + frequency.
9. SPECIALIST TYPES: list comma-separated. No restriction → NOT_FOUND.
10. AUTHORIZATION DURATIONS: in months only. Convert weeks/years if needed.

OUTPUT FORMAT:
- Plain readable English values — no codes, no abbreviations.
- Multiple items: comma-separated.
- Ordered steps or criteria: numbered lines.
- Yes/No/Not specified fields: exact token only.
- Age: inequality format (>=18, FDA labelled age).
- Step counts: plain number (1, 2, NA).
- Durations: X Months format (12 Months, 6 Months, Unspecified).
"""

EXTRACTION_PROMPT = """DOCUMENT CONTEXT (retrieved from payer PA policy):
=====================================================
{context}
=====================================================

Drug Brand: {brand}

PARAMETERS:
{param_block}

Respond with EXACTLY this JSON shape:
{{
  "<Parameter_Name>": {{
    "value": "<extracted value or NOT_FOUND>",
    "confidence": <float 0.0-1.0>,
    "evidence": "<verbatim phrase from document or null>"
  }}
}}

Confidence: 1.0 explicit, 0.7 implied, 0.4 ambiguous, 0.0 not found."""

def build_retrieval_context(brand, chunks, bm25, chroma_coll, embed_model):
    seen, merged, per_param = set(), [], {}
    for param_name, cfg in PARAMETERS.items():
        query = cfg["query"].format(brand=brand)
        result = hybrid_retrieve(query, chunks, bm25, chroma_coll, embed_model)
        per_param[param_name] = result
        for c in result.chunks[:5]:
            if c.idx in seen:
                continue
            seen.add(c.idx)
            merged.append(c)
            if len(merged) >= 36:
                break
        if len(merged) >= 36:
            break
    context = "\n\n---\n\n".join(f"[chunk {c.idx} | page {c.page}]\n{c.text}" for c in merged)
    return context, per_param

def extract_all_parameters(brand, chunks, bm25, chroma_coll, source_conf, embed_model, api_key, model):
    context, per_param = build_retrieval_context(brand, chunks, bm25, chroma_coll, embed_model)
    param_lines = "\n".join(
        f"- {name}: {cfg['definition'].format(brand=brand)}"
        for name, cfg in PARAMETERS.items()
    )
    prompt = EXTRACTION_PROMPT.format(context=context, brand=brand, param_block=param_lines)
    raw = call_llm_json(prompt, EXTRACTION_SYSTEM, api_key, model) or {}

    out = {}
    for param_name in PARAMETERS:
        entry = raw.get(param_name) or {}
        value = str(entry.get("value", "NOT_FOUND")).strip()
        conf  = float(entry.get("confidence", 0.0) or 0.0)
        evid  = str(entry.get("evidence") or "").strip()
        verified = bool(evid and fuzz.partial_ratio(evid.lower(), context.lower()) >= EVIDENCE_MATCH_THRESHOLD)
        eff_conf = conf * (0.5 + 0.5 * source_conf)
        if value == "NOT_FOUND":
            value_out = PARAMETERS[param_name]["null_default"]
        elif conf < CONFIDENCE_THRESHOLD:
            value_out = FALLBACK_TOKEN
        elif evid and not verified:
            value_out = FALLBACK_TOKEN
        else:
            value_out = value
        out[param_name] = {"value": value_out, "confidence": conf, "effective_confidence": eff_conf, "evidence": evid}
    return out, per_param

# ── Business rules ────────────────────────────────────────────────────────────
def apply_business_rules(row):
    dur  = str(row.get("Reauthorization_Duration", ""))
    reqs = str(row.get("Reauthorization_Requirements", ""))
    req  = str(row.get("Reauthorization_Required", ""))
    placeholders = {"NA", "Not specified", "Unspecified", "NOT_FOUND", "", FALLBACK_TOKEN, "None required"}
    if dur not in placeholders or reqs not in placeholders:
        row["Reauthorization_Required"] = "Yes"
    elif req.strip().lower() == "yes":
        row["Reauthorization_Required"] = FALLBACK_TOKEN
    if row.get("Initial_Authorization_Duration", "") in {"Not specified", "NOT_FOUND", "", FALLBACK_TOKEN}:
        row["Initial_Authorization_Duration"] = "Unspecified"
    ql = str(row.get("Quantity_Limits", ""))
    if re.search(r"\b(dosage|dosing limit|dose)\b", ql, re.IGNORECASE):
        row["Quantity_Limits"] = "No quantity limit specified"
    return row

# ── Access score ──────────────────────────────────────────────────────────────
def compute_access_score(row):
    def age(v):
        v = str(v).lower()
        if "insufficient" in v or "not specified" in v: return 0.0
        if "fda labelled" in v: return 12.0
        m = re.search(r"(\d+)", v)
        if not m: return 10.0
        a = int(m.group(1))
        return 15.0 if a<=18 else 13.0 if a<=21 else 10.0 if a<=30 else 6.0 if a<=50 else 3.0
    def steps(bv, gv):
        def parse(x):
            s = str(x).strip().upper()
            if s in ("NA", "INSUFFICIENT EVIDENCE FOUND"): return 0
            m = re.search(r"\d+", s); return int(m.group()) if m else 0
        t = parse(bv) + parse(gv)
        return {0:35.0,1:28.0,2:20.0,3:12.0,4:7.0}.get(t, 3.0)
    def photo(v):
        return 0.0 if str(v).strip().lower() == "yes" or "insufficient" in str(v).lower() else 10.0
    def auth(v):
        v = str(v).strip().lower()
        if "insufficient" in v: return 0.0
        if v in ("unspecified", "not specified"): return 6.0
        m = re.search(r"(\d+)", v)
        if not m: return 10.0
        mo = int(m.group(1))
        return 15.0 if mo>=12 else 8.0 if mo>=6 else 4.0
    def tb(v):
        v = str(v).strip().upper()
        if "INSUFFICIENT" in v: return 0.0
        return 0.0 if v.startswith("Y") else 5.0 if v.startswith("N") else 2.0
    def reauth(req, dur):
        if str(req).strip().lower() == "no": return 10.0
        m = re.search(r"(\d+)", str(dur))
        if not m: return 5.0
        mo = int(m.group(1))
        return 7.0 if mo>=12 else 4.0 if mo>=6 else 2.0
    def specialist(v):
        s = str(v).lower()
        if "insufficient" in s: return 0.0
        return 10.0 if any(x in s for x in ("no restriction", "not specified")) else 5.0

    score = (
        age(row.get("Age", "Not specified"))
        + steps(row.get("Number_of_Steps_through_Brands", "NA"), row.get("Number_of_Steps_through_Generic", "NA"))
        + photo(row.get("Step_Through_Phototherapy", "No"))
        + auth(row.get("Initial_Authorization_Duration", "Unspecified"))
        + tb(row.get("TB_Test_Required", "Not specified"))
        + reauth(row.get("Reauthorization_Required", "Not specified"), row.get("Reauthorization_Duration", "Unspecified"))
        + specialist(row.get("Specialist_Types", "No restriction specified"))
    )
    return round(min(score, 100.0), 1)

# ── Process one PDF ───────────────────────────────────────────────────────────
def process_one_pdf(pdf_bytes, filename, api_key, model, embed_model, status):
    status.info(f"Extracting text from **{filename}**…")
    doc = extract_pdf(pdf_bytes, filename)
    cleaned = clean_text(doc.full_text)
    if not cleaned:
        return []

    status.info(f"Detecting brands in **{filename}**…")
    brands = detect_brands(cleaned, api_key, model)
    if not brands:
        return []

    status.info(f"Building index for **{filename}**…")
    chunks = chunk_document(cleaned, doc.page_count)
    if not chunks:
        return []

    bm25 = build_bm25_index(chunks)
    chroma_coll, _client = build_chroma_collection(chunks, embed_model)
    freshness = compute_freshness_score(doc.effective_date)

    rows = []
    for brand in brands:
        status.info(f"Extracting **{brand}** from **{filename}**…")
        extracted_prelim, per_param = extract_all_parameters(
            brand, chunks, bm25, chroma_coll, 0.5, embed_model, api_key, model,
        )
        trust = float(np.mean([compute_trust_score(r, doc.page_count) for r in per_param.values()])) if per_param else 0.5
        consistency = float(np.mean([compute_retrieval_consistency(r) for r in per_param.values()])) if per_param else 0.5
        source_conf = aggregate_source_confidence(freshness, trust, consistency)

        extracted, per_param = extract_all_parameters(
            brand, chunks, bm25, chroma_coll, source_conf, embed_model, api_key, model,
        )
        row = {"Filename": filename, "Brand": brand}
        for param_name, entry in extracted.items():
            row[param_name] = entry["value"]
        row = apply_business_rules(row)
        row["Access_Score"] = compute_access_score(row)
        row["_details"] = extracted
        rows.append(row)
    return rows

# ── UI helpers ────────────────────────────────────────────────────────────────
def _score_color(score):
    if score >= 75:
        return "green"
    if score >= 50:
        return "orange"
    return "red"

def _score_label(score):
    if score >= 90: return "Best-in-class access"
    if score >= 75: return "Preferred / open access"
    if score >= 50: return "Roughly parity with FDA label"
    if score >= 25: return "Restricted vs FDA label"
    return "Heavily restricted"

def _render_brand_card(row, details):
    score = row.get("Access_Score", 0)
    color = _score_color(score)
    st.markdown(
        f"<h4 style='margin-bottom:4px'>{row['Brand']} &nbsp;"
        f"<span style='color:{color};font-size:1.1em'>{score}/100</span> "
        f"<span style='font-size:0.8em;color:gray'>— {_score_label(score)}</span></h4>",
        unsafe_allow_html=True,
    )

    display_fields = [f for f in SUBMISSION_COLUMNS if f not in ("Filename", "Brand", "Access_Score")]
    col1, col2 = st.columns(2)
    for i, field in enumerate(display_fields):
        val = row.get(field, "—")
        conf = details.get(field, {}).get("effective_confidence", None) if details else None
        conf_str = f"  *(conf: {conf:.2f})*" if conf is not None else ""
        target_col = col1 if i % 2 == 0 else col2
        with target_col:
            label = field.replace("_", " ")
            st.markdown(f"**{label}**{conf_str}")
            st.write(val if val else "—")

# ── Streamlit app ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="PA Policy Extractor", page_icon="💊", layout="wide")

st.title("Payer PA Policy Extractor")
st.caption("Upload payer Prior Authorization PDFs to extract 12 business parameters and an Access Score.")

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("OpenRouter API Key", type="password", placeholder="sk-or-v1-…")
    model = st.selectbox(
        "LLM Model",
        [
            "meta-llama/llama-3.1-8b-instruct",
            "meta-llama/llama-3.1-70b-instruct",
            "mistralai/mistral-7b-instruct",
            "google/gemma-3-27b-it",
        ],
        index=0,
    )
    st.divider()
    st.caption("Get a free API key at [openrouter.ai/keys](https://openrouter.ai/keys)")
    st.caption(f"LLM cache: `{CACHE_DIR / 'llm.sqlite'}`")

uploaded_files = st.file_uploader(
    "Upload PA Policy PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload one or more payer PA policy PDFs.",
)

if uploaded_files:
    st.info(f"{len(uploaded_files)} PDF(s) ready. Click **Run Extraction** to process.")

run = st.button("Run Extraction", type="primary", disabled=not (uploaded_files and api_key))

if run:
    if not api_key:
        st.error("Enter your OpenRouter API key in the sidebar.")
        st.stop()
    if not uploaded_files:
        st.error("Upload at least one PDF.")
        st.stop()

    embed_model = load_embedding_model()
    all_rows = []
    progress = st.progress(0, text="Starting…")
    status = st.empty()

    for i, uploaded_file in enumerate(uploaded_files):
        progress.progress(i / len(uploaded_files), text=f"Processing {uploaded_file.name}…")
        try:
            rows = process_one_pdf(
                uploaded_file.read(), uploaded_file.name,
                api_key, model, embed_model, status,
            )
            all_rows.extend(rows)
        except Exception as e:
            st.warning(f"Failed to process **{uploaded_file.name}**: {e}")

    progress.progress(1.0, text="Done!")
    status.empty()

    if not all_rows:
        st.error("No results extracted. Check that your PDFs contain PA criteria for any drug brand.")
        st.stop()

    details_map = {(r["Filename"], r["Brand"]): r.pop("_details", {}) for r in all_rows}

    df = pd.DataFrame(all_rows)
    for col in SUBMISSION_COLUMNS:
        if col not in df.columns:
            df[col] = "Not specified"
    df = df[SUBMISSION_COLUMNS]

    st.success(f"Extracted {len(df)} row(s) from {df['Filename'].nunique()} PDF(s).")

    st.subheader("Extracted Results")
    st.dataframe(df, use_container_width=True, hide_index=True)

    csv_bytes = df.to_csv(index=False).encode()
    st.download_button(
        label="Download submission.csv",
        data=csv_bytes,
        file_name="submission.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("Per-Brand Detail")
    for _, row_data in df.iterrows():
        det = details_map.get((row_data["Filename"], row_data["Brand"]), {})
        label = f"{row_data['Filename']}  —  {row_data['Brand']}  —  Access Score: {row_data.get('Access_Score', '—')}"
        with st.expander(label):
            _render_brand_card(row_data.to_dict(), det)
