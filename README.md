# Payer PA Policy Extraction — Hybrid RAG Pipeline

End-to-end extraction of 12 business parameters + an Access Score (0–100) from payer Prior Authorization (PA) policy PDFs.

---

## 1. How to run

Two main files:

- `zsads-rag.ipynb` — full pipeline notebook
- `requirements.txt` — pinned dependencies

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Set your OpenRouter API key

In the `Configuration` cell, find the `else:` branch and set your key:

```python
OPENROUTER_API_KEY = "sk-or-v1-YOUR_KEY_HERE"
OPENROUTER_MODEL   = "meta-llama/llama-3.1-8b-instruct"
```

Get a key at <https://openrouter.ai/keys>. The free tier covers Llama 3.1 8B at ~$0.02 per million tokens.

### Step 3 — Point the notebook at your PDFs

In the `Configuration` cell:

```python
PDF_DIR = Path("/absolute/path/to/Sample_PsO_ADS_Track")
```

### Step 4 — Run

Open `zsads-rag.ipynb` in Jupyter and run all cells. It produces `submission.csv` with 15 columns, one row per `(Filename, Brand)`. The notebook writes incrementally — if it crashes midway, re-run and it resumes from the next unprocessed PDF.

---

## 2. Models used

### 2.1 Generation LLM — Llama 3.1 8B Instruct

`meta-llama/llama-3.1-8b-instruct` via OpenRouter. Selected for:

- Strong JSON-mode adherence (essential for structured extraction)
- 128k native context window (pipeline uses 8k)
- Effectively free on the OpenRouter free tier
- Open-weights — reproducible, not tied to a closed vendor

Called with `temperature=0` and `response_format={"type":"json_object"}`. The system prompt grounds the model in PA policy domain knowledge and forbids use of external knowledge — every field must be backed by a verbatim quote from the retrieved context.

### 2.2 Embedding model — BAAI/bge-large-en-v1.5

| Property | Value |
|---|---|
| Parameters | 560M |
| Embedding dim | 1024 |
| Tokenizer | BERT WordPiece, 512 max tokens |
| MTEB retrieval avg | 54.2 |
| License | MIT |

Loaded once via `HuggingFaceEmbeddings`, normalized to unit length. Vectors are stored in ChromaDB (`PersistentClient`, sqlite-backed) on disk.

---

## 3. Architecture

### High-level flow

```
                      ┌────────────────────────────┐
                      │     Payer PA Policy PDF    │
                      └─────────────┬──────────────┘
                                    │
                        ┌───────────▼───────────┐
                        │   PDF Extraction       │
                        │   PyMuPDF (primary)    │
                        │   MineU (complex PDFs) │
                        │   pypdf (fallback)     │
                        └───────────┬───────────┘
                                    │
                        ┌───────────▼───────────┐
                        │   Clean + Chunk        │  900-char chunks,
                        │   provenance-tagged    │  200-char overlap
                        └───────────┬───────────┘
                                    │
                        ┌───────────▼───────────┐
                        │   Brand Detection      │ ◀── Llama 3.1 8B
                        │   regex → indication   │     per-brand calls
                        │   → PA relevance check │
                        │   → discovery fallback │
                        └───────────┬───────────┘
                                    │
                          for each (brand, query)
                                    │
          ┌─────────────────────────┼──────────────────────┐
          ▼                         ▼                      ▼
    ┌──────────┐              ┌──────────┐          ┌────────────┐
    │   BM25   │              │  BGE +   │          │  Source    │
    │ lexical  │              │ ChromaDB │          │ confidence │
    │ retrieval│              │ semantic │          │ freshness  │
    └────┬─────┘              └────┬─────┘          └─────┬──────┘
         │                         │                      │
         └──────────┬──────────────┘                      │
                    ▼                                      │
            ┌───────────────┐                             │
            │  Reciprocal   │                             │
            │  Rank Fusion  │                             │
            │   (RRF, k=20) │                             │
            └───────┬───────┘                             │
                    ▼                                      │
            ┌──────────────┐                              │
            │  Top-20      │                              │
            │  chunks      │                              │
            │ (dedup ≤36)  │                              │
            └──────┬───────┘                              │
                   │                                      │
                   ▼                                      ▼
          ┌─────────────────────────────────────────────────┐
          │   LLM Extraction (Llama 3.1 8B)                 │
          │     • JSON mode, temperature 0                  │
          │     • PA domain context in system prompt        │
          │     • One call returns all 12 fields per brand  │
          │     • English-format values, not codes          │
          └────────────────────────┬────────────────────────┘
                                   │
                                   ▼
                   ┌────────────────────────────┐
                   │   Confidence gate          │  drop if conf < 0.45
                   │   Evidence-grounding gate  │  drop if quote ≠ chunk
                   └─────────────┬──────────────┘
                                 │
                                 ▼
                   ┌────────────────────────────┐
                   │   Business-rule pass        │
                   └─────────────┬──────────────┘
                                 │
                                 ▼
                   ┌────────────────────────────┐
                   │   Access Score (0–100)      │
                   └─────────────┬──────────────┘
                                 │
                                 ▼
                   ┌────────────────────────────┐
                   │   submission.csv            │
                   └────────────────────────────┘
```

---

## 4. PDF Extraction

### 4.1 Hybrid extractor — PyMuPDF + MineU

The pipeline uses a two-tier strategy per PDF:

1. **PyMuPDF** (primary) — fast block-sort extraction, runs on every PDF
2. **MineU** (upgrade) — ML-based layout detection, triggered only for complex PDFs

A PDF is treated as complex if either condition is true:

| Signal | Threshold |
|---|---|
| Average characters per page | < 300 (sparse text, likely image-heavy) |
| Image block ratio | > 35% of total blocks are images |

When triggered, MineU runs its pipeline-mode backend (no GPU required) and produces structured Markdown with proper table formatting (`| col | col |`). If MineU yields ≥ 90% of the char count PyMuPDF found, the MineU result is used; otherwise PyMuPDF is kept. MineU is disabled on Kaggle by default (`USE_MINERU = not IS_KAGGLE`) due to resource constraints.

### 4.2 Column layout detection

PyMuPDF extraction uses dynamic column boundary detection rather than a hardcoded pixel threshold:

- Collects all block x-start positions (rounded to 10pt grid)
- Finds gaps between adjacent positions; marks gaps ≥ `max(8% of page width, 2.5× median gap)` as column boundaries
- Filters out boundaries in the outer 5% margins
- Sorts blocks by `(column_index, y, x)` — handles 1, 2, 3, 4+ columns on any page size or orientation

### 4.3 Checkbox handling

Three types are handled:

| Type | Handling |
|---|---|
| Unicode glyphs (☑ ☐ ✓) | Normalized in `clean_text`: `☑` → `[X]`, `☐` → `[ ]` |
| AcroForm interactive widgets | Extracted via `page.widgets()`, injected as synthetic text blocks at their page position |
| Wingdings/ZapfDingbats font glyphs | Mapped via private-use-area codepoint table |

---

## 5. Brand Detection

Four-layer pipeline per document:

1. **Regex alias scan** — checks for TREMFYA/STELARA and their generic names (guselkumab, ustekinumab)
2. **Discovery pass** (if neither found) — LLM lists all brands with PA criteria in the document; returns `{"brands": [...]}`
3. **Indication check** — per-brand LLM call (`max_tokens=32`) confirms PsO criteria; falls back to PsA; falls back to regex heuristic
4. **PA relevance validation** — per-brand LLM call confirms the brand has actual PA approval criteria (not just listed in an applicable drug list, step-therapy alternative list, or formulary table with no criteria)

If all brands fail the PA relevance check, the pipeline falls back to the unvalidated list to ensure no PDF is skipped silently.

---

## 6. LLM Extraction

### 6.1 System prompt

The extraction system prompt gives the model full PA policy domain context before it sees any document text:

- Explains what a payer PA policy is and what it governs
- Lists all 12 fields to extract
- Provides domain-specific extraction rules:
  - **TB mapping**: "latent tuberculosis screening", "QuantiFERON", "tuberculin skin test" → `TB_Test_Required = Yes`
  - **Step therapy**: only Yes if the policy explicitly requires prior treatment failure — not just a mention of another drug
  - **Durations**: extracted in months only; weeks/years converted
  - **Specialist types**: only explicitly named specialties; no inference
  - **Quantity limits**: describe the actual restriction in plain English (form + quantity + frequency)

### 6.2 Output format

All values are plain readable English:

| Field type | Format |
|---|---|
| Multiple items (specialists, drugs) | Comma-separated: `Dermatologist, Rheumatologist` |
| Ordered steps | Numbered lines: `1. Trial methotrexate...\n2. Trial adalimumab...` |
| Yes/No fields | Exact token: `Yes`, `No`, `Not specified` |
| Age | Inequality: `>=18`, `FDA labelled age` |
| Step counts | Plain number: `1`, `2`, `NA` |
| Durations | `X Months` format: `12 Months`, `6 Months` |

### 6.3 Quality gates

Two gates filter each extracted field before it enters the output:

- **Confidence gate**: fields with confidence < 0.45 → replaced with `"Insufficient evidence found"`
- **Evidence-grounding gate**: fields whose evidence quote does not fuzzy-match (rapidfuzz partial ratio < 70) any retrieved chunk → replaced with fallback token

---

## 7. What each component does

| Stage | What it produces | Why |
|---|---|---|
| **PDF extraction** | Per-page text in reading order | PyMuPDF block-sort + MineU for tables; pypdf fallback for corrupt files |
| **Column detection** | Dynamic column boundaries | Handles 1–4+ columns on any page size without hardcoded pixel values |
| **Checkbox extraction** | `[X]`/`[ ]` inline with adjacent text | Three types handled: Unicode, AcroForm widgets, Wingdings |
| **Brand detection** | List of brands with PA criteria | Regex + indication LLM + PA relevance validation + discovery fallback |
| **Chunking** | 900-char chunks with page metadata | Position metadata feeds the trust signal |
| **BM25** | Top-12 lexical matches per query | Catches exact drug names and numerals embeddings miss |
| **ChromaDB** | Top-12 semantic matches per query | Catches paraphrases like "step therapy" ↔ "must first trial and fail" |
| **RRF fusion** | Single ranked list (top-20) | k=20 gives top-ranked results more weight |
| **Source confidence** | Freshness, trust, consistency signals | Diagnostic signals per retrieval |
| **LLM extraction** | JSON with 12 fields × {value, conf, evidence} | One call per brand, shared attention across all fields |
| **Confidence gate** | Replaces low-confidence values | Better silence than wrong |
| **Evidence gate** | Replaces ungrounded quotes | Catches confident-sounding hallucinations |
| **Business-rule pass** | Spec-mandated post-processing | Reauth consistency, duration normalization |
| **Access Score** | Integer 0–100 | Field-weighted, zero credit for fallback fields |
| **CSV writer** | `submission.csv` appended per PDF | Crash-safe and resumable |

---

## 8. Access Score (0–100)

Higher score = better patient access (fewer barriers).

### 8.1 Seven sub-scorers

| Feature | Max pts | Scoring logic |
|---|---:|---|
| Age threshold | 15 | `≤18 → 15` · `≤21 → 13` · `≤30 → 10` · `≤50 → 6` · `>50 → 3` · FDA-labelled → 12 |
| Step therapy (total steps) | 35 | 0 → 35 · 1 → 28 · 2 → 20 · 3 → 12 · 4 → 7 · 5+ → 3 |
| Phototherapy required | 10 | No → 10 · Yes → 0 |
| Initial auth duration | 15 | ≥12 mo → 15 · 6–11 mo → 8 · <6 mo → 4 · Unspecified → 6 |
| TB test required | 5 | No → 5 · Yes → 0 · Not specified → 2 |
| Reauthorization | 10 | Not required → 10 · required ≥12 mo → 7 · 6–11 mo → 4 · <6 mo → 2 |
| Specialist restriction | 10 | No restriction → 10 · Restricted → 5 |

### 8.2 Zero credit for fallback values

A field that came back as `"Insufficient evidence found"` contributes **0 points** — not the default credit. This stops artificially high scores for empty extractions.

---

## 9. Caching

Three persistent tiers under `.rag_cache/`:

| Tier | Mechanism | Key |
|---|---|---|
| L1 query cache | `lru_cache(2048)` | normalized query string (process-local) |
| L2 retrieval cache | `diskcache` | `(doc_hash, query)` — survives restarts |
| L3 LLM cache | SQLite | `(model, system_prompt, prompt)` — survives restarts |
| MineU output cache | filesystem | PDF stem → `.mineru_cache/` directory |

Changing a prompt invalidates only the affected LLM calls. Everything else is free on re-run.

---

## 10. Output schema

`submission.csv` — 15 columns, one row per `(Filename, Brand)`:

```
Filename, Brand,
Age, Step_Therapy_Requirements,
Number_of_Steps_through_Brands, Number_of_Steps_through_Generic,
Step_Through_Phototherapy, TB_Test_Required,
Initial_Authorization_Duration, Reauthorization_Duration,
Reauthorization_Required, Reauthorization_Requirements,
Specialist_Types, Quantity_Limits,
Access_Score
```
