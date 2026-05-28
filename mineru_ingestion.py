"""
MinerU-backed PDF ingestion for the PA policy RAG pipeline.

Use this file as a drop-in upgrade for the existing PyMuPDF/pypdf extraction
step. It returns the same PDFDoc shape used by zsads-rag.ipynb and
streamlit_app.py:

    PDFDoc(filename, pages, full_text, effective_date, page_count)

Typical notebook usage:

    from mineru_ingestion import MinerUConfig, extract_pdf

    MINERU_CONFIG = MinerUConfig(
        output_root=PROJECT_ROOT / ".mineru_cache",
        backend="hybrid-auto-engine",
        lang="en",
        formula=False,
        table=True,
    )

    doc = extract_pdf(pdf_path, config=MINERU_CONFIG)

If MinerU is unavailable or fails, extract_pdf falls back to the current
PyMuPDF/pypdf layout-aware extraction by default.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional


@dataclass
class PDFDoc:
    filename: str
    pages: list[str]
    full_text: str
    effective_date: Optional[str]
    page_count: int


@dataclass
class MinerUConfig:
    output_root: Path
    mineru_bin: str = "mineru"
    backend: str = "hybrid-auto-engine"
    lang: str = "en"
    formula: bool = False
    table: bool = True
    force: bool = False
    timeout_seconds: Optional[int] = None
    extra_args: list[str] = field(default_factory=list)


_DATE_RE = re.compile(
    r"(?im)(effective|revision|revised|policy\s+date|approved|last\s+updated|"
    r"reviewed|last\s+review|review|next\s+review|date\s+of\s+origin|origination|implementation|"
    r"adopted|publish(?:ed)?|date)\s*[:\-]?\s*"
    r"((?:\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})|"
    r"(?:[A-Z][a-z]+\.?\s+\d{1,2},?\s+\d{4})|"
    r"(?:\d{4}-\d{2}-\d{2}))"
)


def extract_pdf(
    pdf_path: str | Path,
    *,
    config: MinerUConfig,
    fallback_to_pymupdf: bool = True,
) -> PDFDoc:
    """Extract a local PDF through MinerU, with an optional PyMuPDF fallback."""
    pdf_path = Path(pdf_path)
    try:
        return extract_pdf_with_mineru(pdf_path, config=config)
    except Exception:
        if not fallback_to_pymupdf:
            raise
        return extract_pdf_with_pymupdf(pdf_path)


def extract_pdf_bytes(
    pdf_bytes: bytes,
    filename: str,
    *,
    config: MinerUConfig,
    fallback_to_pymupdf: bool = True,
) -> PDFDoc:
    """Streamlit-friendly wrapper for uploaded PDF bytes."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = Path(tmp.name)
    try:
        doc = extract_pdf(
            tmp_path,
            config=config,
            fallback_to_pymupdf=fallback_to_pymupdf,
        )
        return PDFDoc(
            filename=filename,
            pages=doc.pages,
            full_text=doc.full_text,
            effective_date=doc.effective_date,
            page_count=doc.page_count,
        )
    finally:
        tmp_path.unlink(missing_ok=True)


def extract_pdf_with_mineru(pdf_path: Path, *, config: MinerUConfig) -> PDFDoc:
    """Run MinerU if needed, then parse its Markdown/JSON outputs."""
    mineru_bin = _resolve_mineru_bin(config.mineru_bin)
    if not mineru_bin:
        raise RuntimeError(f"MinerU executable not found: {config.mineru_bin}")

    output_root = Path(config.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    existing = _find_mineru_outputs(output_root, pdf_path.stem)
    if config.force or not _has_parseable_output(existing):
        _run_mineru(pdf_path, output_root, config, mineru_bin)
        existing = _find_mineru_outputs(output_root, pdf_path.stem)

    pages = _pages_from_mineru_outputs(existing)
    if not pages:
        raise RuntimeError(f"MinerU produced no parseable text for {pdf_path.name}")

    return _build_doc(pdf_path.name, pages)


def extract_pdf_with_pymupdf(pdf_path: str | Path) -> PDFDoc:
    """Current baseline extraction: PyMuPDF block sort, then pypdf fallback."""
    pdf_path = Path(pdf_path)
    pages: list[str] = []
    try:
        import fitz

        doc = fitz.open(str(pdf_path))
        try:
            for page in doc:
                blocks = page.get_text("blocks") or []
                text_blocks = [
                    b for b in blocks
                    if len(b) >= 7 and b[6] == 0 and str(b[4]).strip()
                ]
                if not text_blocks:
                    pages.append(page.get_text("text") or "")
                    continue
                text_blocks.sort(key=lambda b: (int(float(b[0]) // 250), b[1], b[0]))
                pages.append("\n".join(str(b[4]).strip() for b in text_blocks))
        finally:
            doc.close()
    except Exception:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        pages = [(page.extract_text() or "") for page in reader.pages]

    return _build_doc(pdf_path.name, pages)


def clean_text(raw: str) -> str:
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", " ", raw)
    replacements = {
        "\u2019": "'",
        "\u2018": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00ae": "",
        "\u00a0": " ",
        "\u2022": "-",
        "\u2265": ">=",
        "\u2264": "<=",
    }
    for src, tgt in replacements.items():
        text = text.replace(src, tgt)
    text = re.sub(
        r"(?im)^(page\s+\d+(\s+of\s+\d+)?|confidential|proprietary|"
        r"copyright|all rights reserved)\s*$",
        "",
        text,
    )
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [re.sub(r"[ \t]{2,}", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _run_mineru(
    pdf_path: Path,
    output_root: Path,
    config: MinerUConfig,
    mineru_bin: Path,
) -> None:
    cmd = [
        str(mineru_bin),
        "-p",
        str(pdf_path),
        "-o",
        str(output_root),
        "-b",
        config.backend,
        "-l",
        config.lang,
        "-f",
        str(config.formula).lower(),
        "-t",
        str(config.table).lower(),
        *config.extra_args,
    ]
    env = os.environ.copy()
    env.setdefault("MINERU_TABLE_ENABLE", str(config.table).lower())
    env.setdefault("MINERU_FORMULA_ENABLE", str(config.formula).lower())
    proc = subprocess.run(
        cmd,
        env=env,
        text=True,
        capture_output=True,
        timeout=config.timeout_seconds,
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"MinerU failed for {pdf_path.name}: {detail[-2000:]}")


def _resolve_mineru_bin(mineru_bin: str) -> Optional[Path]:
    direct = Path(mineru_bin).expanduser()
    if direct.exists():
        return direct.resolve()

    found = shutil.which(mineru_bin)
    if found:
        return Path(found).resolve()

    sibling = Path(sys.executable).with_name(mineru_bin)
    if sibling.exists():
        return sibling.resolve()

    return None


def _find_mineru_outputs(output_root: Path, pdf_stem: str) -> dict[str, list[Path]]:
    files = {
        "middle": [],
        "content": [],
        "markdown": [],
    }
    for path in output_root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        suffix = path.suffix.lower()
        if suffix == ".json" and name.endswith("_middle.json"):
            files["middle"].append(path)
        elif suffix == ".json" and "content_list" in name:
            files["content"].append(path)
        elif suffix == ".md":
            files["markdown"].append(path)

    for key in files:
        files[key].sort(key=lambda p: _output_score(p, pdf_stem))
    return files


def _output_score(path: Path, pdf_stem: str) -> tuple[int, int, str]:
    path_text = str(path).lower()
    stem = pdf_stem.lower()
    if path.stem.lower() == stem:
        match_score = 0
    elif stem in path_text:
        match_score = 1
    else:
        match_score = 2
    return (match_score, len(path_text), path_text)


def _has_parseable_output(outputs: dict[str, list[Path]]) -> bool:
    return any(outputs.get(key) for key in ("middle", "content", "markdown"))


def _pages_from_mineru_outputs(outputs: dict[str, list[Path]]) -> list[str]:
    for middle_path in outputs.get("middle", []):
        pages = _parse_middle_json(middle_path)
        if pages:
            return pages

    for content_path in outputs.get("content", []):
        pages = _parse_content_list_json(content_path)
        if pages:
            return pages

    for markdown_path in outputs.get("markdown", []):
        pages = _parse_markdown(markdown_path)
        if pages:
            return pages

    return []


def _parse_middle_json(path: Path) -> list[str]:
    data = _read_json(path)
    pdf_info = data.get("pdf_info") if isinstance(data, dict) else None
    if not isinstance(pdf_info, list):
        return []

    pages: list[str] = []
    for page in pdf_info:
        if not isinstance(page, dict):
            pages.append("")
            continue
        blocks = page.get("para_blocks") or page.get("preproc_blocks") or []
        text = "\n".join(_extract_text_from_node(block) for block in blocks)
        pages.append(clean_text(text))
    return pages


def _parse_content_list_json(path: Path) -> list[str]:
    data = _read_json(path)
    items = data
    if isinstance(data, dict):
        for key in ("content", "content_list", "items", "data"):
            if isinstance(data.get(key), list):
                items = data[key]
                break
    if not isinstance(items, list):
        return []

    by_page: dict[int, list[str]] = {}
    without_page: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = _extract_content_list_item_text(item)
        if not text:
            continue
        page = _item_page_number(item)
        if page is None:
            without_page.append(text)
        else:
            by_page.setdefault(page, []).append(text)

    if by_page:
        pages = []
        for page_num in range(1, max(by_page) + 1):
            pages.append(clean_text("\n".join(by_page.get(page_num, []))))
        if without_page:
            pages.append(clean_text("\n".join(without_page)))
        return pages

    joined = clean_text("\n".join(without_page))
    return [joined] if joined else []


def _parse_markdown(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    text = clean_text(text)
    return [text] if text else []


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _item_page_number(item: dict[str, Any]) -> Optional[int]:
    for key in ("page_idx", "page_index", "page"):
        value = item.get(key)
        if isinstance(value, int):
            return value + 1 if key in {"page_idx", "page_index"} else value
        if isinstance(value, str) and value.isdigit():
            number = int(value)
            return number + 1 if key in {"page_idx", "page_index"} else number
    return None


def _extract_content_list_item_text(item: dict[str, Any]) -> str:
    item_type = str(item.get("type", "")).lower()
    fields = (
        "text",
        "content",
        "table_body",
        "table_html",
        "html",
        "caption",
        "footnote",
        "image_caption",
        "table_caption",
        "table_footnote",
    )
    parts = []
    for field_name in fields:
        value = item.get(field_name)
        if value:
            parts.append(_stringify(value))
    text = "\n".join(part for part in parts if part.strip())
    if not text:
        text = _extract_text_from_node(item)
    if item_type == "table" and text:
        text = "[TABLE]\n" + text
    return clean_text(_html_to_text(text))


def _extract_text_from_node(node: Any) -> str:
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return "\n".join(_extract_text_from_node(item) for item in node)
    if not isinstance(node, dict):
        return ""

    parts: list[str] = []
    for key in ("content", "text", "html", "table_body", "table_html"):
        value = node.get(key)
        if value:
            parts.append(_stringify(value))

    for span in node.get("spans") or []:
        if isinstance(span, dict):
            value = span.get("content") or span.get("text")
            if value:
                parts.append(str(value))

    for line in node.get("lines") or []:
        parts.append(_extract_text_from_node(line))

    for block in node.get("blocks") or []:
        parts.append(_extract_text_from_node(block))

    return clean_text(_html_to_text("\n".join(parts)))


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(_stringify(v) for v in value.values())
    if isinstance(value, list):
        return "\n".join(_stringify(v) for v in value)
    return str(value)


def _html_to_text(text: str) -> str:
    if "<" not in text or ">" not in text:
        return text
    text = re.sub(r"(?i)</(td|th)>", " | ", text)
    text = re.sub(r"(?i)</tr>", "\n", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return text


def _build_doc(filename: str, pages: Iterable[str]) -> PDFDoc:
    cleaned_pages = [clean_text(page or "") for page in pages]
    full_text = "\n\n".join(
        f"[PAGE {idx + 1}]\n{text}" for idx, text in enumerate(cleaned_pages)
    )
    effective_date = _extract_effective_date(cleaned_pages)
    return PDFDoc(
        filename=filename,
        pages=cleaned_pages,
        full_text=full_text,
        effective_date=effective_date,
        page_count=len(cleaned_pages),
    )


def _extract_effective_date(pages: list[str]) -> Optional[str]:
    if not pages:
        return None
    window = pages[0][:2500] + "\n" + pages[-1][-2500:]
    matches = [
        (label.lower(), _normalize_date(raw))
        for label, raw in _DATE_RE.findall(window)
    ]
    matches = [(label, date) for label, date in matches if date]
    if not matches:
        return None

    # "Next review" is useful metadata, but it should not drive freshness.
    policy_dates = [
        date for label, date in matches
        if "next" not in label
    ]
    return max(policy_dates or [date for _, date in matches])


def _normalize_date(raw: str) -> Optional[str]:
    raw = raw.strip().rstrip(",.;:")
    formats = (
        "%m/%d/%Y",
        "%m/%d/%y",
        "%m-%d-%Y",
        "%m-%d-%y",
        "%m.%d.%Y",
        "%m.%d.%y",
        "%Y-%m-%d",
        "%B %d, %Y",
        "%B %d %Y",
        "%b %d, %Y",
        "%b %d %Y",
    )
    for fmt in formats:
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _main() -> int:
    parser = argparse.ArgumentParser(description="Parse a PDF with MinerU into RAG-ready text.")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path(".mineru_cache"))
    parser.add_argument("--backend", default="hybrid-auto-engine")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--mineru-bin", default="mineru")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--formula", action="store_true")
    parser.add_argument("--no-table", action="store_true")
    parser.add_argument("--preview-chars", type=int, default=1200)
    args = parser.parse_args()

    config = MinerUConfig(
        output_root=args.output_root,
        mineru_bin=args.mineru_bin,
        backend=args.backend,
        lang=args.lang,
        formula=args.formula,
        table=not args.no_table,
        force=args.force,
    )
    doc = extract_pdf(
        args.pdf,
        config=config,
        fallback_to_pymupdf=not args.no_fallback,
    )
    preview = doc.full_text[: args.preview_chars]
    print(
        json.dumps(
            {
                "filename": doc.filename,
                "page_count": doc.page_count,
                "effective_date": doc.effective_date,
                "preview": preview,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
