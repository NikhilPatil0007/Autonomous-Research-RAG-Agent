"""PDF extract (PyMuPDF) → light clean → token chunks with page/section metadata."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf
import tiktoken

from app.config import ROOT_DIR, get_settings
from app.observability import get_logger, span

log = get_logger("app.loaders")

TARGET_TOKENS = 640
OVERLAP_TOKENS = 80
_HEADING_MAX_WORDS = 12
_enc = tiktoken.get_encoding("cl100k_base")


@dataclass
class PageText:
    page: int
    text: str
    headings: list[str]


@dataclass
class Chunk:
    text: str
    source: str
    page: int
    title: str
    section: str
    chunk_id: str
    document_id: str

    def payload(self) -> dict:
        return {
            "text": self.text,
            "source": self.source,
            "page": self.page,
            "title": self.title,
            "section": self.section,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
        }


def document_id_for(pdf_path: Path) -> str:
    rel = _rel_source(pdf_path)
    return hashlib.sha256(rel.encode("utf-8")).hexdigest()[:16]


def _rel_source(pdf_path: Path) -> str:
    try:
        return pdf_path.resolve().relative_to(ROOT_DIR.resolve()).as_posix()
    except ValueError:
        return pdf_path.name


def _clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _looks_like_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 120:
        return False
    words = line.split()
    if len(words) > _HEADING_MAX_WORDS:
        return False
    if line.endswith((".", ",", ";", ":")) and not re.match(r"^\d+(\.\d+)*\s", line):
        return False
    letters = [c for c in line if c.isalpha()]
    if len(letters) < 3:
        return False
    if line.isupper() or line.istitle():
        return True
    if re.match(r"^(\d+(\.\d+)*|[IVXLC]+)[\.\)]\s+\S", line):
        return True
    return False


def _extract_pages(pdf_path: Path) -> tuple[str, list[PageText]]:
    doc = pymupdf.open(pdf_path)
    try:
        meta_title = (doc.metadata or {}).get("title") or ""
        title = _clean(meta_title) or pdf_path.stem.replace("_", " ").replace("-", " ")
        pages: list[PageText] = []
        for i, page in enumerate(doc):
            raw = page.get_text("text") or ""
            text = _clean(raw)
            headings = [ln.strip() for ln in text.splitlines() if _looks_like_heading(ln)]
            pages.append(PageText(page=i + 1, text=text, headings=headings))
        return title, pages
    finally:
        doc.close()


def _split_tokens(text: str, page: int, section: str) -> list[tuple[str, int, str]]:
    """Return (chunk_text, page, section) pieces ~TARGET_TOKENS with overlap."""
    tokens = _enc.encode(text)
    if not tokens:
        return []
    pieces: list[tuple[str, int, str]] = []
    start = 0
    n = len(tokens)
    while start < n:
        end = min(start + TARGET_TOKENS, n)
        chunk = _enc.decode(tokens[start:end]).strip()
        if chunk:
            pieces.append((chunk, page, section))
        if end >= n:
            break
        start = max(end - OVERLAP_TOKENS, start + 1)
    return pieces


def load_pdf_chunks(pdf_path: Path) -> list[Chunk]:
    pdf_path = Path(pdf_path)
    with span("load_pdf_chunks", component="ingestion", source=str(pdf_path)) as bag:
        title, pages = _extract_pages(pdf_path)
        source = _rel_source(pdf_path)
        doc_id = document_id_for(pdf_path)
        current_section = title
        chunks: list[Chunk] = []
        idx = 0
        empty_pages = 0

        for page in pages:
            if page.headings:
                current_section = page.headings[0]
            body = page.text
            if not body:
                empty_pages += 1
                continue
            for text, pg, section in _split_tokens(body, page.page, current_section):
                chunk_id = f"{doc_id}:p{pg}:c{idx}"
                chunks.append(
                    Chunk(
                        text=text,
                        source=source,
                        page=pg,
                        title=title,
                        section=section,
                        chunk_id=chunk_id,
                        document_id=doc_id,
                    )
                )
                idx += 1
                for h in page.headings[1:]:
                    if h in text:
                        current_section = h
        bag["n_pages"] = len(pages)
        bag["n_empty_pages"] = empty_pages
        bag["n_chunks"] = len(chunks)
        bag["document_id"] = doc_id
        bag["title"] = title
        log.info(
            "pdf.chunked",
            extra={
                "component": "ingestion",
                "event": "pdf.chunked",
                "source": source,
                "document_id": doc_id,
                "title": title,
                "n_pages": len(pages),
                "n_empty_pages": empty_pages,
                "n_chunks": len(chunks),
            },
        )
        if not chunks:
            log.warning(
                "pdf.no_text",
                extra={
                    "component": "ingestion",
                    "event": "pdf.no_text",
                    "source": str(pdf_path),
                },
            )
        return chunks


def iter_pdfs(pdf_dir: Path | None = None) -> list[Path]:
    settings = get_settings()
    root = Path(pdf_dir or settings.pdf_dir)
    return sorted(p for p in root.rglob("*.pdf") if p.is_file())
