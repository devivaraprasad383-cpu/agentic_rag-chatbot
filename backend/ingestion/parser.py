"""
Document parser: PDF · DOCX · TXT · Markdown
Returns structured output with table detection.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field


SUPPORTED = {".pdf", ".docx", ".doc", ".txt", ".md", ".markdown"}


@dataclass
class ParsedDocument:
    raw_text:    str
    tables:      list[dict] = field(default_factory=list)   # [{title, content, markdown}]
    metadata:    dict       = field(default_factory=dict)
    has_tables:  bool       = False


def parse_document(file_path: str) -> ParsedDocument:
    ext = Path(file_path).suffix.lower()
    if ext not in SUPPORTED:
        raise ValueError(f"Unsupported extension '{ext}'. Supported: {SUPPORTED}")
    if ext == ".pdf":
        return _parse_pdf(file_path)
    elif ext in {".docx", ".doc"}:
        return _parse_docx(file_path)
    else:
        return _parse_text(file_path)


# ── Parsers ───────────────────────────────────────────────────────

def _parse_pdf(path: str) -> ParsedDocument:
    from pypdf import PdfReader
    reader = PdfReader(path)
    pages, tables = [], []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[Page {i+1}]\n{text}")
            # Heuristic table detection: lines with multiple | chars
            table_lines = [l for l in text.split("\n") if l.count("|") >= 2]
            if len(table_lines) >= 3:
                tables.append({
                    "title":    f"Table on page {i+1}",
                    "content":  "\n".join(table_lines),
                    "page":     i + 1,
                })
    if not pages:
        raise ValueError("PDF has no extractable text (may be scanned).")
    return ParsedDocument(
        raw_text   = "\n\n".join(pages),
        tables     = tables,
        has_tables = bool(tables),
        metadata   = {"page_count": len(reader.pages)},
    )


def _parse_docx(path: str) -> ParsedDocument:
    from docx import Document
    doc    = Document(path)
    blocks, tables = [], []

    for para in doc.paragraphs:
        if para.text.strip():
            blocks.append(para.text)

    for idx, table in enumerate(doc.tables):
        rows, md_rows = [], []
        for i, row in enumerate(table.rows):
            cells = [c.text.strip() for c in row.cells]
            rows.append(" | ".join(cells))
            md_rows.append("| " + " | ".join(cells) + " |")
            if i == 0:
                md_rows.append("|" + "---|" * len(cells))

        table_text = "\n".join(rows)
        blocks.append(table_text)
        tables.append({
            "title":    f"Table {idx+1}",
            "content":  table_text,
            "markdown": "\n".join(md_rows),
        })

    if not blocks:
        raise ValueError("DOCX appears to be empty.")
    return ParsedDocument(
        raw_text   = "\n\n".join(blocks),
        tables     = tables,
        has_tables = bool(tables),
        metadata   = {"paragraph_count": len(doc.paragraphs)},
    )


def _parse_text(path: str) -> ParsedDocument:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    if not content.strip():
        raise ValueError("File is empty.")
    # Heuristic table detection for markdown tables
    lines  = content.split("\n")
    tables = []
    buf, in_table = [], False
    for line in lines:
        if "|" in line and line.count("|") >= 2:
            buf.append(line)
            in_table = True
        else:
            if in_table and len(buf) >= 3:
                tables.append({"title": "Markdown Table", "content": "\n".join(buf), "markdown": "\n".join(buf)})
            buf, in_table = [], False

    return ParsedDocument(
        raw_text   = content,
        tables     = tables,
        has_tables = bool(tables),
        metadata   = {"line_count": len(lines)},
    )
