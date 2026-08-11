from __future__ import annotations

import csv
import hashlib
import html
import mimetypes
import re
import tempfile
import zipfile
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from functools import lru_cache
from html.parser import HTMLParser
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".json",
    ".jsonl",
    ".xml",
    ".html",
    ".htm",
    ".csv",
    ".tsv",
    ".pdf",
    ".docx",
    ".rtf",
    ".odt",
    ".xlsx",
    ".xls",
    ".eml",
    ".msg",
    ".zip",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
}


@dataclass(frozen=True, slots=True)
class ExtractionLimits:
    maximum_file_bytes: int = 26_214_400
    maximum_archive_entries: int = 100
    maximum_archive_uncompressed_bytes: int = 52_428_800
    maximum_spreadsheet_cells: int = 100_000
    maximum_ocr_pages: int = 50
    maximum_extracted_characters: int = 250_000
    maximum_nested_depth: int = 2


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    filename: str
    text: str
    media_type: str
    sha256: str
    size_bytes: int
    warnings: tuple[str, ...] = ()
    ocr_used: bool = False
    page_count: int | None = None


@dataclass(slots=True)
class _ExtractionState:
    limits: ExtractionLimits
    archive_entries: int = 0
    archive_uncompressed_bytes: int = 0
    ocr_pages: int = 0


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.parts.append(data.strip())


def _decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _safe_name(value: str) -> str:
    name = Path(value.replace(chr(92), "/")).name
    clean = re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip(" .")
    return clean[:180] or "document.txt"


def _bounded_text(text: str, limits: ExtractionLimits) -> tuple[str, tuple[str, ...]]:
    clean = text.replace("\x00", "").strip()
    if len(clean) <= limits.maximum_extracted_characters:
        return clean, ()
    return (
        clean[: limits.maximum_extracted_characters].rstrip(),
        ("Extracted text was truncated at the configured character limit.",),
    )


@lru_cache(maxsize=1)
def _ocr_engine() -> Any:
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()


def _ocr_image(image: Any, state: _ExtractionState) -> str:
    if state.ocr_pages >= state.limits.maximum_ocr_pages:
        raise ValueError("OCR page limit reached for this upload.")
    import numpy as np

    state.ocr_pages += 1
    result, _ = _ocr_engine()(np.asarray(image.convert("RGB")))
    return "\n".join(str(row[1]) for row in (result or []) if len(row) > 1).strip()


def _extract_pdf(data: bytes, state: _ExtractionState) -> tuple[str, bool, int, list[str]]:
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(data))
    page_text = [(page.extract_text() or "").strip() for page in reader.pages]
    ocr_used = False
    warnings: list[str] = []
    sparse_pages = [index for index, value in enumerate(page_text) if len(value) < 20]
    if sparse_pages:
        try:
            import pypdfium2 as pdfium

            pdf = pdfium.PdfDocument(data)
            for index in sparse_pages:
                if state.ocr_pages >= state.limits.maximum_ocr_pages:
                    warnings.append("Some scanned PDF pages were skipped at the OCR page limit.")
                    break
                image = pdf[index].render(scale=2).to_pil()
                recognized = _ocr_image(image, state)
                if recognized:
                    page_text[index] = recognized
                    ocr_used = True
        except Exception as exc:
            warnings.append(f"OCR was unavailable for sparse PDF pages: {exc}")
    joined = "\n\n".join(
        f"[Page {index}]\n{text}" for index, text in enumerate(page_text, start=1) if text
    )
    return joined, ocr_used, len(page_text), warnings


def _extract_docx(data: bytes) -> str:
    from docx import Document

    document = Document(BytesIO(data))
    parts = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
    for table_index, table in enumerate(document.tables, start=1):
        parts.append(f"[Table {table_index}]")
        for row in table.rows:
            values = [cell.text.replace("\n", " ").strip() for cell in row.cells]
            parts.append(" | ".join(values))
    return "\n\n".join(parts)


def _extract_odt(data: bytes) -> str:
    with zipfile.ZipFile(BytesIO(data)) as archive:
        root = ElementTree.fromstring(archive.read("content.xml"))
    paragraphs = []
    for element in root.iter():
        if element.tag.endswith(("}p", "}h")):
            value = "".join(element.itertext()).strip()
            if value:
                paragraphs.append(value)
    return "\n\n".join(paragraphs)


def _extract_spreadsheet(data: bytes, suffix: str, maximum_cells: int) -> str:
    rows: list[str] = []
    consumed = 0
    if suffix == ".xlsx":
        from openpyxl import load_workbook

        workbook = load_workbook(BytesIO(data), read_only=True, data_only=True)
        for sheet in workbook.worksheets:
            rows.append(f"[Sheet: {sheet.title}]")
            for row in sheet.iter_rows(values_only=True):
                values = ["" if value is None else str(value) for value in row]
                consumed += len(values)
                if consumed > maximum_cells:
                    rows.append("[Spreadsheet truncated at configured cell limit]")
                    return "\n".join(rows)
                rows.append(" | ".join(values))
    else:
        import xlrd

        workbook = xlrd.open_workbook(file_contents=data)
        for sheet in workbook.sheets():
            rows.append(f"[Sheet: {sheet.name}]")
            for row_index in range(sheet.nrows):
                values = [str(value) for value in sheet.row_values(row_index)]
                consumed += len(values)
                if consumed > maximum_cells:
                    rows.append("[Spreadsheet truncated at configured cell limit]")
                    return "\n".join(rows)
                rows.append(" | ".join(values))
    return "\n".join(rows)


def _extract_eml(data: bytes, state: _ExtractionState, depth: int) -> list[ExtractedDocument]:
    message = BytesParser(policy=policy.default).parsebytes(data)
    headers = [
        f"From: {message.get('from', '')}",
        f"To: {message.get('to', '')}",
        f"Cc: {message.get('cc', '')}",
        f"Date: {message.get('date', '')}",
        f"Subject: {message.get('subject', '')}",
    ]
    bodies: list[str] = []
    attachments: list[ExtractedDocument] = []
    for part in message.walk():
        disposition = part.get_content_disposition()
        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        if disposition == "attachment" and filename and payload:
            attachments.extend(_extract(payload, filename, state, depth + 1))
        elif part.get_content_type() == "text/plain" and disposition != "attachment":
            try:
                bodies.append(part.get_content())
            except Exception:
                bodies.append(_decode(payload))
    text = "\n".join(headers + ["", *bodies]).strip()
    own = _make_document("email.eml", data, text, warnings=())
    return [own, *attachments]


def _extract_msg(data: bytes, state: _ExtractionState, depth: int) -> list[ExtractedDocument]:
    import extract_msg

    with tempfile.TemporaryDirectory(prefix="contractlab-msg-") as directory:
        path = Path(directory) / "message.msg"
        path.write_bytes(data)
        message = extract_msg.openMsg(path)
        try:
            text = "\n".join(
                [
                    f"From: {message.sender or ''}",
                    f"To: {message.to or ''}",
                    f"Cc: {message.cc or ''}",
                    f"Date: {message.date or ''}",
                    f"Subject: {message.subject or ''}",
                    "",
                    message.body or "",
                ]
            ).strip()
            documents = [_make_document("email.msg", data, text, warnings=())]
            for attachment in message.attachments:
                payload = getattr(attachment, "data", None)
                filename = getattr(attachment, "longFilename", None) or getattr(
                    attachment, "shortFilename", None
                )
                if isinstance(payload, bytes) and filename:
                    documents.extend(_extract(payload, filename, state, depth + 1))
            return documents
        finally:
            message.close()


def _make_document(
    filename: str,
    data: bytes,
    text: str,
    *,
    warnings: tuple[str, ...] | list[str],
    ocr_used: bool = False,
    page_count: int | None = None,
    limits: ExtractionLimits | None = None,
) -> ExtractedDocument:
    active = limits or ExtractionLimits()
    bounded, truncation = _bounded_text(text, active)
    safe = _safe_name(filename)
    media_type = mimetypes.guess_type(safe)[0] or "application/octet-stream"
    return ExtractedDocument(
        filename=safe,
        text=bounded,
        media_type=media_type,
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        warnings=tuple(warnings) + truncation,
        ocr_used=ocr_used,
        page_count=page_count,
    )


def _extract(
    data: bytes, filename: str, state: _ExtractionState, depth: int
) -> list[ExtractedDocument]:
    if depth > state.limits.maximum_nested_depth:
        raise ValueError("Nested attachment or archive depth exceeds the configured limit.")
    if len(data) > state.limits.maximum_file_bytes:
        raise ValueError(f"{filename} exceeds the configured per-file size limit.")
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix or 'no extension'}")

    if suffix == ".zip":
        documents: list[ExtractedDocument] = []
        with zipfile.ZipFile(BytesIO(data)) as archive:
            members = [member for member in archive.infolist() if not member.is_dir()]
            if len(members) + state.archive_entries > state.limits.maximum_archive_entries:
                raise ValueError("ZIP archive exceeds the configured entry limit.")
            for member in members:
                if member.flag_bits & 0x1:
                    raise ValueError("Encrypted ZIP entries are not supported.")
                state.archive_entries += 1
                state.archive_uncompressed_bytes += member.file_size
                if (
                    state.archive_uncompressed_bytes
                    > state.limits.maximum_archive_uncompressed_bytes
                ):
                    raise ValueError("ZIP archive exceeds the configured uncompressed-size limit.")
                member_name = _safe_name(member.filename)
                if Path(member_name).suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue
                documents.extend(_extract(archive.read(member), member_name, state, depth + 1))
        return documents

    if suffix == ".eml":
        documents = _extract_eml(data, state, depth)
        return [
            document
            if document.filename != "email.eml"
            else _make_document(
                filename, data, document.text, warnings=document.warnings, limits=state.limits
            )
            for document in documents
        ]
    if suffix == ".msg":
        documents = _extract_msg(data, state, depth)
        return [
            document
            if document.filename != "email.msg"
            else _make_document(
                filename, data, document.text, warnings=document.warnings, limits=state.limits
            )
            for document in documents
        ]

    warnings: list[str] = []
    ocr_used = False
    page_count = None
    if suffix in {".txt", ".md", ".json", ".jsonl", ".xml"}:
        text = _decode(data)
    elif suffix in {".html", ".htm"}:
        parser = _HTMLTextExtractor()
        parser.feed(_decode(data))
        text = "\n".join(html.unescape(part) for part in parser.parts)
    elif suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        rows = csv.reader(StringIO(_decode(data)), delimiter=delimiter)
        text = "\n".join(" | ".join(row) for row in rows)
    elif suffix == ".pdf":
        text, ocr_used, page_count, warnings = _extract_pdf(data, state)
    elif suffix == ".docx":
        text = _extract_docx(data)
    elif suffix == ".rtf":
        from striprtf.striprtf import rtf_to_text

        text = rtf_to_text(_decode(data))
    elif suffix == ".odt":
        text = _extract_odt(data)
    elif suffix in {".xlsx", ".xls"}:
        text = _extract_spreadsheet(data, suffix, state.limits.maximum_spreadsheet_cells)
    else:
        from PIL import Image

        image = Image.open(BytesIO(data))
        text = _ocr_image(image, state)
        ocr_used = True
        page_count = 1
    if not text.strip():
        warnings.append("No readable text was extracted from this file.")
    return [
        _make_document(
            filename,
            data,
            text,
            warnings=warnings,
            ocr_used=ocr_used,
            page_count=page_count,
            limits=state.limits,
        )
    ]


def extract_documents(
    data: bytes,
    filename: str,
    *,
    limits: ExtractionLimits | None = None,
) -> list[ExtractedDocument]:
    """Extract one upload and any supported archive/email attachments entirely in memory."""
    active = limits or ExtractionLimits()
    return _extract(data, filename, _ExtractionState(active), 0)


def extract_text(data: bytes, filename: str) -> str:
    """Backward-compatible single-file extraction used by the original assistant."""
    documents = extract_documents(data, filename)
    if not documents:
        raise ValueError("No supported documents were found.")
    return "\n\n".join(document.text for document in documents if document.text).strip()
