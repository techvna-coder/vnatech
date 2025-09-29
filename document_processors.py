# document_processors.py
from __future__ import annotations
from io import BytesIO
from typing import Dict, Any, List, Tuple, Optional
import re

# PDF
from PyPDF2 import PdfReader

# PPTX
from pptx import Presentation

TEXT_SLIDE_HEADER = "--- Slide {idx} ---"

def _normalize_whitespace(s: str) -> str:
    return re.sub(r"[ \t]+", " ", s).strip()

def process_pdf(file_content: BytesIO) -> Tuple[str, Dict[str, Any]]:
    """
    Đọc PDF an toàn, bỏ qua trang lỗi cục bộ, trả về full_text + meta cơ bản.
    """
    reader = PdfReader(file_content)
    pages_text: List[str] = []
    errors: List[str] = []
    for i, page in enumerate(reader.pages, 1):
        try:
            text = page.extract_text() or ""
            text = text.replace("\x00", "").strip()
            if text:
                pages_text.append(f"--- Page {i} ---\n{text}")
        except Exception as e:
            errors.append(f"page {i}: {e}")

    full_text = "\n".join(pages_text).strip()
    if not full_text:
        raise Exception("No text could be extracted from the PDF")

    meta = {
        "total_pages": len(reader.pages),
        "has_sections": True,
        "sections": [{"type": "page", "number": i} for i in range(1, len(reader.pages) + 1)],
        "errors": errors,
        "key_terms": [],
    }
    return full_text, meta


def process_pptx(file_content: BytesIO) -> Tuple[str, Dict[str, Any]]:
    """
    Đọc PPTX an toàn, dùng has_text_frame/has_table, giữ header --- Slide N --- để tiện chunk theo slide.
    """
    prs = Presentation(file_content)
    slides_text: List[str] = []
    total_slides = len(list(prs.slides))
    has_any_table = False
    errors: List[str] = []

    for s_idx, slide in enumerate(prs.slides, 1):
        buf: List[str] = [TEXT_SLIDE_HEADER.format(idx=s_idx)]
        for shape in slide.shapes:
            # Text
            if getattr(shape, "has_text_frame", False) and getattr(shape, "text_frame", None):
                try:
                    paras = [p.text for p in shape.text_frame.paragraphs if p.text]
                    txt = "\n".join([_normalize_whitespace(t) for t in paras if _normalize_whitespace(t)])
                except Exception:
                    # Fallback an toàn
                    txt = _normalize_whitespace(getattr(shape, "text", "") or "")
                if txt:
                    buf.append(txt)

            # Table
            if getattr(shape, "has_table", False):
                try:
                    t = shape.table
                    has_any_table = True
                    for row in t.rows:
                        cells = [ _normalize_whitespace(cell.text) for cell in row.cells ]
                        cells = [c for c in cells if c]
                        if cells:
                            buf.append(" | ".join(cells))
                except Exception as e:
                    errors.append(f"slide {s_idx} table: {e}")
                    # Bỏ qua lỗi cục bộ, không ngắt toàn bộ file

        # Gom slide
        slide_block = "\n".join([b for b in buf if b and b.strip()])
        if slide_block.strip():
            slides_text.append(slide_block)

    full_text = "\n".join(slides_text).strip()
    if not full_text:
        raise Exception("No text could be extracted from the PowerPoint")

    meta = {
        "total_slides": total_slides,
        "has_sections": True,
        "sections": [{"type": "slide", "number": i} for i in range(1, total_slides + 1)],
        "has_tables": has_any_table,
        "errors": errors,
        "key_terms": [],
    }
    return full_text, meta


def chunk_text_smart(text: str, chunk_size: int = 1200, chunk_overlap: int = 150) -> List[str]:
    """
    Chiến lược:
      1) Ưu tiên tách theo '--- Slide ' hoặc '--- Page ' nếu có.
      2) Với mỗi block dài, cắt mềm theo số ký tự (có overlap).
    """
    # Ưu tiên theo Slide/Page header
    slide_blocks = re.split(r"\n(?=---\s*(Slide|Page)\s+\d+\s*---)", text)
    if len(slide_blocks) > 1:
        blocks = []
        # Vì re.split giữ delimiter trong danh sách trả về, ghép lại từng cặp cho đầy đủ block
        i = 0
        while i < len(slide_blocks):
            if re.match(r"---\s*(Slide|Page)\s+\d+\s*---", slide_blocks[i]):
                header = slide_blocks[i]
                body = slide_blocks[i + 1] if i + 1 < len(slide_blocks) else ""
                block = f"{header}\n{body}".strip()
                if block:
                    blocks.append(block)
                i += 2
            else:
                # Phần đầu có thể là văn bản trước slide đầu tiên
                if slide_blocks[i].strip():
                    blocks.append(slide_blocks[i].strip())
                i += 1
    else:
        # Không có header → xem như 1 block
        blocks = [text]

    def _split_soft(s: str) -> List[str]:
        if len(s) <= chunk_size:
            return [s]
        out = []
        start = 0
        while start < len(s):
            end = min(start + chunk_size, len(s))
            out.append(s[start:end])
            if end == len(s):
                break
            start = max(0, end - chunk_overlap)
        return out

    final_chunks: List[str] = []
    for b in blocks:
        final_chunks.extend(_split_soft(b))
    # Loại bỏ rỗng
    return [c for c in final_chunks if c and c.strip()]
