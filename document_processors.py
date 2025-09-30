import streamlit as st
from io import BytesIO
import time
from typing import List, Dict, Any, Tuple

# OpenAI (cho embeddings)
try:
    from openai import OpenAI
    client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", ""))
except ImportError:
    st.error("OpenAI library not installed")
    client = None

# PDF/PPTX
try:
    import PyPDF2
except ImportError:
    st.error("PyPDF2 not installed")
    PyPDF2 = None

try:
    from pptx import Presentation
except ImportError:
    st.error("python-pptx not installed")
    Presentation = None

# Tokenizer
try:
    import tiktoken
    tokenizer = tiktoken.encoding_for_model("text-embedding-3-small")
except Exception:
    tokenizer = None
    st.warning("tiktoken not available; chunking will fall back to rough splitting.")

# ---------- Utilities ----------
def _safe_tokenize(text: str) -> List[int]:
    if tokenizer:
        return tokenizer.encode(text)
    # fallback: giả lập 'token' theo ký tự
    return list(text)

def _safe_detokenize(tokens: List[int]) -> str:
    if tokenizer:
        return tokenizer.decode(tokens)
    # fallback: ghép lại ký tự
    return "".join(tokens)

def count_tokens(text: str) -> int:
    try:
        return len(_safe_tokenize(text))
    except Exception:
        return 0

def preprocess_text(text: str) -> str:
    # Chuẩn hóa nhẹ, tránh phá vỡ nội dung kỹ thuật
    try:
        text = text.replace("\f", "\n").replace("\r", "\n")
        # Chuẩn hóa dấu ngoặc kép và gạch ngang thường gặp từ OCR
        text = text.replace("–", "-").replace("—", "-")
        text = text.replace(""", '"').replace(""", '"')
        text = text.replace("'", "'").replace("'", "'")
        # Dọn khoảng trắng thừa theo dòng, giữ xuống dòng hợp lý
        lines = [ln.strip() for ln in text.split("\n")]
        text = "\n".join([ln for ln in lines if ln])
        return text.strip()
    except Exception:
        return text

# ---------- Extractors with metadata ----------
def process_pdf(file_content: BytesIO) -> Tuple[str, Dict[str, Any]]:
    """Extract text + metadata from PDF."""
    if PyPDF2 is None:
        raise Exception("PyPDF2 not installed")
    try:
        reader = PyPDF2.PdfReader(file_content)
        pages = []
        for i, page in enumerate(reader.pages, 1):
            try:
                t = page.extract_text() or ""
            except Exception as e:
                st.warning(f"Failed to extract text from page {i}: {e}")
                t = ""
            if t.strip():
                pages.append(f"--- Page {i} ---\n{t}")
        full_text = "\n".join(pages).strip()
        if not full_text:
            raise Exception("No text could be extracted from the PDF")
        meta = {
            "total_pages": len(reader.pages),
            "has_sections": True,
            "sections": [{"type": "page", "number": i} for i in range(1, len(reader.pages)+1)],
            "has_tables": False,  # placeholder
            "has_lists": False,   # placeholder
            "key_terms": []
        }
        return full_text, meta
    except Exception as e:
        raise Exception(f"Failed to process PDF: {str(e)}")

def process_pptx(file_content: BytesIO) -> Tuple[str, Dict[str, Any]]:
    """Extract text + metadata from PPTX."""
    if Presentation is None:
        raise Exception("python-pptx not installed")
    try:
        prs = Presentation(file_content)
        slides_text = []
        has_tables = False
        
        for s_idx, slide in enumerate(prs.slides, 1):
            buf = [f"--- Slide {s_idx} ---"]
            for shape in slide.shapes:
                try:
                    # Textbox hoặc text frame
                    if hasattr(shape, "text") and shape.text:
                        buf.append(shape.text)
                    
                    # Table - kiểm tra kỹ càng hơn
                    if shape.shape_type == 19:  # MSO_SHAPE_TYPE.TABLE = 19
                        has_tables = True
                        if hasattr(shape, "table"):
                            table = shape.table
                            for row in table.rows:
                                cells = []
                                for cell in row.cells:
                                    try:
                                        cell_text = cell.text.strip()
                                        if cell_text:
                                            cells.append(cell_text)
                                    except Exception:
                                        continue
                                if cells:
                                    buf.append(" | ".join(cells))
                except Exception as e:
                    # Bỏ qua shape lỗi, tiếp tục xử lý
                    st.warning(f"Skipped shape in slide {s_idx}: {str(e)}")
                    continue
            
            slide_block = "\n".join([b for b in buf if b and b.strip() != f"--- Slide {s_idx} ---"])
            if slide_block.strip():
                slides_text.append(slide_block)
        
        full_text = "\n".join(slides_text).strip()
        if not full_text:
            raise Exception("No text could be extracted from the PowerPoint")
        
        meta = {
            "total_slides": len(list(prs.slides)),
            "has_sections": True,
            "sections": [{"type": "slide", "number": i} for i in range(1, len(list(prs.slides))+1)],
            "has_tables": has_tables,
            "has_lists": False,
            "key_terms": []
        }
        return full_text, meta
    except Exception as e:
        raise Exception(f"Failed to process PPTX: {str(e)}")

# ---------- Chunking ----------
def _chunk_by_tokens(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    tokens = _safe_tokenize(text)
    if len(tokens) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        piece = _safe_detokenize(tokens[start:end]).strip()
        if piece:
            chunks.append(piece)
        if end == len(tokens):
            break
        start = end - chunk_overlap
    return chunks

def chunk_text_smart(text: str,
                     doc_metadata: Dict[str, Any],
                     chunk_size: int = 1000,
                     chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Tách theo 'đơn vị logic' (Page/Slide) nếu có, sau đó chặt theo token.
    Trả về list dicts với đầy đủ metadata như streamlit_app.py đang kỳ vọng.
    """
    text = preprocess_text(text)
    sections = []
    # Ưu tiên cắt theo nhãn đã chèn trong extractor
    split_pages = [p for p in text.split("\n--- Page ") if p.strip()]
    split_slides = [s for s in text.split("\n--- Slide ") if s.strip()]

    if len(split_pages) > 1:  # PDF
        for blk in split_pages:
            # blk bắt đầu dạng: "X ---\n<content>" hoặc "X --- <content>"
            try:
                header, *rest = blk.split("---", 1)
                number = int(header.strip())
                content = rest[0] if rest else ""
            except Exception:
                number, content = 0, blk
            sections.append(("page", number, content.strip()))
    elif len(split_slides) > 1:  # PPTX
        for blk in split_slides:
            try:
                header, *rest = blk.split("---", 1)
                number = int(header.strip())
                content = rest[0] if rest else ""
            except Exception:
                number, content = 0, blk
            sections.append(("slide", number, content.strip()))
    else:
        # fallback: cả tài liệu 1 khối
        sections.append(("document", 0, text))

    out: List[Dict[str, Any]] = []
    total_chunks_counter = 0
    temp_chunks_per_section = []

    # Chunk theo từng section để giữ ngữ cảnh
    for (stype, snum, scontent) in sections:
        smalls = _chunk_by_tokens(scontent, chunk_size, chunk_overlap)
        temp_chunks_per_section.append((stype, snum, smalls))
        total_chunks_counter += len(smalls)

    # Duyệt lần 2 để gán chỉ số chunk toàn cục & metadata đẹp
    running_idx = 0
    for (stype, snum, smalls) in temp_chunks_per_section:
        for i, txt in enumerate(smalls):
            out.append({
                "text": txt,
                "chunk_index": running_idx,
                "total_chunks": total_chunks_counter,
                "section_type": stype,
                "section_number": snum,
                "is_complete_section": (len(smalls) == 1),
                "token_count": count_tokens(txt),
                "word_count": len(txt.split())
            })
            running_idx += 1

    return out

# ---------- Embeddings ----------
def get_embeddings(texts: List[str], batch_size: int = 100) -> List[List[float]]:
    if client is None:
        raise Exception("OpenAI client is not initialized")
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        if len(texts) > batch_size:
            st.progress((i + len(batch)) / len(texts), text=f"Generating embeddings... {i + len(batch)}/{len(texts)}")
        resp = client.embeddings.create(model="text-embedding-3-small", input=batch)
        all_embeddings.extend([d.embedding for d in resp.data])
        if i + batch_size < len(texts):
            time.sleep(0.1)
    return all_embeddings
