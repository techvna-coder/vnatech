# Retry writing the file without the split trick.
from pathlib import Path
code_str = """
import os
import json
import time
import pickle
from io import BytesIO
from typing import List, Dict, Any, Tuple

import streamlit as st
import numpy as np
import pandas as pd

# --- Third‑party libs required ---
# openai>=1.12.0
# faiss-cpu>=1.7.4
# streamlit-authenticator>=0.3.3
# bcrypt>=4.0.1
try:
    import faiss  # type: ignore
except Exception as e:
    st.error("FAISS is required. Please add 'faiss-cpu>=1.7.4' to requirements.txt")
    st.stop()

try:
    from openai import OpenAI
except Exception as e:
    st.error("OpenAI SDK is required. Please add 'openai>=1.12.0' to requirements.txt")
    st.stop()

# Authentication
try:
    import streamlit_authenticator as stauth
except Exception:
    st.error("Authentication library missing. Please add 'streamlit-authenticator' and 'bcrypt' to requirements.txt")
    st.stop()

# Local modules
try:
    from drive_utils import (
        authenticate_drive,
        list_files_in_folder,
        download_file,
        upload_file,
        format_file_size,
    )
except Exception as e:
    st.error(f"Cannot import drive_utils: {e}")
    st.stop()

try:
    from document_processors import (
        process_pdf,
        process_pptx,
        chunk_text_smart,
        get_embeddings,
        count_tokens,
    )
except Exception as e:
    st.error(f"Cannot import document_processors: {e}")
    st.stop()


# =========================
# App Constants & Settings
# =========================
EMBEDDINGS_FILE = "embeddings_meta.pkl"
FAISS_INDEX_FILE = "faiss_index.bin"
TOP_K = 10
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

st.set_page_config(page_title="VNA Tech RAG", layout="wide")


# =========================
# Authentication Gate
# =========================
def auth_gate() -> Tuple[bool, str]:
    \"\"\"
    Enforce login using streamlit-authenticator.
    Expecting a structure in .streamlit/secrets.toml like:
    [auth]
    cookie_name = "vnatech_auth"
    cookie_key = "some_random_signing_key"
    cookie_expiry_days = 30

    [auth.users.user1]
    name = "Alice"
    username = "alice"
    password = "$2b$12$...hashed_bcrypt..."

    [auth.users.user2]
    name = "Bob"
    username = "bob"
    password = "$2b$12$...hashed_bcrypt..."
    \"\"\"
    if "auth" not in st.secrets:
        st.error("Authentication is not configured. Please add an [auth] section in secrets.")
        st.stop()

    auth_cfg = st.secrets["auth"]

    # Cookie/session config
    cookie_name = auth_cfg.get("cookie_name", "vnatech_auth")
    cookie_key = auth_cfg.get("cookie_key", "change_me")
    cookie_expiry_days = int(auth_cfg.get("cookie_expiry_days", 30))

    # Build lists for streamlit-authenticator
    users = auth_cfg.get("users", {})
    if not users:
        st.error("No users configured under [auth.users]. Please add at least one user with a bcrypt-hashed password.")
        st.stop()

    names, usernames, hashed_passwords = [], [], []
    for key, u in users.items():
        names.append(u.get("name", key))
        usernames.append(u.get("username", key))
        hashed_passwords.append(u.get("password", ""))

    # Instantiate authenticator
    authenticator = stauth.Authenticate(
        names=names,
        usernames=usernames,
        passwords=hashed_passwords,
        cookie_name=cookie_name,
        key=cookie_key,
        cookie_expiry_days=cookie_expiry_days,
    )

    name, auth_status, username = authenticator.login(location="main", max_login_attempts=3)

    if auth_status:
        with st.sidebar:
            st.success(f"Signed in as **{name}**")
            authenticator.logout("Sign out", "sidebar")
        return True, username
    elif auth_status is False:
        st.error("Invalid username or password.")
        return False, ""
    else:
        st.info("Please sign in to continue.")
        return False, ""


# =========================
# Google Drive Helpers
# =========================
@st.cache_resource(show_spinner=False)
def _drive_service():
    return authenticate_drive()


def _list_drive_files() -> List[Dict[str, Any]]:
    folder_id = st.secrets.get("DRIVE_FOLDER_ID")
    if not folder_id:
        st.error("DRIVE_FOLDER_ID is missing in secrets.")
        st.stop()
    service = _drive_service()
    files = list_files_in_folder(service, folder_id)
    # Keep only PDFs and PPTX
    filtered = []
    for f in files:
        name = f.get("name", "")
        mime = f.get("mimeType", "")
        if name.lower().endswith(".pdf") or name.lower().endswith(".pptx"):
            filtered.append(f)
    return filtered


# =========================
# Embeddings Store & FAISS
# =========================
def _build_or_load_index(process_all: bool = False) -> Tuple[Any, List[Dict[str, Any]]]:
    \"\"\"
    Returns (faiss_index, metadata_list). If local cache exists, load it.
    If process_all=True, re-build from Drive files and overwrite caches.
    \"\"\"
    if (not process_all) and os.path.exists(EMBEDDINGS_FILE) and os.path.exists(FAISS_INDEX_FILE):
        try:
            with open(EMBEDDINGS_FILE, "rb") as f:
                meta = pickle.load(f)
            index = faiss.read_index(FAISS_INDEX_FILE)
            return index, meta
        except Exception:
            st.warning("Local cache is corrupted. Rebuilding index...")

    # Build index from scratch
    service = _drive_service()
    drive_folder = st.secrets.get("DRIVE_FOLDER_ID")
    files = _list_drive_files()

    all_vectors = []
    all_meta: List[Dict[str, Any]] = []

    progress = st.progress(0.0, text="Processing documents...")
    for i, f in enumerate(files, start=1):
        file_id = f["id"]
        file_name = f["name"]
        mime_type = f.get("mimeType", "")
        progress.progress(i / max(len(files), 1), text=f"Downloading {file_name}")

        # Download
        content: BytesIO = download_file(service, file_id)

        # Parse
        try:
            if file_name.lower().endswith(".pdf"):
                text, meta = process_pdf(content)
            elif file_name.lower().endswith(".pptx"):
                text, meta = process_pptx(content)
            else:
                st.info(f"Skipping unsupported file: {file_name} ({mime_type})")
                continue
        except Exception as e:
            st.warning(f"Failed to parse '{file_name}': {e}")
            continue

        # Chunk
        chunks = chunk_text_smart(text, meta, chunk_size=1000, chunk_overlap=200)

        # Embeddings
        texts = [c["text"] for c in chunks]
        try:
            vecs = get_embeddings(texts, batch_size=100)
        except Exception as e:
            st.error(f"Embedding failed for {file_name}: {e}")
            continue

        # Collect
        for j, c in enumerate(chunks):
            all_vectors.append(vecs[j])
            all_meta.append({
                "file_id": file_id,
                "file_name": file_name,
                **c,  # section info & stats
            })

    if not all_vectors:
        st.error("No embeddings were created. Please check your Drive folder and parsers.")
        st.stop()

    # Build FAISS index (cosine similarity via normalized dot-product)
    mat = np.array(all_vectors, dtype="float32")
    # Normalize vectors
    faiss.normalize_L2(mat)
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)

    # Save caches locally
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(all_meta, f)
    faiss.write_index(index, FAISS_INDEX_FILE)

    # Optionally upload caches to Drive (so they persist across restarts)
    try:
        upload_file(service, drive_folder, EMBEDDINGS_FILE, mime_type="application/octet-stream")
        upload_file(service, drive_folder, FAISS_INDEX_FILE, mime_type="application/octet-stream")
    except Exception as e:
        st.info(f"Upload cache to Drive skipped or failed: {e}")

    return index, all_meta


# =========================
# Retrieval & Answering
# =========================
def _embed_query(client: OpenAI, query: str) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=[query])
    v = np.array(resp.data[0].embedding, dtype="float32")
    v = v / np.linalg.norm(v)  # normalize for cosine
    return v

def _search(index, meta: List[Dict[str, Any]], qvec: np.ndarray, topk: int = TOP_K):
    D, I = index.search(qvec.reshape(1, -1), topk)
    results = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx < 0 or idx >= len(meta):
            continue
        item = meta[idx].copy()
        item["similarity"] = float(score)
        results.append(item)
    return results

def _format_context(chunks: List[Dict[str, Any]]) -> str:
    blocks = []
    for c in chunks:
        loc = f"{c.get('section_type','?').title()} {c.get('section_number','?')} | Chunk {c.get('chunk_index',0)+1}/{c.get('total_chunks','?')}"
        header = f"[{c['file_name']}] · {loc} · sim={c['similarity']:.3f}"
        text = c['text']
        blocks.append(f"{header}\\n{text}")
    return "\\n\\n---\\n\\n".join(blocks)

def _ask_llm(client: OpenAI, question: str, chunks: List[Dict[str, Any]]) -> str:
    context = _format_context(chunks)
    system = (
        "You are a concise technical assistant. "
        "Answer strictly based on the provided context. "
        "If the answer is not in the context, say you cannot find it. "
        "Write in Vietnamese with formal tone."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Câu hỏi: {question}\\n\\nNguồn tham chiếu:\\n{context}"}
    ]
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
    )
    return resp.choices[0].message.content


# =========================
# UI
# =========================
def sidebar_panel(index, meta):
    st.sidebar.header("VNA Tech – RAG")
    with st.sidebar.expander("Bộ nhớ đệm", expanded=True):
        st.write(f"- **Embeddings**: `{EMBEDDINGS_FILE}`")
        st.write(f"- **FAISS index**: `{FAISS_INDEX_FILE}`")
        st.write(f"- **Số chunk**: {len(meta) if meta else 0}")
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Xây dựng lại index", use_container_width=True):
                st.session_state["force_rebuild"] = True
                st.rerun()
        with col2:
            if st.button("Xoá cache (local)", type="secondary", use_container_width=True):
                try:
                    if os.path.exists(EMBEDDINGS_FILE):
                        os.remove(EMBEDDINGS_FILE)
                    if os.path.exists(FAISS_INDEX_FILE):
                        os.remove(FAISS_INDEX_FILE)
                except Exception:
                    pass
                st.success("Đã xoá cache local.")
                st.rerun()

    st.sidebar.divider()
    # Drive file list
    try:
        files = _list_drive_files()
    except Exception as e:
        st.sidebar.error(f"Lỗi liệt kê Drive: {e}")
        files = []

    if files:
        st.sidebar.subheader("Tài liệu trong Drive")
        for f in files[:100]:
            st.sidebar.caption(f"• {f['name']} ({format_file_size(f.get('size',''))})")


def main():
    ok, username = auth_gate()
    if not ok:
        st.stop()

    st.title("VNA Tech Streamlit RAG App")
    st.caption("Truy vấn trực tiếp các tài liệu PDF/PPTX trong Google Drive.")

    # OpenAI client
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY is missing in secrets.")
        st.stop()
    client = OpenAI(api_key=api_key)

    # Load or build index
    force = st.session_state.get("force_rebuild", False)
    index, meta = _build_or_load_index(process_all=force)
    st.session_state["force_rebuild"] = False

    sidebar_panel(index, meta)

    st.subheader("Đặt câu hỏi")
    question = st.text_input("Nhập câu hỏi (tiếng Việt hoặc tiếng Anh):", value="", placeholder="Ví dụ: Tóm tắt nội dung chính của tài liệu X...")
    run = st.button("Truy hồi & Trả lời", type="primary")

    if run:
        if not question.strip():
            st.warning("Vui lòng nhập câu hỏi.")
            st.stop()

        with st.spinner("Đang tính toán..."):
            qvec = _embed_query(client, question)
            results = _search(index, meta, qvec, topk=TOP_K)

        if not results:
            st.info("Không tìm thấy đoạn trích phù hợp.")
            return

        # Hiển thị câu trả lời
        with st.spinner("Đang tạo câu trả lời..."):
            answer = _ask_llm(client, question, results)

        st.markdown("### Kết quả")
        st.write(answer)

        # Bảng nguồn
        st.markdown("### Nguồn tham chiếu")
        df = pd.DataFrame([
            {
                "file_name": r["file_name"],
                "section": f"{r.get('section_type','?')} {r.get('section_number','?')}",
                "chunk": f"{r.get('chunk_index',0)+1}/{r.get('total_chunks','?')}",
                "similarity": round(r["similarity"], 3),
                "words": r.get("word_count", None),
            }
            for r in results
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)

        with st.expander("Xem chi tiết các đoạn trích"):
            for i, c in enumerate(results, start=1):
                st.markdown(f"**{i}. {c['file_name']}** — {c.get('section_type','?').title()} {c.get('section_number','?')} · Chunk {c.get('chunk_index',0)+1}/{c.get('total_chunks','?')} · sim={c['similarity']:.3f}")
                st.code(c['text'][:1200] + ('...' if len(c['text']) > 1200 else ''), language='markdown')
                st.markdown('---')


if __name__ == "__main__":
    main()
"""
Path('/mnt/data/streamlit_app.py').write_text(code_str, encoding='utf-8')
print("Updated /mnt/data/streamlit_app.py created.")
