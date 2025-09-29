# -*- coding: utf-8 -*-
import os
import pickle
from io import BytesIO
from typing import List, Dict, Any, Tuple

import streamlit as st
import numpy as np
import pandas as pd

# Third-party deps expected:
#   openai>=1.12.0
#   faiss-cpu>=1.7.4
#   streamlit-authenticator>=0.3.3
#   bcrypt>=4.0.1

# FAISS
try:
    import faiss  # type: ignore
except Exception:
    st.error("FAISS is required. Please add 'faiss-cpu>=1.7.4' to requirements.txt")
    st.stop()

# OpenAI
try:
    from openai import OpenAI
except Exception:
    st.error("OpenAI SDK is required. Please add 'openai>=1.12.0' to requirements.txt")
    st.stop()

# Authentication
try:
    import streamlit_authenticator as stauth
except Exception:
    st.error("Please add 'streamlit-authenticator' and 'bcrypt' to requirements.txt")
    st.stop()

# Local modules (Drive + processors)
try:
    from drive_utils import (
        authenticate_drive,
        list_files_in_folder,
        download_file,
        upload_file,
        format_file_size,
        download_embeddings_from_drive,
        upload_embeddings_to_drive,
    )
except Exception as e:
    st.error("Failed to import drive_utils: %s" % e)
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
    st.error("Failed to import document_processors: %s" % e)
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
    if "auth" not in st.secrets:
        st.error("Authentication is not configured. Please add an [auth] section in secrets.")
        st.stop()

    auth_cfg = st.secrets["auth"]
    cookie_name = auth_cfg.get("cookie_name", "vnatech_auth")
    cookie_key = auth_cfg.get("cookie_key", "change_me")
    cookie_expiry_days = int(auth_cfg.get("cookie_expiry_days", 30))

    users = auth_cfg.get("users", {})
    if not users:
        st.error("No users configured under [auth.users]. Please add at least one user with a bcrypt-hashed password.")
        st.stop()

    names, usernames, hashed_passwords = [], [], []
    for key, u in users.items():
        names.append(u.get("name", key))
        usernames.append(u.get("username", key))
        hashed_passwords.append(u.get("password", ""))

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
            st.success("Signed in as **%s**" % name)
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
        if name.lower().endswith(".pdf") or name.lower().endswith(".pptx"):
            filtered.append(f)
    return filtered


# =========================
# Embeddings Store & FAISS
# =========================
def _try_load_local_index():
    if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(FAISS_INDEX_FILE):
        try:
            with open(EMBEDDINGS_FILE, "rb") as f:
                meta = pickle.load(f)
            index = faiss.read_index(FAISS_INDEX_FILE)
            return index, meta
        except Exception:
            return None, None
    return None, None


def _load_or_pull_cache_from_drive() -> Tuple[Any, List[Dict[str, Any]]]:
    # First try local
    idx, meta = _try_load_local_index()
    if idx is not None and meta is not None:
        return idx, meta

    # If not local, pull from Drive (if exists)
    service = _drive_service()
    folder_id = st.secrets.get("DRIVE_FOLDER_ID")
    paths = download_embeddings_from_drive(service, folder_id, EMBEDDINGS_FILE, FAISS_INDEX_FILE)
    if paths.get("embeddings_path") and paths.get("faiss_path"):
        try:
            with open(EMBEDDINGS_FILE, "rb") as f:
                meta = pickle.load(f)
            idx = faiss.read_index(FAISS_INDEX_FILE)
            return idx, meta
        except Exception:
            pass

    return None, None


def _build_or_load_index(process_all: bool = False) -> Tuple[Any, List[Dict[str, Any]]]:
    # If not forcing rebuild, try load local or from Drive
    if not process_all:
        idx, meta = _load_or_pull_cache_from_drive()
        if idx is not None and meta is not None:
            return idx, meta

    # Build index from scratch
    service = _drive_service()
    drive_folder = st.secrets.get("DRIVE_FOLDER_ID")
    files = _list_drive_files()

    all_vectors = []
    all_meta: List[Dict[str, Any]] = []

    progress = st.progress(0.0, text="Processing documents...")
    n = max(len(files), 1)
    for i, f in enumerate(files, start=1):
        file_id = f["id"]
        file_name = f["name"]
        progress.progress(i / n, text="Downloading %s" % file_name)

        # Download
        content: BytesIO = download_file(service, file_id)

        # Parse
        try:
            if file_name.lower().endswith(".pdf"):
                text, meta = process_pdf(content)
            elif file_name.lower().endswith(".pptx"):
                text, meta = process_pptx(content)
            else:
                continue
        except Exception as e:
            st.warning("Failed to parse '%s': %s" % (file_name, e))
            continue

        # Chunk
        chunks = chunk_text_smart(text, meta, chunk_size=1000, chunk_overlap=200)

        # Embeddings
        texts = [c["text"] for c in chunks]
        try:
            vecs = get_embeddings(texts, batch_size=100)
        except Exception as e:
            st.error("Embedding failed for %s: %s" % (file_name, e))
            continue

        # Collect
        for j, c in enumerate(chunks):
            all_vectors.append(vecs[j])
            row = {"file_id": file_id, "file_name": file_name}
            row.update(c)
            all_meta.append(row)

    if not all_vectors:
        st.error("No embeddings were created. Please check your Drive folder and parsers.")
        st.stop()

    # Build FAISS index (cosine via inner product on L2-normalized vectors)
    mat = np.array(all_vectors, dtype="float32")
    faiss.normalize_L2(mat)
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)

    # Save caches locally
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(all_meta, f)
    faiss.write_index(index, FAISS_INDEX_FILE)

    # Upload caches to Drive (persist across restarts)
    try:
        upload_embeddings_to_drive(service, drive_folder, EMBEDDINGS_FILE, FAISS_INDEX_FILE)
    except Exception as e:
        st.info("Upload cache to Drive skipped or failed: %s" % e)

    return index, all_meta


# =========================
# Retrieval & Answering
# =========================
def _embed_query(client: OpenAI, query: str) -> np.ndarray:
    resp = client.embeddings.create(model=EMBEDDINGS_FILE and "text-embedding-3-small", input=[query])
    v = np.array(resp.data[0].embedding, dtype="float32")
    v = v / np.linalg.norm(v)
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
        loc = "%s %s | Chunk %s/%s" % (
            str(c.get("section_type", "?")).title(),
            c.get("section_number", "?"),
            c.get("chunk_index", 0) + 1,
            c.get("total_chunks", "?"),
        )
        header = "[%s] · %s · sim=%.3f" % (c["file_name"], loc, c["similarity"])
        text = c["text"]
        blocks.append(header + "\n" + text)
    return "\n\n---\n\n".join(blocks)


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
        {"role": "user", "content": "Câu hỏi: %s\n\nNguồn tham chiếu:\n%s" % (question, context)},
    ]
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
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
        st.write("- **Embeddings**: `%s`" % EMBEDDINGS_FILE)
        st.write("- **FAISS index**: `%s`" % FAISS_INDEX_FILE)
        st.write("- **Số chunk**: %s" % (len(meta) if meta else 0))
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
    try:
        files = _list_drive_files()
    except Exception as e:
        st.sidebar.error("Lỗi liệt kê Drive: %s" % e)
        files = []

    if files:
        st.sidebar.subheader("Tài liệu trong Drive")
        for f in files[:100]:
            st.sidebar.caption("• %s (%s)" % (f["name"], format_file_size(f.get("size", ""))))


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

        with st.spinner("Đang tạo câu trả lời..."):
            answer = _ask_llm(client, question, results)

        st.markdown("### Kết quả")
        st.write(answer)

        st.markdown("### Nguồn tham chiếu")
        df = pd.DataFrame([
            {
                "file_name": r["file_name"],
                "section": "%s %s" % (r.get("section_type","?"), r.get("section_number","?")),
                "chunk": "%s/%s" % (r.get("chunk_index",0)+1, r.get("total_chunks","?")),
                "similarity": round(r["similarity"], 3),
                "words": r.get("word_count", None),
            }
            for r in results
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)

        with st.expander("Xem chi tiết các đoạn trích"):
            for i, c in enumerate(results, start=1):
                st.markdown("**%d. %s** — %s %s · Chunk %s/%s · sim=%.3f" % (
                    i,
                    c["file_name"],
                    str(c.get("section_type","?")).title(),
                    c.get("section_number","?"),
                    c.get("chunk_index",0)+1,
                    c.get("total_chunks","?"),
                    c["similarity"],
                ))
                txt = c["text"]
                if len(txt) > 1200:
                    txt = txt[:1200] + "..."
                st.code(txt, language="markdown")
                st.markdown('---')


if __name__ == "__main__":
    main()
