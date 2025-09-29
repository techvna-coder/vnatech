# -*- coding: utf-8 -*-
import os
import pickle
from io import BytesIO
from typing import List, Dict, Any, Tuple

import streamlit as st
import numpy as np
import pandas as pd

# Deps expected:
#   bcrypt>=4.0.1
#   faiss-cpu>=1.7.4
#   openai>=1.12.0
#   google-api-python-client>=2.129.0
#   google-auth>=2.29.0
#   google-auth-httplib2>=0.2.0
#   streamlit>=1.31.0

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

# Bcrypt (custom auth)
try:
    import bcrypt
except Exception:
    st.error("Please add 'bcrypt>=4.0.1' to requirements.txt")
    st.stop()

# Local modules (Drive + processors)
try:
    from drive_utils import (
        authenticate_drive,
        list_files_in_folder,
        download_file,
        format_file_size,
        download_embeddings_from_drive,   # keep: chỉ tải về, KHÔNG upload
        # upload_file,                    # removed (không dùng)
        # upload_embeddings_to_drive,     # removed (không dùng)
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

st.set_page_config(page_title="VNA Tech RAG", layout="wide")


# =========================
# Custom Authentication (bcrypt + secrets)
# =========================
def _load_credentials_from_secrets() -> Dict[str, Dict[str, str]]:
    """Read users from secrets [auth.users.*] and return {username: {name, password_hash}}"""
    if "auth" not in st.secrets:
        raise RuntimeError("Missing [auth] in secrets.")
    users = st.secrets["auth"].get("users", {})
    creds = {}
    for _, u in users.items():
        uname = u.get("username")
        pwd = u.get("password")
        name = u.get("name", uname)
        if uname and pwd:
            creds[uname] = {"name": name, "password": pwd}
    if not creds:
        raise RuntimeError("No valid users under [auth.users].")
    return creds

def _verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False

def login_gate() -> Tuple[bool, str, str]:
    """Render a simple login form and verify against bcrypt hashes in secrets.
    Returns (ok, username, name).
    """
    try:
        creds = _load_credentials_from_secrets()
    except Exception as e:
        st.error(str(e))
        st.stop()

    if "auth_user" in st.session_state and st.session_state.get("auth_ok"):
        u = st.session_state["auth_user"]
        display_name = st.session_state.get("auth_name", u)
        return True, u, display_name

    with st.form("login_form", clear_on_submit=False):
        st.subheader("Đăng nhập để truy cập VNA Tech RAG")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

    if submitted:
        if username in creds and _verify_password(password, creds[username]["password"]):
            st.session_state["auth_ok"] = True
            st.session_state["auth_user"] = username
            st.session_state["auth_name"] = creds[username]["name"]
            st.success("Đăng nhập thành công.")
            st.rerun()
        else:
            st.error("Sai username hoặc password.")

    return False, "", ""

def logout_button():
    if st.session_state.get("auth_ok"):
        if st.sidebar.button("Sign out"):
            for k in ["auth_ok", "auth_user", "auth_name"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.success("Đã đăng xuất.")
            st.rerun()


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
    # 1) thử local trước
    idx, meta = _try_load_local_index()
    if idx is not None and meta is not None:
        return idx, meta
    # 2) nếu local không có thì thử kéo từ Drive (nếu tồn tại)
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
    # Không ép rebuild thì ưu tiên dùng cache (local → Drive)
    if not process_all:
        idx, meta = _load_or_pull_cache_from_drive()
        if idx is not None and meta is not None:
            return idx, meta

    # Build mới từ tài liệu trên Drive
    service = _drive_service()
    files = _list_drive_files()

    all_vectors = []
    all_meta: List[Dict[str, Any]] = []

    progress = st.progress(0.0, text="Processing documents...")
    n = max(len(files), 1)
    for i, f in enumerate(files, start=1):
        file_id = f["id"]
        file_name = f["name"]
        progress.progress(i / n, text="Downloading %s" % file_name)

        content: BytesIO = download_file(service, file_id)

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

        chunks = chunk_text_smart(text, meta, chunk_size=1000, chunk_overlap=200)
        texts = [c["text"] for c in chunks]
        try:
            vecs = get_embeddings(texts, batch_size=100)
        except Exception as e:
            st.error("Embedding failed for %s: %s" % (file_name, e))
            continue

        for j, c in enumerate(chunks):
            all_vectors.append(vecs[j])
            row = {"file_id": file_id, "file_name": file_name}
            row.update(c)
            all_meta.append(row)

    if not all_vectors:
        st.error("No embeddings were created. Please check your Drive folder and parsers.")
        st.stop()

    mat = np.array(all_vectors, dtype="float32")
    faiss.normalize_L2(mat)
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)

    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(all_meta, f)
    faiss.write_index(index, FAISS_INDEX_FILE)

    # ⛔️ ĐÃ BỎ: upload cache lên Google Drive để tránh lỗi quota SA
    # (giữ local-only; nếu cần persistence dài hạn, dùng Shared Drive + upload riêng ngoài app)

    return index, all_meta


# =========================
# Retrieval & Answering
# =========================
def _embed_query(client: OpenAI, query: str) -> np.ndarray:
    resp = client.embeddings.create(model="text-embedding-3-small", input=[query])
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
        st.caption("Cache hiện được lưu **local-only** trong phiên chạy (không upload lên Drive).")
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

    logout_button()


def main():
    ok, username, display_name = login_gate()
    if not ok:
        st.stop()

    st.title("VNA Tech Streamlit RAG App")
    st.caption("Truy vấn trực tiếp các tài liệu PDF/PPTX trong Google Drive.")

    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY is missing in secrets.")
        st.stop()
    client = OpenAI(api_key=api_key)

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

st.caption("Sản phẩm thử nghiệm của Ban Kỹ thuật – VNA. Mọi ý kiến đóng góp vui lòng liên hệ Phòng Kỹ thuật Máy bay.")


if __name__ == "__main__":
    main()
