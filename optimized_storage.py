# -*- coding: utf-8 -*-
"""
Persistent storage cho embeddings trên Streamlit Cloud
Không cần Google Drive upload, chỉ cache local trong repo
"""
import os
import pickle
import streamlit as st
import faiss
from typing import Tuple, List, Dict, Any

# Persistent cache directory trong Streamlit Cloud
# Files ở đây tồn tại lâu hơn và ít bị xóa
CACHE_DIR = os.path.join(os.getcwd(), ".cache")
EMBEDDINGS_FILE = os.path.join(CACHE_DIR, "embeddings_meta.pkl")
FAISS_INDEX_FILE = os.path.join(CACHE_DIR, "faiss_index.bin")

# Tạo thư mục cache nếu chưa có
os.makedirs(CACHE_DIR, exist_ok=True)


def save_embeddings_local(metadata: Dict[str, Any], index) -> bool:
    """Lưu embeddings và FAISS index vào local filesystem."""
    try:
        with open(EMBEDDINGS_FILE, 'wb') as f:
            pickle.dump(metadata, f)
        
        faiss.write_index(index, FAISS_INDEX_FILE)
        
        # Tạo file marker để biết là đã build xong
        marker_file = os.path.join(CACHE_DIR, ".last_build")
        import time
        with open(marker_file, 'w') as f:
            f.write(str(time.time()))
        
        st.success(f"✅ Đã lưu embeddings tại: {CACHE_DIR}")
        return True
    except Exception as e:
        st.error(f"Lỗi lưu embeddings: {e}")
        return False


def load_embeddings_local() -> Tuple[Any, List[Dict[str, Any]]]:
    """Load embeddings từ local filesystem."""
    if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(FAISS_INDEX_FILE):
        try:
            with open(EMBEDDINGS_FILE, 'rb') as f:
                metadata = pickle.load(f)
            
            index = faiss.read_index(FAISS_INDEX_FILE)
            
            # Kiểm tra thời gian build
            marker_file = os.path.join(CACHE_DIR, ".last_build")
            if os.path.exists(marker_file):
                with open(marker_file, 'r') as f:
                    build_time = float(f.read())
                import datetime
                dt = datetime.datetime.fromtimestamp(build_time)
                st.info(f"📦 Cache được tạo lúc: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
            
            return index, metadata
        except Exception as e:
            st.warning(f"Không thể load cache: {e}")
            return None, None
    return None, None


def get_cache_info() -> Dict[str, Any]:
    """Lấy thông tin về cache hiện tại."""
    info = {
        "exists": False,
        "embeddings_size": 0,
        "faiss_size": 0,
        "total_size": 0,
        "last_build": None
    }
    
    if os.path.exists(EMBEDDINGS_FILE):
        info["exists"] = True
        info["embeddings_size"] = os.path.getsize(EMBEDDINGS_FILE)
    
    if os.path.exists(FAISS_INDEX_FILE):
        info["faiss_size"] = os.path.getsize(FAISS_INDEX_FILE)
    
    info["total_size"] = info["embeddings_size"] + info["faiss_size"]
    
    marker_file = os.path.join(CACHE_DIR, ".last_build")
    if os.path.exists(marker_file):
        try:
            with open(marker_file, 'r') as f:
                build_time = float(f.read())
            import datetime
            info["last_build"] = datetime.datetime.fromtimestamp(build_time)
        except:
            pass
    
    return info


def clear_cache():
    """Xóa toàn bộ cache."""
    files = [EMBEDDINGS_FILE, FAISS_INDEX_FILE, os.path.join(CACHE_DIR, ".last_build")]
    for f in files:
        if os.path.exists(f):
            try:
                os.remove(f)
            except:
                pass
    st.success("✅ Đã xóa cache")


def format_bytes(size: int) -> str:
    """Format bytes thành đơn vị dễ đọc."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


# ========================================
# OPTIONAL: Backup/Restore từ Google Drive
# ========================================

def try_restore_from_drive(drive_service, folder_id: str) -> bool:
    """
    Thử download embeddings từ Drive để khôi phục (nếu có).
    Chỉ dùng Drive để BACKUP/RESTORE, không dùng làm primary storage.
    """
    try:
        from drive_utils import download_embeddings_from_drive
        
        st.info("🔄 Đang thử khôi phục cache từ Google Drive...")
        paths = download_embeddings_from_drive(
            drive_service, 
            folder_id, 
            os.path.basename(EMBEDDINGS_FILE),
            os.path.basename(FAISS_INDEX_FILE)
        )
        
        # Di chuyển file đã download vào CACHE_DIR
        if paths.get("embeddings_path"):
            import shutil
            shutil.move(paths["embeddings_path"], EMBEDDINGS_FILE)
        
        if paths.get("faiss_path"):
            import shutil
            shutil.move(paths["faiss_path"], FAISS_INDEX_FILE)
        
        if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(FAISS_INDEX_FILE):
            st.success("✅ Đã khôi phục cache từ Google Drive!")
            return True
    except Exception as e:
        st.warning(f"Không thể khôi phục từ Drive: {e}")
    
    return False


def try_backup_to_drive(drive_service, folder_id: str) -> bool:
    """
    Thử backup embeddings lên Drive (optional, không bắt buộc).
    Nếu thất bại cũng không sao vì đã có local cache.
    """
    if not os.path.exists(EMBEDDINGS_FILE) or not os.path.exists(FAISS_INDEX_FILE):
        return False
    
    try:
        from drive_utils import upload_embeddings_to_drive
        
        st.info("📤 Đang backup cache lên Google Drive...")
        
        # NOTE: Sẽ FAIL nếu không dùng Shared Drive
        # Nhưng app vẫn hoạt động bình thường vì đã có local cache
        upload_embeddings_to_drive(
            drive_service, 
            folder_id, 
            EMBEDDINGS_FILE,
            FAISS_INDEX_FILE
        )
        
        st.success("✅ Đã backup cache lên Google Drive!")
        return True
    except Exception as e:
        st.info(f"ℹ️ Backup lên Drive thất bại (không ảnh hưởng app): {e}")
        return False


# ========================================
# Demo sidebar để quản lý cache
# ========================================

def cache_management_sidebar():
    """Render sidebar để quản lý cache."""
    st.sidebar.subheader("💾 Quản lý Cache")
    
    info = get_cache_info()
    
    if info["exists"]:
        st.sidebar.success("✅ Cache đã tồn tại")
        st.sidebar.text(f"📦 Embeddings: {format_bytes(info['embeddings_size'])}")
        st.sidebar.text(f"🔍 FAISS Index: {format_bytes(info['faiss_size'])}")
        st.sidebar.text(f"📊 Tổng cộng: {format_bytes(info['total_size'])}")
        
        if info["last_build"]:
            st.sidebar.text(f"🕒 Build lúc: {info['last_build'].strftime('%Y-%m-%d %H:%M')}")
    else:
        st.sidebar.warning("⚠️ Chưa có cache")
    
    st.sidebar.divider()
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🔄 Rebuild", use_container_width=True):
            st.session_state["force_rebuild"] = True
            st.rerun()
    
    with col2:
        if st.button("🗑️ Xóa Cache", use_container_width=True, type="secondary"):
            clear_cache()
            st.rerun()
    
    # Optional: Backup/Restore buttons
    if st.sidebar.checkbox("🔧 Tùy chọn nâng cao"):
        if st.sidebar.button("📥 Khôi phục từ Drive"):
            try:
                from drive_utils import authenticate_drive
                service = authenticate_drive()
                folder_id = st.secrets.get("DRIVE_FOLDER_ID")
                try_restore_from_drive(service, folder_id)
            except Exception as e:
                st.sidebar.error(f"Lỗi: {e}")
        
        if st.sidebar.button("📤 Backup lên Drive"):
            try:
                from drive_utils import authenticate_drive
                service = authenticate_drive()
                folder_id = st.secrets.get("DRIVE_FOLDER_ID")
                try_backup_to_drive(service, folder_id)
            except Exception as e:
                st.sidebar.info(f"Backup thất bại (bình thường nếu không dùng Shared Drive)")
