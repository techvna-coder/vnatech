# Write a complete drive_utils.py with robust upload (update-or-create), listing, download,
# and authentication via Service Account from Streamlit secrets.
from pathlib import Path

drive_utils_code = r"""
import io
import os
import time
from typing import List, Dict, Any, Optional

import streamlit as st

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

# ============================
# Authentication
# ============================

# Default scopes:
# - drive: full access (upload/update/list)
# - drive.readonly: safe fallback for read-only operations (not used by default)
SCOPES = [
    "https://www.googleapis.com/auth/drive",
]

@st.cache_resource(show_spinner=False)
def authenticate_drive():
    \"\"\"Authenticate to Google Drive using a Service Account from Streamlit secrets.
    Expected in .streamlit/secrets.toml:
        GOOGLE_SERVICE_ACCOUNT_JSON = \"\"\"{ ... }\"\"\"  # JSON string
    \"\"\"
    sa_json = st.secrets.get("GOOGLE_SERVICE_ACCOUNT_JSON", None)
    if not sa_json:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON is missing in secrets.")

    # Accept either a JSON string or a dict-like
    if isinstance(sa_json, str):
        import json
        try:
            sa_info = json.loads(sa_json)
        except Exception as e:
            raise RuntimeError(f"Invalid GOOGLE_SERVICE_ACCOUNT_JSON: {e}")
    elif isinstance(sa_json, dict):
        sa_info = sa_json
    else:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON must be a JSON string or dict.")

    # Normalize private_key newlines if provided as escaped \\n
    if "private_key" in sa_info and isinstance(sa_info["private_key"], str):
        sa_info["private_key"] = sa_info["private_key"].replace("\\\\n", "\n").replace("\\n", "\n")

    creds = service_account.Credentials.from_service_account_info(sa_info, scopes=SCOPES)
    service = build("drive", "v3", credentials=creds, cache_discovery=False)
    return service


# ============================
# Helpers
# ============================

def format_file_size(size_str: Optional[str]) -> str:
    if not size_str:
        return "-"
    try:
        size = int(size_str)
    except Exception:
        return size_str
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.0f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def _retry(fn, max_tries: int = 3, base_delay: float = 0.8):
    last_err = None
    for attempt in range(1, max_tries + 1):
        try:
            return fn()
        except HttpError as e:
            last_err = e
            # exponential backoff
            time.sleep(base_delay * (2 ** (attempt - 1)))
        except Exception as e:
            last_err = e
            time.sleep(base_delay * (2 ** (attempt - 1)))
    if last_err:
        raise last_err


# ============================
# File Listing / Query
# ============================

def list_files_in_folder(service, folder_id: str) -> List[Dict[str, Any]]:
    \"\"\"List non-trashed files (id, name, size, mimeType, modifiedTime) within a folder.\"\"\"
    files: List[Dict[str, Any]] = []
    page_token = None
    q = f"'{folder_id}' in parents and trashed = false"
    fields = "nextPageToken, files(id,name,size,mimeType,modifiedTime)"
    while True:
        def _call():
            return service.files().list(q=q, pageToken=page_token, fields=fields, orderBy="name").execute()
        resp = _retry(_call)
        files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return files


def _find_file_by_name(service, folder_id: str, filename: str) -> Optional[str]:
    safe_name = filename.replace("'", "\\'")
    q = f"'{folder_id}' in parents and name = '{safe_name}' and trashed = false"
    fields = "files(id,name)"
    def _call():
        return service.files().list(q=q, fields=fields, pageSize=1).execute()
    resp = _retry(_call)
    files = resp.get("files", [])
    return files[0]["id"] if files else None


# ============================
# Download
# ============================

def download_file(service, file_id: str) -> io.BytesIO:
    \"\"\"Download a Drive file as BytesIO.\"\"\"
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False

    def _step():
        return downloader.next_chunk()

    while not done:
        status, done = _retry(_step)
        # Optional: could display progress in Streamlit if desired

    fh.seek(0)
    return fh


# ============================
# Upload (Update-or-Create)
# ============================

def upload_file(service, folder_id: str, local_path: str, mime_type: str = "application/octet-stream") -> str:
    \"\"\"Upload a local file to Drive folder. If a file with the same name exists in that folder, update it.
    Returns the Drive file id.
    \"\"\"
    filename = os.path.basename(local_path)
    file_id = _find_file_by_name(service, folder_id, filename)
    media = MediaFileUpload(local_path, mimetype=mime_type, resumable=True)

    if file_id:
        # Try update existing file
        try:
            def _call():
                return service.files().update(fileId=file_id, media_body=media, fields="id").execute()
            _retry(_call)
            return file_id
        except HttpError:
            # Fall back to create if update is not permitted
            pass

    # Create new file in the folder
    file_metadata = {"name": filename, "parents": [folder_id]}
    def _create():
        return service.files().create(body=file_metadata, media_body=media, fields="id").execute()
    created = _retry(_create)
    return created["id"]
"""
Path("/mnt/data/drive_utils.py").write_text(drive_utils_code, encoding="utf-8")
print("Created updated drive_utils.py at /mnt/data")
