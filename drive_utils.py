import streamlit as st
import json
import ssl
import certifi
import time
import os
from typing import Tuple
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, HttpError
from typing import List, Dict, Any
from io import BytesIO

def authenticate_drive():
    """Authenticate with Google Drive using service account credentials."""
    try:
        # Get service account JSON from secrets
        service_account_json = st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"]
        
        # Parse JSON string to dict
        if isinstance(service_account_json, str):
            service_account_dict = json.loads(service_account_json)
        else:
            service_account_dict = service_account_json
        
        # Fix private key newlines if necessary
        if 'private_key' in service_account_dict:
            service_account_dict['private_key'] = service_account_dict['private_key'].replace('\\n', '\n')
        
        # Create credentials
        credentials = service_account.Credentials.from_service_account_info(
            service_account_dict,
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        
        # Build the Drive service directly with credentials
        service = build('drive', 'v3', credentials=credentials, cache_discovery=False)
        
        return service
        
    except Exception as e:
        raise Exception(f"Failed to authenticate with Google Drive: {str(e)}")

def list_files_in_folder(service, folder_id: str, max_retries: int = 3) -> List[Dict[str, Any]]:
    """List all PDF and PPTX files in a Google Drive folder with retry logic."""
    for attempt in range(max_retries):
        try:
            # Query for PDF and PPTX files in the specified folder
            query = f"'{folder_id}' in parents and (mimeType='application/pdf' or mimeType='application/vnd.openxmlformats-officedocument.presentationml.presentation') and trashed=false"
            
            results = service.files().list(
                q=query,
                fields="files(id, name, mimeType, size, modifiedTime)",
                pageSize=100
            ).execute()
            
            files = results.get('files', [])
            
            formatted_files = []
            for file in files:
                formatted_files.append({
                    'id': file['id'],
                    'name': file['name'],
                    'mimeType': file['mimeType'],
                    'size': file.get('size', 'Unknown'),
                    'modifiedDate': file.get('modifiedTime', 'Unknown')
                })
            
            # Sort files by name
            formatted_files.sort(key=lambda x: x['name'].lower())
            
            return formatted_files
            
        except (ssl.SSLError, ConnectionError) as e:
            if attempt < max_retries - 1:
                st.warning(f"Connection issue (attempt {attempt + 1}/{max_retries}), retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            else:
                raise Exception(f"Failed after {max_retries} attempts: {str(e)}")
        except HttpError as e:
            raise Exception(f"HTTP Error {e.resp.status}: {e.error_details}")
        except Exception as e:
            raise Exception(f"Failed to list files in folder: {str(e)}")

def download_file(service, file_id: str, max_retries: int = 3) -> BytesIO:
    """Download a file from Google Drive and return as BytesIO object with retry logic."""
    for attempt in range(max_retries):
        try:
            # Request file content
            request = service.files().get_media(fileId=file_id)
            
            # Download to BytesIO
            file_content = BytesIO()
            downloader = MediaIoBaseDownload(file_content, request)
            
            done = False
            while not done:
                status, done = downloader.next_chunk()
            
            file_content.seek(0)
            return file_content
            
        except (ssl.SSLError, ConnectionError) as e:
            if attempt < max_retries - 1:
                st.warning(f"Download connection issue (attempt {attempt + 1}/{max_retries}), retrying...")
                time.sleep(2 ** attempt)
                continue
            else:
                raise Exception(f"Failed to download after {max_retries} attempts: {str(e)}")
        except Exception as e:
            raise Exception(f"Failed to download file {file_id}: {str(e)}")


def upload_file_to_drive(service, file_path: str, folder_id: str, file_name: str = None) -> str:
    """Upload a file to Google Drive folder and return the file ID."""
    try:
        from googleapiclient.http import MediaFileUpload
        
        if file_name is None:
            file_name = os.path.basename(file_path)
        
        # Check if file already exists in the folder
        existing_file_id = find_file_by_name(service, folder_id, file_name)
        
        file_metadata = {
            'name': file_name,
            'parents': [folder_id]
        }
        
        media = MediaFileUpload(file_path, resumable=True)
        
        if existing_file_id:
            # Update existing file
            file = service.files().update(
                fileId=existing_file_id,
                media_body=media
            ).execute()
            st.info(f"Updated existing file: {file_name}")
        else:
            # Create new file
            file = service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id, name'
            ).execute()
            st.success(f"Uploaded new file: {file_name}")
        
        return file['id']
        
    except Exception as e:
        raise Exception(f"Failed to upload file {file_name}: {str(e)}")


def find_file_by_name(service, folder_id: str, file_name: str) -> str:
    """Find a file by name in a specific folder. Returns file ID or None."""
    try:
        query = f"name='{file_name}' and '{folder_id}' in parents and trashed=false"
        
        results = service.files().list(
            q=query,
            fields="files(id, name)",
            pageSize=1
        ).execute()
        
        files = results.get('files', [])
        
        if files:
            return files[0]['id']
        return None
        
    except Exception as e:
        st.warning(f"Error searching for file {file_name}: {e}")
        return None


def download_embeddings_from_drive(service, folder_id: str, embeddings_file: str, faiss_file: str) -> Tuple[bool, bool]:
    """
    Download embeddings files from Google Drive if they exist.
    Returns: (embeddings_exists, faiss_exists)
    """
    try:
        # Check for embeddings file
        embeddings_id = find_file_by_name(service, folder_id, embeddings_file)
        faiss_id = find_file_by_name(service, folder_id, faiss_file)
        
        embeddings_exists = False
        faiss_exists = False
        
        if embeddings_id:
            st.info(f"📥 Downloading {embeddings_file} from Google Drive...")
            content = download_file(service, embeddings_id)
            with open(embeddings_file, 'wb') as f:
                f.write(content.read())
            embeddings_exists = True
            st.success(f"✅ Downloaded {embeddings_file}")
        
        if faiss_id:
            st.info(f"📥 Downloading {faiss_file} from Google Drive...")
            content = download_file(service, faiss_id)
            with open(faiss_file, 'wb') as f:
                f.write(content.read())
            faiss_exists = True
            st.success(f"✅ Downloaded {faiss_file}")
        
        return embeddings_exists, faiss_exists
        
    except Exception as e:
        st.error(f"Error downloading embeddings from Drive: {e}")
        return False, False


def upload_embeddings_to_drive(service, folder_id: str, embeddings_file: str, faiss_file: str):
    """Upload embeddings files to Google Drive."""
    try:
        st.info("📤 Uploading embeddings to Google Drive...")
        
        if os.path.exists(embeddings_file):
            upload_file_to_drive(service, embeddings_file, folder_id, os.path.basename(embeddings_file))
        
        if os.path.exists(faiss_file):
            upload_file_to_drive(service, faiss_file, folder_id, os.path.basename(faiss_file))
        
        st.success("✅ Embeddings saved to Google Drive!")
        
    except Exception as e:
        st.error(f"Error uploading embeddings to Drive: {e}")

def get_file_metadata(service, file_id: str) -> Dict[str, Any]:
    """Get metadata for a specific file."""
    try:
        file = service.files().get(
            fileId=file_id,
            fields='id, name, mimeType, size, modifiedTime, createdTime, owners'
        ).execute()
        
        return {
            'id': file['id'],
            'name': file['name'],
            'mimeType': file['mimeType'],
            'size': file.get('size', 'Unknown'),
            'modifiedDate': file.get('modifiedTime', 'Unknown'),
            'createdDate': file.get('createdTime', 'Unknown'),
            'owners': file.get('owners', [])
        }
        
    except Exception as e:
        raise Exception(f"Failed to get metadata for file {file_id}: {str(e)}")

def check_folder_access(service, folder_id: str) -> bool:
    """Check if the service account has access to the specified folder."""
    try:
        folder = service.files().get(
            fileId=folder_id,
            fields='name, mimeType'
        ).execute()
        
        # Check if it's actually a folder
        if folder['mimeType'] != 'application/vnd.google-apps.folder':
            raise Exception(f"ID {folder_id} is not a folder")
        
        return True
        
    except Exception as e:
        raise Exception(f"Cannot access folder {folder_id}: {str(e)}")

def validate_service_account_json(service_account_json: str) -> Dict[str, Any]:
    """Validate and parse service account JSON."""
    try:
        if isinstance(service_account_json, str):
            service_account_dict = json.loads(service_account_json)
        else:
            service_account_dict = service_account_json
        
        # Check required fields
        required_fields = ['type', 'project_id', 'private_key_id', 'private_key', 'client_email', 'client_id']
        for field in required_fields:
            if field not in service_account_dict:
                raise ValueError(f"Missing required field: {field}")
        
        return service_account_dict
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {str(e)}")
    except Exception as e:
        raise ValueError(f"Service account validation failed: {str(e)}")

def get_supported_mime_types() -> List[str]:
    """Return list of supported MIME types."""
    return [
        'application/pdf',
        'application/vnd.openxmlformats-officedocument.presentationml.presentation'
    ]

def is_supported_file_type(mime_type: str) -> bool:
    """Check if file type is supported."""
    return mime_type in get_supported_mime_types()

def format_file_size(size_bytes: str) -> str:
    """Format file size in human-readable format."""
    try:
        size = int(size_bytes)
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
    except (ValueError, TypeError):
        return "Unknown"
