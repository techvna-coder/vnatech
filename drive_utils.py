import streamlit as st
import json
import ssl
import certifi
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, HttpError
from typing import List, Dict, Any
from io import BytesIO
import httplib2

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
        service = build('drive', 'v3', credentials=credentials)
        
        return service
        
    except Exception as e:
        raise Exception(f"Failed to authenticate with Google Drive: {str(e)}")

def list_files_in_folder(service, folder_id: str) -> List[Dict[str, Any]]:
    """List all PDF and PPTX files in a Google Drive folder."""
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
        
    except HttpError as e:
        raise Exception(f"HTTP Error {e.resp.status}: {e.error_details}")
    except Exception as e:
        raise Exception(f"Failed to list files in folder: {str(e)}")

def download_file(service, file_id: str) -> BytesIO:
    """Download a file from Google Drive and return as BytesIO object."""
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
        
    except Exception as e:
        raise Exception(f"Failed to download file {file_id}: {str(e)}")

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
