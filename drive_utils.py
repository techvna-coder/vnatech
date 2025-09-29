import streamlit as st
import json
from pydrive2.auth import ServiceAuth
from pydrive2.drive import GoogleDrive
from typing import List, Dict, Any
from io import BytesIO

def authenticate_drive() -> GoogleDrive:
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
        
        # Authenticate using service account
        auth = ServiceAuth(keyfile_dict=service_account_dict)
        auth.authenticate()
        
        # Create GoogleDrive instance
        drive = GoogleDrive(auth)
        
        return drive
        
    except Exception as e:
        raise Exception(f"Failed to authenticate with Google Drive: {str(e)}")

def list_files_in_folder(drive: GoogleDrive, folder_id: str) -> List[Dict[str, Any]]:
    """List all PDF and PPTX files in a Google Drive folder."""
    try:
        # Query for PDF and PPTX files in the specified folder
        query = f"'{folder_id}' in parents and (mimeType='application/pdf' or mimeType='application/vnd.openxmlformats-officedocument.presentationml.presentation') and trashed=false"
        
        file_list = drive.ListFile({'q': query}).GetList()
        
        files = []
        for file in file_list:
            files.append({
                'id': file['id'],
                'name': file['title'],
                'mimeType': file['mimeType'],
                'size': file.get('fileSize', 'Unknown'),
                'modifiedDate': file.get('modifiedDate', 'Unknown')
            })
        
        # Sort files by name
        files.sort(key=lambda x: x['name'].lower())
        
        return files
        
    except Exception as e:
        raise Exception(f"Failed to list files in folder: {str(e)}")

def download_file(drive: GoogleDrive, file_id: str) -> BytesIO:
    """Download a file from Google Drive and return as BytesIO object."""
    try:
        # Get file metadata
        file = drive.CreateFile({'id': file_id})
        
        # Download file content
        file_content = BytesIO()
        file.GetContentIOBuffer(file_content)
        file_content.seek(0)
        
        return file_content
        
    except Exception as e:
        raise Exception(f"Failed to download file {file_id}: {str(e)}")

def get_file_metadata(drive: GoogleDrive, file_id: str) -> Dict[str, Any]:
    """Get metadata for a specific file."""
    try:
        file = drive.CreateFile({'id': file_id})
        file.FetchMetadata()
        
        return {
            'id': file['id'],
            'name': file['title'],
            'mimeType': file['mimeType'],
            'size': file.get('fileSize', 'Unknown'),
            'modifiedDate': file.get('modifiedDate', 'Unknown'),
            'createdDate': file.get('createdDate', 'Unknown'),
            'owners': file.get('owners', []),
            'lastModifyingUser': file.get('lastModifyingUser', {})
        }
        
    except Exception as e:
        raise Exception(f"Failed to get metadata for file {file_id}: {str(e)}")

def check_folder_access(drive: GoogleDrive, folder_id: str) -> bool:
    """Check if the service account has access to the specified folder."""
    try:
        folder = drive.CreateFile({'id': folder_id})
        folder.FetchMetadata(['title', 'mimeType'])
        
        # Check if it's actually a folder
        if folder['mimeType'] != 'application/vnd.google-apps.folder':
            raise Exception(f"ID {folder_id} is not a folder")
        
        return True
        
    except Exception as e:
        raise Exception(f"Cannot access folder {folder_id}: {str(e)}")

# Additional utility functions for better error handling and logging

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