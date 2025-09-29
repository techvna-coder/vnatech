import streamlit as st
import OpenAI
import numpy as np
import pandas as pd
import pickle
import os
import json
from typing import List, Dict, Any
import tiktoken
from io import BytesIO

# Import our custom modules
from drive_utils import authenticate_drive, list_files_in_folder, download_file
from document_processors import process_pdf, process_pptx, chunk_text, get_embeddings

# Configure OpenAI
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Constants
DRIVE_FOLDER_ID = st.secrets["DRIVE_FOLDER_ID"]
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 10

@st.cache_data
def load_cached_embeddings(file_id: str) -> Dict[str, Any]:
    """Load cached embeddings if they exist."""
    cache_file = f"{file_id}.pkl"
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            st.warning(f"Failed to load cache for {file_id}: {e}")
    return None

def save_embeddings_cache(file_id: str, data: Dict[str, Any]):
    """Save embeddings to cache."""
    cache_file = f"{file_id}.pkl"
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        st.error(f"Failed to save cache for {file_id}: {e}")

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_relevant_chunks(query: str, embeddings_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Retrieve the most relevant chunks for a query."""
    # Get query embedding
    query_embedding = get_embeddings([query])[0]
    
    # Calculate similarities
    similarities = []
    for i, chunk_embedding in enumerate(embeddings_data['embeddings']):
        similarity = cosine_similarity(query_embedding, np.array(chunk_embedding))
        similarities.append({
            'chunk_idx': i,
            'similarity': similarity,
            'text': embeddings_data['chunks'][i]
        })
    
    # Sort by similarity and return top K
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    return similarities[:TOP_K]

def generate_answer(query: str, relevant_chunks: List[Dict[str, Any]]) -> str:
    """Generate answer using OpenAI with retrieved chunks as context."""
    context = "\n\n".join([chunk['text'] for chunk in relevant_chunks])
    
    prompt = f"""Based on the following context, please answer the question. If the answer cannot be found in the context, please say so.

Context:
{context}

Question: {query}

Answer:"""
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context. Be precise and cite relevant information from the context when possible."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating answer: {str(e)}"

def main():
    st.set_page_config(
        page_title="VNA Tech RAG App",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 VNA Tech RAG App")
    st.markdown("Query documents from Google Drive using AI-powered search and retrieval.")
    
    # Initialize session state
    if 'drive_service' not in st.session_state:
        st.session_state.drive_service = None
    if 'selected_file' not in st.session_state:
        st.session_state.selected_file = None
    if 'embeddings_data' not in st.session_state:
        st.session_state.embeddings_data = None
    
    # Sidebar for file selection
    with st.sidebar:
        st.header("📁 Document Selection")
        
        # Authenticate with Google Drive
        if st.session_state.drive_service is None:
            with st.spinner("Connecting to Google Drive..."):
                try:
                    st.session_state.drive_service = authenticate_drive()
                    st.success("✅ Connected to Google Drive")
                except Exception as e:
                    st.error(f"❌ Failed to connect to Google Drive: {e}")
                    return
        
        # List files in folder
        try:
            files = list_files_in_folder(st.session_state.drive_service, DRIVE_FOLDER_ID)
            if not files:
                st.warning("No files found in the specified folder.")
                return
            
            file_options = {f"{file['name']} ({file['mimeType'].split('.')[-1]})": file for file in files}
            selected_file_name = st.selectbox(
                "Select a document:",
                options=list(file_options.keys()),
                key="file_selector"
            )
            
            if selected_file_name:
                st.session_state.selected_file = file_options[selected_file_name]
                
        except Exception as e:
            st.error(f"Error listing files: {e}")
            return
    
    # Main panel
    if st.session_state.selected_file:
        file = st.session_state.selected_file
        st.subheader(f"📄 Selected Document: {file['name']}")
        
        # Process document and load/create embeddings
        col1, col2 = st.columns([3, 1])
        
        with col2:
            if st.button("🔄 Process Document", type="primary"):
                process_document(file)
        
        with col1:
            st.markdown(f"**File ID:** `{file['id']}`")
            st.markdown(f"**Type:** {file['mimeType']}")
        
        # Check if embeddings are loaded
        if st.session_state.embeddings_data is None:
            # Try to load from cache
            cached_data = load_cached_embeddings(file['id'])
            if cached_data:
                st.session_state.embeddings_data = cached_data
                st.success(f"✅ Loaded cached embeddings ({len(cached_data['chunks'])} chunks)")
            else:
                st.info("👆 Click 'Process Document' to extract and embed text content.")
                return
        
        # Query interface
        st.markdown("---")
        st.subheader("❓ Ask a Question")
        
        query = st.text_input(
            "Enter your question:",
            placeholder="What is this document about?",
            key="query_input"
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            search_button = st.button("🔍 Search", type="primary")
        
        if search_button and query:
            with st.spinner("Searching for relevant information..."):
                # Retrieve relevant chunks
                relevant_chunks = retrieve_relevant_chunks(query, st.session_state.embeddings_data)
                
                # Generate answer
                answer = generate_answer(query, relevant_chunks)
                
                # Display results
                st.markdown("---")
                st.subheader("🎯 Answer")
                st.markdown(answer)
                
                # Show relevant chunks
                with st.expander("📋 Retrieved Context Chunks", expanded=False):
                    for i, chunk in enumerate(relevant_chunks[:5]):  # Show top 5
                        st.markdown(f"**Chunk {i+1}** (Similarity: {chunk['similarity']:.3f})")
                        st.markdown(f"```\n{chunk['text'][:500]}{'...' if len(chunk['text']) > 500 else ''}\n```")
                        st.markdown("---")
    else:
        st.info("👈 Please select a document from the sidebar to get started.")

def process_document(file: Dict[str, Any]):
    """Process the selected document and create embeddings."""
    file_id = file['id']
    file_name = file['name']
    
    with st.spinner(f"Processing {file_name}..."):
        try:
            # Download file
            file_content = download_file(st.session_state.drive_service, file_id)
            
            # Extract text based on file type
            if file['mimeType'] == 'application/pdf':
                text = process_pdf(file_content)
            elif file['mimeType'] == 'application/vnd.openxmlformats-officedocument.presentationml.presentation':
                text = process_pptx(file_content)
            else:
                st.error(f"Unsupported file type: {file['mimeType']}")
                return
            
            if not text.strip():
                st.error("No text could be extracted from the document.")
                return
            
            # Chunk the text
            chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
            st.info(f"Created {len(chunks)} text chunks")
            
            # Generate embeddings
            with st.spinner("Generating embeddings..."):
                embeddings = get_embeddings(chunks)
            
            # Prepare data for caching
            embeddings_data = {
                'file_id': file_id,
                'file_name': file_name,
                'chunks': chunks,
                'embeddings': embeddings,
                'chunk_size': CHUNK_SIZE,
                'chunk_overlap': CHUNK_OVERLAP
            }
            
            # Save to cache and session state
            save_embeddings_cache(file_id, embeddings_data)
            st.session_state.embeddings_data = embeddings_data
            
            st.success(f"✅ Successfully processed {file_name} ({len(chunks)} chunks, {len(embeddings)} embeddings)")
            
        except Exception as e:
            st.error(f"Error processing document: {e}")

if __name__ == "__main__":
    main()
