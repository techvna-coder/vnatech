import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import json
import time
from typing import List, Dict, Any
import tiktoken
from io import BytesIO

# Try importing required modules
try:
    from openai import OpenAI
except ImportError:
    st.error("OpenAI library not installed. Please add 'openai>=1.12.0' to requirements.txt")
    st.stop()

try:
    import faiss
except ImportError:
    st.error("FAISS library not installed. Please add 'faiss-cpu>=1.7.4' to requirements.txt")
    st.stop()

# Import our custom modules
try:
    from drive_utils import (
        authenticate_drive, 
        list_files_in_folder, 
        download_file,
        download_embeddings_from_drive,
        upload_embeddings_to_drive
    )
    from document_processors import process_pdf, process_pptx, chunk_text_smart, get_embeddings
except ImportError as e:
    st.error(f"Failed to import custom modules: {e}")
    st.info("Make sure drive_utils.py and document_processors.py are in the same directory as this app.")
    st.stop()

# Initialize OpenAI client
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception as e:
    st.error(f"Failed to initialize OpenAI client: {e}")
    st.info("Make sure OPENAI_API_KEY is set in your Streamlit secrets.")
    st.stop()

# Constants
DRIVE_FOLDER_ID = st.secrets["DRIVE_FOLDER_ID"]
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 10
EMBEDDINGS_FILE = "embeddings_meta.pkl"
FAISS_INDEX_FILE = "faiss_index.bin"

@st.cache_data
def load_cached_data(_drive_service, folder_id):
    """Load cached embeddings metadata and FAISS index from local or Google Drive."""
    # First try to download from Google Drive
    embeddings_exist, faiss_exist = download_embeddings_from_drive(
        _drive_service, 
        folder_id, 
        EMBEDDINGS_FILE, 
        FAISS_INDEX_FILE
    )
    
    # Then try to load from local files
    if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(FAISS_INDEX_FILE):
        try:
            with open(EMBEDDINGS_FILE, 'rb') as f:
                metadata = pickle.load(f)
            
            index = faiss.read_index(FAISS_INDEX_FILE)
            
            st.success("✅ Loaded embeddings from cache")
            return metadata, index
        except Exception as e:
            st.warning(f"Failed to load cached data: {e}")
    
    return None, None

def save_embeddings_data(metadata: Dict[str, Any], index, drive_service, folder_id: str):
    """Save embeddings metadata and FAISS index to local disk and Google Drive."""
    try:
        # Save locally first
        with open(EMBEDDINGS_FILE, 'wb') as f:
            pickle.dump(metadata, f)
        
        faiss.write_index(index, FAISS_INDEX_FILE)
        
        st.success(f"✅ Saved embeddings locally: {EMBEDDINGS_FILE} and {FAISS_INDEX_FILE}")
        
        # Upload to Google Drive
        upload_embeddings_to_drive(drive_service, folder_id, EMBEDDINGS_FILE, FAISS_INDEX_FILE)
        
    except Exception as e:
        st.error(f"Failed to save embeddings: {e}")

def process_all_documents(drive_service, folder_id: str):
    """Process all documents in the Google Drive folder with enhanced extraction."""
    try:
        # Get all files
        files = list_files_in_folder(drive_service, folder_id)
        
        if not files:
            st.warning("No files found in the folder.")
            return None, None
        
        st.info(f"Found {len(files)} documents to process...")
        
        all_chunks = []
        all_chunk_metadata = []
        document_metadata = []
        
        progress_bar = st.progress(0)
        
        for idx, file in enumerate(files):
            file_id = file['id']
            file_name = file['name']
            
            # Skip embeddings files
            if file_name in [EMBEDDINGS_FILE, FAISS_INDEX_FILE]:
                st.info(f"⏭️ Skipping embeddings file: {file_name}")
                continue
            
            st.write(f"📄 Processing: {file_name}")
            
            try:
                # Download file
                file_content = download_file(drive_service, file_id)
                
                # Extract text with metadata based on file type
                if file['mimeType'] == 'application/pdf':
                    text, doc_metadata = process_pdf(file_content)
                elif file['mimeType'] == 'application/vnd.openxmlformats-officedocument.presentationml.presentation':
                    text, doc_metadata = process_pptx(file_content)
                else:
                    st.warning(f"Skipping unsupported file type: {file_name}")
                    continue
                
                if not text.strip():
                    st.warning(f"No text extracted from {file_name}")
                    continue
                
                # Store document-level metadata
                doc_info = {
                    'file_id': file_id,
                    'file_name': file_name,
                    'file_type': file['mimeType'],
                    'metadata': doc_metadata
                }
                document_metadata.append(doc_info)
                
                # Smart chunking with context preservation
                chunks = chunk_text_smart(text, doc_metadata, CHUNK_SIZE, CHUNK_OVERLAP)
                
                # Store chunks with comprehensive metadata
                for chunk_data in chunks:
                    all_chunks.append(chunk_data['text'])
                    all_chunk_metadata.append({
                        'file_id': file_id,
                        'file_name': file_name,
                        'file_type': file['mimeType'],
                        'chunk_index': chunk_data['chunk_index'],
                        'total_chunks': chunk_data['total_chunks'],
                        'section_type': chunk_data.get('section_type', 'document'),
                        'section_number': chunk_data.get('section_number', 0),
                        'is_complete_section': chunk_data.get('is_complete_section', False),
                        'token_count': chunk_data.get('token_count', 0),
                        'word_count': chunk_data.get('word_count', 0),
                        'has_previous': 'previous_chunk_preview' in chunk_data,
                        'has_next': 'next_chunk_preview' in chunk_data
                    })
                
                st.success(f"✅ {file_name}: {len(chunks)} chunks (smart chunking)")
                
                # Show document insights
                with st.expander(f"📊 Document Insights: {file_name}"):
                    if 'total_pages' in doc_metadata:
                        st.write(f"📄 Pages: {doc_metadata['total_pages']}")
                    if 'total_slides' in doc_metadata:
                        st.write(f"📽️ Slides: {doc_metadata['total_slides']}")
                    if doc_metadata.get('has_sections'):
                        st.write(f"📑 Sections found: {len(doc_metadata.get('sections', []))}")
                    if doc_metadata.get('key_terms'):
                        st.write(f"🔑 Key terms: {', '.join(doc_metadata['key_terms'][:10])}")
                    if doc_metadata.get('has_tables'):
                        st.write("📊 Contains tables")
                    if doc_metadata.get('has_lists'):
                        st.write("📝 Contains lists")
                
            except Exception as e:
                st.error(f"Failed to process {file_name}: {e}")
                continue
            
            progress_bar.progress((idx + 1) / len(files))
        
        if not all_chunks:
            st.error("No chunks were extracted from any document.")
            return None, None
        
        st.info(f"Total chunks across all documents: {len(all_chunks)}")
        
        # Generate embeddings for all chunks
        st.write("🔄 Generating embeddings...")
        embeddings = get_embeddings(all_chunks)
        embeddings_array = np.array(embeddings, dtype='float32')
        
        # Create FAISS index
        st.write("🔍 Creating FAISS index...")
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embeddings_array)
        index.add(embeddings_array)
        
        # Prepare comprehensive metadata
        metadata = {
            'chunks': all_chunks,
            'chunk_metadata': all_chunk_metadata,
            'document_metadata': document_metadata,
            'chunk_size': CHUNK_SIZE,
            'chunk_overlap': CHUNK_OVERLAP,
            'total_files': len(document_metadata),
            'total_chunks': len(all_chunks),
            'processing_timestamp': time.time()
        }
        
        # Save to disk and Google Drive
        save_embeddings_data(metadata, index, drive_service, folder_id)
        
        return metadata, index
        
    except Exception as e:
        st.error(f"Error processing documents: {e}")
        return None, None

def search_documents(query: str, metadata: Dict[str, Any], index) -> List[Dict[str, Any]]:
    """Search for relevant chunks using FAISS with enhanced metadata."""
    try:
        # Get query embedding
        query_embedding = get_embeddings([query])[0]
        query_vector = np.array([query_embedding], dtype='float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_vector)
        
        # Search in FAISS index
        distances, indices = index.search(query_vector, TOP_K)
        
        # Prepare results with comprehensive metadata
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(metadata['chunks']):
                chunk_meta = metadata['chunk_metadata'][idx]
                results.append({
                    'chunk_idx': int(idx),
                    'text': metadata['chunks'][idx],
                    'similarity': float(distances[0][i]),
                    'file_name': chunk_meta['file_name'],
                    'file_id': chunk_meta['file_id'],
                    'file_type': chunk_meta['file_type'],
                    'chunk_number': chunk_meta['chunk_index'] + 1,
                    'total_chunks': chunk_meta['total_chunks'],
                    'section_type': chunk_meta.get('section_type', 'document'),
                    'section_number': chunk_meta.get('section_number', 0),
                    'is_complete_section': chunk_meta.get('is_complete_section', False),
                    'token_count': chunk_meta.get('token_count', 0),
                    'word_count': chunk_meta.get('word_count', 0)
                })
        
        return results
        
    except Exception as e:
        st.error(f"Search failed: {e}")
        return []
                    'file_type': chunk_meta['file_type'],
                    'chunk_number': chunk_meta['chunk_index'] + 1,
                    'total_chunks': chunk_meta['total_chunks'],
                    'section_type': chunk_meta.get('section_type', 'document'),
                    'section_number': chunk_meta.get('section_number', 0),
                    'is_complete_section': chunk_meta.get('is_complete_section', False),
                    'token_count': chunk_meta.get('token_count', 0),
                    'word_count': chunk_meta.get('word_count', 0)
                })
        
        return results
        
    except Exception as e:
        st.error(f"Search failed: {e}")
        return []    'chunk_number': metadata['metadata'][idx]['chunk_idx'] + 1,
                    'total_chunks': metadata['metadata'][idx]['total_chunks']
                })
        
        return results
        
    except Exception as e:
        st.error(f"Search failed: {e}")
        return []

def generate_answer(query: str, relevant_chunks: List[Dict[str, Any]]) -> str:
    """Generate answer using OpenAI with retrieved chunks as context."""
    context_parts = []
    for chunk in relevant_chunks:
        section_info = ""
        if chunk.get('section_type') == 'page':
            section_info = f"Page {chunk['section_number']}"
        elif chunk.get('section_type') == 'slide':
            section_info = f"Slide {chunk['section_number']}"
        
        context_parts.append(
            f"[Source: {chunk['file_name']}, {section_info}, "
            f"Chunk {chunk['chunk_number']}/{chunk['total_chunks']}, "
            f"Similarity: {chunk['similarity']:.3f}]\n{chunk['text']}"
        )
    
    context = "\n\n---\n\n".join(context_parts)
    
    prompt = f"""Based on the following context from multiple documents, please answer the question comprehensively. 

Instructions:
- Provide detailed, accurate answers based on the context
- Always cite which document and section the information comes from
- If information is found across multiple documents, synthesize it coherently
- If the answer cannot be found in the context, clearly state that
- Use specific details and examples from the documents when available

Context:
{context}

Question: {query}

Answer:"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context from multiple documents. Always provide detailed, well-structured answers and cite your sources clearly."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
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
    st.markdown("Query documents from Google Drive using AI-powered multi-document search.")
    
    # Initialize session state
    if 'drive_service' not in st.session_state:
        st.session_state.drive_service = None
    if 'metadata' not in st.session_state:
        st.session_state.metadata = None
    if 'faiss_index' not in st.session_state:
        st.session_state.faiss_index = None
    
    # Sidebar for document management
    with st.sidebar:
        st.header("📁 Document Management")
        
        # Authenticate with Google Drive
        if st.session_state.drive_service is None:
            with st.spinner("Connecting to Google Drive..."):
                try:
                    st.session_state.drive_service = authenticate_drive()
                    st.success("✅ Connected to Google Drive")
                except Exception as e:
                    st.error(f"❌ Failed to connect: {e}")
                    return
        
        # Try to load cached data
        if st.session_state.metadata is None or st.session_state.faiss_index is None:
            cached_metadata, cached_index = load_cached_data(st.session_state.drive_service, DRIVE_FOLDER_ID)
            if cached_metadata and cached_index:
                st.session_state.metadata = cached_metadata
                st.session_state.faiss_index = cached_index
                st.success(f"✅ Loaded {cached_metadata['total_files']} documents ({cached_metadata['total_chunks']} chunks)")
        
        st.markdown("---")
        
        # Process all documents button
        if st.button("🔄 Process All Documents", type="primary", use_container_width=True):
            with st.spinner("Processing all documents..."):
                metadata, index = process_all_documents(st.session_state.drive_service, DRIVE_FOLDER_ID)
                if metadata and index:
                    st.session_state.metadata = metadata
                    st.session_state.faiss_index = index
                    st.rerun()
        
        if st.session_state.metadata:
            st.markdown("---")
            st.subheader("📊 Index Statistics")
            st.metric("Total Documents", st.session_state.metadata['total_files'])
            st.metric("Total Chunks", st.session_state.metadata['total_chunks'])
            
            # Show processing timestamp
            if 'processing_timestamp' in st.session_state.metadata:
                import datetime
                ts = datetime.datetime.fromtimestamp(st.session_state.metadata['processing_timestamp'])
                st.caption(f"Last processed: {ts.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Show document list with detailed info
            with st.expander("📄 Indexed Documents"):
                for doc in st.session_state.metadata.get('document_metadata', []):
                    st.markdown(f"**{doc['file_name']}**")
                    
                    # Count chunks for this document
                    doc_chunks = sum(1 for m in st.session_state.metadata['chunk_metadata'] 
                                   if m['file_id'] == doc['file_id'])
                    st.text(f"  • Chunks: {doc_chunks}")
                    
                    # Show metadata insights
                    meta = doc.get('metadata', {})
                    if 'total_pages' in meta:
                        st.text(f"  • Pages: {meta['total_pages']}")
                    if 'total_slides' in meta:
                        st.text(f"  • Slides: {meta['total_slides']}")
                    if meta.get('key_terms'):
                        st.text(f"  • Key terms: {', '.join(meta['key_terms'][:5])}")
                    
                    st.markdown("---")
    
    # Main panel - Query interface
    if st.session_state.metadata is None or st.session_state.faiss_index is None:
        st.info("👈 Click 'Process All Documents' in the sidebar to get started.")
        st.markdown("""
        ### How it works:
        1. **Connect to Google Drive** - Automatically done
        2. **Process All Documents** - Extract and embed all PDFs and PPTX files
        3. **Ask Questions** - Search across all documents simultaneously
        4. **Cached Results** - Embeddings are saved locally, no need to reprocess
        """)
        return
    
    st.markdown("---")
    st.subheader("❓ Ask Your Question")
    
    query = st.text_input(
        "Enter your question:",
        placeholder="What information do you need from the documents?",
        key="query_input"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    if search_button and query:
        with st.spinner("Searching across all documents..."):
            # Search for relevant chunks
            relevant_chunks = search_documents(query, st.session_state.metadata, st.session_state.faiss_index)
            
            if not relevant_chunks:
                st.warning("No relevant information found.")
                return
            
            # Generate answer
            answer = generate_answer(query, relevant_chunks)
            
            # Display results
            st.markdown("---")
            st.subheader("🎯 Answer")
            st.markdown(answer)
            
            # Show source documents
            st.markdown("---")
            st.subheader("📚 Sources")
            
            # Group by document
            docs_referenced = {}
            for chunk in relevant_chunks:
                doc_name = chunk['file_name']
                if doc_name not in docs_referenced:
                    docs_referenced[doc_name] = []
                docs_referenced[doc_name].append(chunk)
            
            for doc_name, chunks in docs_referenced.items():
                with st.expander(f"📄 {doc_name} ({len(chunks)} relevant chunks)"):
                    for i, chunk in enumerate(chunks[:3], 1):  # Show top 3 per document
                        # Show section info
                        section_info = ""
                        if chunk.get('section_type') == 'page':
                            section_info = f"Page {chunk['section_number']}"
                        elif chunk.get('section_type') == 'slide':
                            section_info = f"Slide {chunk['section_number']}"
                        
                        st.markdown(
                            f"**{section_info} - Chunk {chunk['chunk_number']}/{chunk['total_chunks']}** "
                            f"(Similarity: {chunk['similarity']:.3f}, "
                            f"Words: {chunk.get('word_count', 'N/A')})"
                        )
                        
                        # Show if it's a complete section
                        if chunk.get('is_complete_section'):
                            st.caption("✓ Complete section")
                        
                        st.markdown(f"```\n{chunk['text'][:500]}{'...' if len(chunk['text']) > 500 else ''}\n```")
                        
                        if i < len(chunks):
                            st.markdown("---")

if __name__ == "__main__":
    main()
