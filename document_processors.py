import streamlit as st
import openai
import PyPDF2
from pptx import Presentation
import tiktoken
from typing import List, Dict, Any
from io import BytesIO
import time

# Initialize tokenizer for text-embedding-3-small
tokenizer = tiktoken.encoding_for_model("text-embedding-3-small")

def process_pdf(file_content: BytesIO) -> str:
    """Extract text from PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(file_content)
        text = ""
        
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Page {page_num + 1} ---\n"
                    text += page_text
            except Exception as e:
                st.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                continue
        
        if not text.strip():
            raise Exception("No text could be extracted from the PDF")
        
        return text.strip()
        
    except Exception as e:
        raise Exception(f"Failed to process PDF: {str(e)}")

def process_pptx(file_content: BytesIO) -> str:
    """Extract text from PowerPoint file."""
    try:
        presentation = Presentation(file_content)
        text = ""
        
        for slide_num, slide in enumerate(presentation.slides, 1):
            slide_text = f"\n--- Slide {slide_num} ---\n"
            
            # Extract text from shapes
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    slide_text += shape.text + "\n"
                
                # Extract text from tables
                if hasattr(shape, "table"):
                    for row in shape.table.rows:
                        row_text = []
                        for cell in row.cells:
                            if cell.text:
                                row_text.append(cell.text.strip())
                        if row_text:
                            slide_text += " | ".join(row_text) + "\n"
            
            # Only add slide if it has content
            if slide_text.strip() != f"--- Slide {slide_num} ---":
                text += slide_text
        
        if not text.strip():
            raise Exception("No text could be extracted from the PowerPoint")
        
        return text.strip()
        
    except Exception as e:
        raise Exception(f"Failed to process PPTX: {str(e)}")

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks based on token count."""
    try:
        # Encode the text
        tokens = tokenizer.encode(text)
        
        if len(tokens) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(tokens):
            # Define the end of the current chunk
            end = min(start + chunk_size, len(tokens))
            
            # Decode the chunk back to text
            chunk_tokens = tokens[start:end]
            chunk_text = tokenizer.decode(chunk_tokens)
            
            # Clean up the chunk
            chunk_text = chunk_text.strip()
            if chunk_text:
                chunks.append(chunk_text)
            
            # Move start position (with overlap)
            if end == len(tokens):
                break
            start = end - chunk_overlap
        
        return chunks
        
    except Exception as e:
        raise Exception(f"Failed to chunk text: {str(e)}")

def get_embeddings(texts: List[str], batch_size: int = 100) -> List[List[float]]:
    """Generate embeddings for a list of texts using OpenAI API."""
    try:
        all_embeddings = []
        
        # Process in batches to avoid API limits
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Add progress indicator for large batches
            if len(texts) > batch_size:
                progress = (i + len(batch)) / len(texts)
                st.progress(progress, text=f"Generating embeddings... {i + len(batch)}/{len(texts)}")
            
            # Generate embeddings for the batch
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=batch
            )
            
            # Extract embeddings
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            
            # Small delay to avoid rate limits
            if i + batch_size < len(texts):
                time.sleep(0.1)
        
        return all_embeddings
        
    except Exception as e:
        raise Exception(f"Failed to generate embeddings: {str(e)}")

def count_tokens(text: str) -> int:
    """Count the number of tokens in a text."""
    try:
        return len(tokenizer.encode(text))
    except Exception as e:
        st.warning(f"Failed to count tokens: {e}")
        return 0

def validate_chunk_size(chunk_size: int, max_tokens: int = 8000) -> bool:
    """Validate that chunk size is appropriate."""
    return 0 < chunk_size <= max_tokens

def get_text_statistics(text: str) -> Dict[str, Any]:
    """Get various statistics about the text."""
    try:
        lines = text.split('\n')
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        words = text.split()
        tokens = tokenizer.encode(text)
        
        return {
            'characters': len(text),
            'characters_no_spaces': len(text.replace(' ', '')),
            'lines': len(lines),
            'paragraphs': len(paragraphs),
            'words': len(words),
            'tokens': len(tokens),
            'avg_words_per_paragraph': len(words) / len(paragraphs) if paragraphs else 0,
            'avg_chars_per_word': len(text.replace(' ', '')) / len(words) if words else 0
        }
        
    except Exception as e:
        st.warning(f"Failed to calculate text statistics: {e}")
        return {}

def preprocess_text(text: str) -> str:
    """Clean and preprocess text before chunking."""
    try:
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove excessive newlines
        text = '\n'.join(line.strip() for line in text.split('\n') if line.strip())
        
        # Remove page break artifacts common in PDFs
        text = text.replace('\f', '\n')
        text = text.replace('\r', '\n')
        
        # Fix common OCR issues
        text = text.replace('—', '-')
        text = text.replace('–', '-')
        text = text.replace('"', '"')
        text = text.replace('"', '"')
        text = text.replace(''', "'")
        text = text.replace(''', "'")
        
        return text.strip()
        
    except Exception as e:
        st.warning(f"Failed to preprocess text: {e}")
        return text

def optimize_chunks_for_retrieval(chunks: List[str], min_chunk_size: int = 100) -> List[str]:
    """Optimize chunks for better retrieval performance."""
    try:
        optimized_chunks = []
        
        for chunk in chunks:
            # Skip very small chunks that might not be meaningful
            if count_tokens(chunk) < min_chunk_size:
                continue
            
            # Preprocess the chunk
            processed_chunk = preprocess_text(chunk)
            
            if processed_chunk:
                optimized_chunks.append(processed_chunk)
        
        return optimized_chunks
        
    except Exception as e:
        st.warning(f"Failed to optimize chunks: {e}")
        return chunks

def extract_metadata_from_text(text: str) -> Dict[str, Any]:
    """Extract useful metadata from the document text."""
    try:
        metadata = {}
        
        # Try to find title (often in first few lines)
        lines = text.split('\n')[:10]
        potential_titles = [line.strip() for line in lines if line.strip() and len(line.strip()) > 10]
        if potential_titles:
            metadata['potential_title'] = potential_titles[0]
        
        # Look for common document patterns
        if 'abstract' in text.lower():
            metadata['has_abstract'] = True
        if 'introduction' in text.lower():
            metadata['has_introduction'] = True
        if 'conclusion' in text.lower():
            metadata['has_conclusion'] = True
        if 'references' in text.lower() or 'bibliography' in text.lower():
            metadata['has_references'] = True
        
        return metadata
        
    except Exception as e:
        st.warning(f"Failed to extract metadata: {e}")
        return {}