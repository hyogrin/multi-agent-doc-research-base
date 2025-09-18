"""
Unified File Upload Plugin

This plugin handles the complete file upload workflow:
- File upload request handling
- Document processing and chunking with multiple strategies (semantic-aware, page-based)
- Vector storage in Azure AI Search
- Upload status tracking and notification

All functionalities are consolidated into a single, simple plugin.
"""

import json
import logging
import hashlib
import os
import re
from typing import Dict, List
from pathlib import Path

from semantic_kernel.functions import kernel_function
from config.config import Settings
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import DocumentContentFormat, AnalyzeDocumentRequest, AnalyzeResult
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import AsyncAzureOpenAI
from utils.token_counter import get_token_numbers

logger = logging.getLogger(__name__)


class SemanticAwareTextSplitter:
    """
    Enhanced text splitter that creates chunks based on semantic sections
    and maintains target token count (400±100 tokens) with overlap.
    """
    
    def __init__(self, 
                 target_tokens: int = 400,
                 token_variance: int = 100,
                 overlap_percentage: float = 0.1,
                 model_name: str = "gpt-4"):
        self.target_tokens = target_tokens
        self.min_tokens = target_tokens - token_variance
        self.max_tokens = target_tokens + token_variance
        self.overlap_percentage = overlap_percentage
        self.model_name = model_name
        
        # Semantic separators prioritized by meaning preservation
        self.semantic_separators = [
            # Document structure markers
            "\n\n# ",      # Major sections (H1)
            "\n\n## ",     # Sub-sections (H2)
            "\n\n### ",    # Sub-sub-sections (H3)
            "\n\n#### ",   # Minor sections (H4)
            
            # Content boundaries
            "\n\n---\n",   # Horizontal rules with newlines
            "\n\n***\n",   # Alternative horizontal rules
            "\n\n___\n",   # Another horizontal rule variant
            "\n---\n",     # Simple horizontal rules
            
            # Logical breaks
            "\n\n\n\n",    # Quad line breaks
            "\n\n\n",      # Triple line breaks
            "\n\n",        # Double line breaks (paragraph boundaries)
            
            # Sentence boundaries with context
            ".\n\n",       # Sentence end with paragraph break
            ". \n\n",      # Sentence end with space and paragraph break
            ".\n",         # Sentence end with newline
            ". \n",        # Sentence end with space and newline
            ". ",          # Simple sentence boundaries
            "! ",          # Exclamation boundaries
            "? ",          # Question boundaries
            
            # List and enumeration patterns
            "\n\n- ",      # Bullet points with paragraph break
            "\n\n* ",      # Alternative bullet points
            "\n\n+ ",      # Plus bullet points
            "\n- ",        # Simple bullet points
            "\n* ",        # Simple alternative bullet points
            "\n+ ",        # Simple plus bullet points
            "\n\n1. ",     # Numbered lists with paragraph break
            "\n\n2. ",     # Continue numbered pattern
            "\n1. ",       # Simple numbered lists
            "\n2. ",       # Simple continue numbered
            "\n\t",        # Tab-indented content
            
            # Final fallbacks (더 안전하게)
            "\n",          # Any line break
            ". ",          # Period with space (repeated for emphasis)
            " ",           # Word boundaries
        ]
    
    def split_text_with_semantic_awareness(self, text: str) -> List[str]:
        """
        Split text into semantically meaningful chunks with target token count.
        """
        if not text or not text.strip():
            logger.warning("Empty or whitespace-only text provided for splitting")
            return []
        
        # Clean and normalize text first
        text = self._clean_text(text)
        
        # Log the text format for debugging
        logger.info(f"Text format analysis - Length: {len(text)}, First 200 chars: {repr(text[:200])}")
        
        try:
            # First, try to identify major sections using headers and structure
            sections = self._identify_semantic_sections(text)
            
            if not sections:
                logger.warning("No sections identified, using fallback chunking")
                return self._fallback_chunking(text)
            
            chunks = []
            for section in sections:
                if section and section.strip():  # Ensure section is not empty
                    section_chunks = self._split_section_to_target_tokens(section)
                    chunks.extend(section_chunks)
            
            if not chunks:
                logger.warning("No chunks created from sections, using fallback")
                return self._fallback_chunking(text)
            
            # Apply overlap between chunks
            overlapped_chunks = self._apply_overlap(chunks)
            
            return overlapped_chunks
            
        except Exception as e:
            logger.error(f"Error in semantic splitting: {e}")
            logger.info("Falling back to simple chunking")
            return self._fallback_chunking(text)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text from Document Intelligence."""
        # Remove excessive whitespace
        text = re.sub(r'\n{4,}', '\n\n\n', text)  # Limit consecutive newlines
        text = re.sub(r' {3,}', '  ', text)       # Limit consecutive spaces
        
        # Normalize markdown-like patterns that might come from Document Intelligence
        text = re.sub(r'\*{3,}', '***', text)    # Normalize asterisks
        text = re.sub(r'-{4,}', '---', text)     # Normalize dashes
        text = re.sub(r'_{4,}', '___', text)     # Normalize underscores
        
        return text.strip()
    
    def _identify_semantic_sections(self, text: str) -> List[str]:
        """
        Identify semantic sections based on document structure.
        Enhanced to handle Document Intelligence output better.
        """
        # Look for clear section markers (more flexible patterns)
        section_patterns = [
            r'\n\n#+\s+[^\n]+\n',           # Markdown headers
            r'\n\n\d+\.\s+[^\n]+\n',        # Numbered sections  
            r'\n\n[A-Z][A-Z\s]{2,}[A-Z]\n', # ALL CAPS headers (more specific)
            r'\n\n[^\n]+:\s*\n',            # Colon-ended headers
            r'\n\n[IVX]+\.\s+[^\n]+\n',     # Roman numeral sections
            r'\n\n[A-Z]\.\s+[^\n]+\n',      # Letter sections (A., B., etc.)
        ]
        
        # Find all potential section breaks
        breaks = [0]  # Always start at beginning
        
        for pattern in section_patterns:
            try:
                matches = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)
                for match in matches:
                    break_pos = match.start()
                    if break_pos not in breaks and break_pos > 0:
                        breaks.append(break_pos)
            except re.error as e:
                logger.warning(f"Regex pattern error: {e}")
                continue
        
        breaks.append(len(text))  # Always end at text end
        breaks = sorted(list(set(breaks)))  # Remove duplicates and sort
        
        # Create sections
        sections = []
        for i in range(len(breaks) - 1):
            start_pos = breaks[i]
            end_pos = breaks[i + 1]
            section = text[start_pos:end_pos].strip()
            
            if section and len(section) > 30:  # Skip very short sections
                sections.append(section)
        
        # If no clear sections found, split by double line breaks
        if len(sections) <= 1:
            logger.info("No clear sections found with patterns, trying paragraph splits")
            paragraph_sections = []
            for para in text.split('\n\n'):
                para = para.strip()
                if para and len(para) > 30:
                    paragraph_sections.append(para)
            
            if paragraph_sections:
                sections = paragraph_sections
            else:
                # Last resort: return the entire text
                logger.info("No paragraph sections found, using entire text")
                sections = [text] if text and text.strip() else []
        
        logger.info(f"Identified {len(sections)} sections")
        return sections
    
    def _split_section_to_target_tokens(self, section: str) -> List[str]:
        """
        Split a section into chunks that meet target token requirements.
        """
        if not section or not section.strip():
            return []
        
        try:
            current_tokens = get_token_numbers(section, self.model_name)
        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            # Fallback: estimate tokens based on character count
            current_tokens = len(section) // 4  # Rough estimate
        
        # If section is within target range, return as is
        if self.min_tokens <= current_tokens <= self.max_tokens:
            return [section]
        
        # If section is too small, return as is (will be handled by overlap logic)
        if current_tokens < self.min_tokens:
            return [section]
        
        # If section is too large, split it recursively
        return self._recursive_split(section)
    
    def _recursive_split(self, text: str) -> List[str]:
        """
        Recursively split text using semantic separators until target token count is reached.
        Enhanced error handling for empty separators.
        """
        if not text or not text.strip():
            return []
        
        try:
            current_tokens = get_token_numbers(text, self.model_name)
        except Exception as e:
            logger.error(f"Error counting tokens in recursive split: {e}")
            current_tokens = len(text) // 4  # Fallback estimation
        
        if current_tokens <= self.max_tokens:
            return [text]
        
        # Try each separator in order of semantic importance
        for separator in self.semantic_separators:
            if not separator:  # Skip empty separators
                logger.warning("Encountered empty separator, skipping")
                continue
                
            if separator in text and len(text.split(separator)) > 1:
                try:
                    parts = text.split(separator, 1)
                    if len(parts) != 2:
                        continue
                        
                    left_part = parts[0].strip()
                    right_part_content = parts[1].strip()
                    
                    # Skip if either part is empty
                    if not left_part or not right_part_content:
                        continue
                    
                    # Reconstruct right part with separator if it's a meaningful separator
                    if separator.strip():  # If separator has meaningful content
                        right_part = separator + right_part_content
                    else:
                        right_part = right_part_content
                    
                    # Check if split produces reasonable chunks
                    try:
                        left_tokens = get_token_numbers(left_part, self.model_name)
                        right_tokens = get_token_numbers(right_part, self.model_name)
                    except Exception as e:
                        logger.error(f"Error counting tokens for split parts: {e}")
                        continue
                    
                    # If left part is in acceptable range, process both parts
                    if self.min_tokens <= left_tokens <= self.max_tokens:
                        result = [left_part]
                        result.extend(self._recursive_split(right_part))
                        return [chunk for chunk in result if chunk and chunk.strip()]
                    
                    # If left part is still too big, continue splitting it
                    elif left_tokens > self.max_tokens:
                        left_result = self._recursive_split(left_part)
                        right_result = self._recursive_split(right_part)
                        result = left_result + right_result
                        return [chunk for chunk in result if chunk and chunk.strip()]
                        
                except Exception as e:
                    logger.error(f"Error processing separator '{separator}': {e}")
                    continue
        
        # Fallback: use character-based splitting if no good separator found
        logger.warning("No suitable separators found, using character-based splitting")
        return self._character_split(text)
    
    def _character_split(self, text: str) -> List[str]:
        """
        Last resort: split by characters while trying to preserve word boundaries.
        """
        if not text or not text.strip():
            return []
        
        # Calculate approximate character count for target tokens
        # Rough estimation: 1 token ≈ 4 characters for English text
        target_chars = self.target_tokens * 4
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + target_chars, len(text))
            
            # Try to end at a word boundary
            if end < len(text):
                # Look backwards for a space or punctuation
                for i in range(end, max(start + target_chars // 2, start), -1):
                    if text[i] in ' .,;:!?\n':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk and len(chunk) > 10:  # Only add non-trivial chunks
                chunks.append(chunk)
            
            start = end
        
        return chunks
    
    def _fallback_chunking(self, text: str) -> List[str]:
        """
        Fallback chunking method using langchain's RecursiveCharacterTextSplitter.
        """
        try:
            logger.info("Using fallback chunking with RecursiveCharacterTextSplitter")
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.target_tokens * 4,  # Approximate character count
                chunk_overlap=int(self.target_tokens * 4 * self.overlap_percentage),
                length_function=len,
                separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""]
            )
            
            chunks = splitter.split_text(text)
            return [chunk.strip() for chunk in chunks if chunk and chunk.strip() and len(chunk.strip()) > 10]
            
        except Exception as e:
            logger.error(f"Error in fallback chunking: {e}")
            # Final fallback: simple character-based split
            return self._character_split(text)
    
    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """
        Apply overlap between consecutive chunks.
        """
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            if not chunk or not chunk.strip():
                continue
                
            if i == 0:
                # First chunk: no prefix overlap needed
                overlapped_chunk = chunk
            else:
                # Calculate overlap with previous chunk
                prev_chunk = chunks[i-1]
                if not prev_chunk or not prev_chunk.strip():
                    overlapped_chunk = chunk
                else:
                    overlap_chars = int(len(prev_chunk) * self.overlap_percentage)
                    
                    if overlap_chars > 0:
                        # Get overlap from end of previous chunk
                        overlap_text = prev_chunk[-overlap_chars:].strip()
                        
                        # Find a good breaking point in the overlap
                        sentences = re.split(r'[.!?]+\s+', overlap_text)
                        if len(sentences) > 1:
                            overlap_text = sentences[-1]  # Take the last complete sentence
                        
                        overlapped_chunk = overlap_text + "\n\n" + chunk
                    else:
                        overlapped_chunk = chunk
            
            # Ensure the final chunk doesn't exceed token limits
            try:
                chunk_tokens = get_token_numbers(overlapped_chunk, self.model_name)
                if chunk_tokens > self.max_tokens:
                    # If overlap made it too big, use original chunk
                    overlapped_chunk = chunk
            except Exception as e:
                logger.error(f"Error checking token count for overlapped chunk: {e}")
                overlapped_chunk = chunk
            
            if overlapped_chunk and overlapped_chunk.strip():
                overlapped_chunks.append(overlapped_chunk)
        
        return overlapped_chunks


class PageBasedTextSplitter:
    """
    Page-based text splitter that chunks documents by pages with overlap.
    """
    
    def __init__(self, overlap_percentage: float = 0.1, model_name: str = "gpt-4"):
        self.overlap_percentage = overlap_percentage
        self.model_name = model_name
    
    def split_text_by_pages(self, result) -> List[Dict]:
        """
        Split document into page-based chunks with overlap.
        
        Args:
            result: Document Intelligence analysis result
            
        Returns:
            List of page-based chunks with metadata
        """
        if not result.pages:
            return []
        
        page_chunks = []
        
        for page_idx, page in enumerate(result.pages):
            page_number = page.page_number
            
            # Extract content for this specific page
            page_content = self._extract_page_content(result, page_number)
            
            if not page_content or len(page_content.strip()) < 50:
                continue
            
            # Apply overlap with previous page if not the first page
            if page_idx > 0 and page_chunks:
                prev_page_content = page_chunks[-1]["raw_content"]
                page_content_with_overlap = self._apply_page_overlap(prev_page_content, page_content)
            else:
                page_content_with_overlap = page_content
            
            # Count tokens for this page
            page_tokens = get_token_numbers(page_content_with_overlap, self.model_name)
            
            chunk_data = {
                "content": page_content_with_overlap,
                "raw_content": page_content,  # Store original content for next page overlap
                "page_number": page_number,
                "tokens": page_tokens,
                "length": len(page_content_with_overlap)
            }
            
            page_chunks.append(chunk_data)
        
        return page_chunks
    
    def _extract_page_content(self, result, target_page_number: int) -> str:
        """
        Extract content for a specific page from Document Intelligence result.
        """
        page_content_parts = []
        
        # Extract paragraphs for this page
        if result.paragraphs:
            for paragraph in result.paragraphs:
                if paragraph.bounding_regions:
                    for region in paragraph.bounding_regions:
                        if region.page_number == target_page_number:
                            page_content_parts.append(paragraph.content)
                            break

        
        
        # If no paragraphs found, try to extract from tables for this page
        if not page_content_parts and result.tables:
            for table in result.tables:
                if table.bounding_regions:
                    for region in table.bounding_regions:
                        if region.page_number == target_page_number:
                            # Extract table content
                            table_content = self._extract_table_content(table)
                            if table_content:
                                page_content_parts.append(table_content)
                            break
        
        return "\n\n".join(page_content_parts)
    
    def _extract_table_content(self, table) -> str:
        """
        Extract content from a table structure.
        """
        if not table.cells:
            return ""
        
        # Simple table extraction - can be enhanced
        table_text = []
        current_row = -1
        row_content = []
        
        for cell in table.cells:
            if cell.row_index != current_row:
                if row_content:
                    table_text.append(" | ".join(row_content))
                current_row = cell.row_index
                row_content = []
            row_content.append(cell.content or "")
        
        if row_content:
            table_text.append(" | ".join(row_content))
        
        return "\n".join(table_text)
    
    def _apply_page_overlap(self, prev_page_content: str, current_page_content: str) -> str:
        """
        Apply overlap between consecutive pages.
        """
        if not prev_page_content:
            return current_page_content
        
        # Calculate overlap size
        overlap_chars = int(len(prev_page_content) * self.overlap_percentage)
        
        if overlap_chars > 0:
            # Get overlap from end of previous page
            overlap_text = prev_page_content[-overlap_chars:].strip()
            
            # Find a good breaking point (end of sentence or paragraph)
            sentences = re.split(r'[.!?]+\s+', overlap_text)
            if len(sentences) > 1:
                overlap_text = ". ".join(sentences[-2:]) + "."  # Take last complete sentences
            
            # Combine with current page content
            return overlap_text + "\n\n" + current_page_content
        
        return current_page_content


class UnifiedFileUploadPlugin:
    """Unified plugin for file upload and document processing with multiple chunking strategies."""
    
    def __init__(self):
        """Initialize the UnifiedFileUploadPlugin with Azure services."""
        self.settings = Settings()
        
        # Get processing method from environment variable
        self.processing_method = os.getenv("PROCESSING_METHOD", "semantic").lower()
        logger.info(f"Using processing method: {self.processing_method}")
        
        # Azure AI Search
        self.search_client = SearchClient(
            endpoint=self.settings.AZURE_AI_SEARCH_ENDPOINT,
            index_name=self.settings.AZURE_AI_SEARCH_INDEX_NAME,
            credential=AzureKeyCredential(self.settings.AZURE_AI_SEARCH_API_KEY)
        )
        
        # Document Intelligence
        self.doc_intelligence_client = DocumentIntelligenceClient(
            endpoint=self.settings.AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT,
            credential=AzureKeyCredential(self.settings.AZURE_DOCUMENT_INTELLIGENCE_API_KEY)
        )
        
        # OpenAI for embeddings
        self.openai_client = AsyncAzureOpenAI(
            api_key=self.settings.AZURE_OPENAI_API_KEY,
            api_version=self.settings.AZURE_OPENAI_API_VERSION,
            azure_endpoint=self.settings.AZURE_OPENAI_ENDPOINT
        )
        
        # Initialize text splitters based on processing method
        if self.processing_method == "semantic":
            self.semantic_splitter = SemanticAwareTextSplitter(
                target_tokens=400,
                token_variance=100,
                overlap_percentage=0.1,
                model_name="gpt-4"
            )
        elif self.processing_method == "page":
            self.page_splitter = PageBasedTextSplitter(
                overlap_percentage=0.1,
                model_name="gpt-4"
            )
        else:
            # Default to semantic if unknown method
            logger.warning(f"Unknown processing method: {self.processing_method}. Defaulting to semantic.")
            self.processing_method = "semantic"
            self.semantic_splitter = SemanticAwareTextSplitter(
                target_tokens=400,
                token_variance=100,
                overlap_percentage=0.1,
                model_name="gpt-4"
            )
        
    async def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using Azure OpenAI."""
        try:
            response = await self.openai_client.embeddings.create(
                input=text,
                model=self.settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def _generate_doc_hash(self, file_path: str) -> str:
        """Generate MD5 hash of file content for duplicate detection."""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Error generating file hash: {e}")
            return ""
    
    async def _file_exists_in_vector_db(self, file_hash: str) -> bool:
        """Check if file already exists in vector database."""
        try:
            search_results = self.search_client.search(
                search_text="*",
                filter=f"file_hash eq '{file_hash}'",
                top=1
            )
            
            for _ in search_results:
                return True
            return False
        except Exception as e:
            logger.error(f"Error checking file existence: {e}")
            return False
    
    async def _process_pdf_with_semantic_chunking(
        self, 
        file_path: str, 
        original_filename: str,
        document_type: str,
        company: str,
        industry: str,
        report_year: str,
        result
    ) -> List[Dict]:
        """Process PDF using semantic-aware chunking."""
        try:
            # Extract content and metadata
            content = result.content if result.content else ""
            
            page_count = len(result.pages) if result.pages else 0
            
            logger.info(f"Processing with semantic chunking: {page_count} pages, {len(content)} characters")
            
            # Generate file hash for duplicate detection
            file_hash = self._generate_doc_hash(file_path)
            file_name = original_filename
            
            documents = []
            
            # Use semantic-aware chunking
            logger.info("Starting semantic-aware chunking...")
            semantic_chunks = self.semantic_splitter.split_text_with_semantic_awareness(content)
            
            logger.info(f"Created {len(semantic_chunks)} semantic chunks")
            
            # Process each semantic chunk
            for chunk_num, chunk_text in enumerate(semantic_chunks, 1):
                chunk_text = chunk_text.strip()
                if len(chunk_text) < 50:  # Skip very short chunks
                    continue
                
                try:
                    # Count tokens for this chunk
                    chunk_tokens = get_token_numbers(chunk_text, "gpt-4")
                    
                    # Generate embedding
                    embedding = await self._get_embedding(chunk_text)
                    
                    # Create document object for vector storage
                    doc_id = f"{file_hash}_semantic_{chunk_num}"
                    
                    # Determine page number (rough estimate based on content position)
                    content_position = content.find(chunk_text[:100])  # Find chunk in original content
                    estimated_page = min(max(1, int((content_position / len(content)) * page_count)), page_count)
                    
                    document = {
                        "docId": doc_id,
                        "content": chunk_text,
                        "content_vector": embedding,
                        "title": f"{file_name} - Section {chunk_num}",
                        "file_name": file_name,
                        "file_hash": file_hash,
                        "page_number": estimated_page,
                        "paragraph_number": chunk_num,
                        "chunk_number": chunk_num,
                        "document_type": document_type,
                        "company": company,
                        "industry": industry,
                        "report_year": report_year,
                        "source": file_path,
                        "metadata": json.dumps({
                            "total_pages": page_count,
                            "total_semantic_chunks": len(semantic_chunks),
                            "chunk_tokens": chunk_tokens,
                            "chunk_length": len(chunk_text),
                            "file_size": Path(file_path).stat().st_size,
                            "processing_method": "semantic_aware_chunking",
                            "target_tokens": 400,
                            "overlap_percentage": 10
                        })
                    }
                    
                    documents.append(document)
                    logger.info(f"Processed semantic chunk {chunk_num}: {chunk_tokens} tokens, {len(chunk_text)} characters")
                
                except Exception as e:
                    logger.error(f"Error processing semantic chunk {chunk_num}: {e}")
                    continue
            
            return documents
            
        except Exception as e:
            logger.error(f"Error in semantic chunking: {e}")
            return []
    
    async def _process_pdf_with_page_chunking(
        self, 
        file_path: str, 
        original_filename: str,
        document_type: str,
        company: str,
        industry: str,
        report_year: str,
        result
    ) -> List[Dict]:
        """Process PDF using page-based chunking."""
        try:
            page_count = len(result.pages) if result.pages else 0
            logger.info(f"Processing with page chunking: {page_count} pages")
            
            # Generate file hash for duplicate detection
            file_hash = self._generate_doc_hash(file_path)
            file_name = original_filename
            
            documents = []
            
            # Use page-based chunking
            logger.info("Starting page-based chunking...")
            page_chunks = self.page_splitter.split_text_by_pages(result)
            
            logger.info(f"Created {len(page_chunks)} page chunks")
            
            # Process each page chunk
            for chunk_num, page_chunk in enumerate(page_chunks, 1):
                chunk_text = page_chunk["content"].strip()
                if len(chunk_text) < 50:  # Skip very short chunks
                    continue
                
                try:
                    # Generate embedding
                    embedding = await self._get_embedding(chunk_text)
                    
                    # Create document object for vector storage
                    doc_id = f"{file_hash}_page_{page_chunk['page_number']}"
                    
                    document = {
                        "docId": doc_id,
                        "content": chunk_text,
                        "content_vector": embedding,
                        "title": f"{file_name} - Page {page_chunk['page_number']}",
                        "file_name": file_name,
                        "file_hash": file_hash,
                        "page_number": page_chunk['page_number'],
                        "paragraph_number": 1,  # Page-level, so always 1
                        "chunk_number": chunk_num,
                        "document_type": document_type,
                        "company": company,
                        "industry": industry,
                        "report_year": report_year,
                        "source": file_path,
                        "metadata": json.dumps({
                            "total_pages": page_count,
                            "total_page_chunks": len(page_chunks),
                            "chunk_tokens": page_chunk['tokens'],
                            "chunk_length": page_chunk['length'],
                            "file_size": Path(file_path).stat().st_size,
                            "processing_method": "page_based_chunking",
                            "overlap_percentage": 10
                        })
                    }
                    
                    documents.append(document)
                    logger.info(f"Processed page chunk {chunk_num} (page {page_chunk['page_number']}): {page_chunk['tokens']} tokens, {page_chunk['length']} characters")
                
                except Exception as e:
                    logger.error(f"Error processing page chunk {chunk_num}: {e}")
                    continue
            
            return documents
            
        except Exception as e:
            logger.error(f"Error in page chunking: {e}")
            return []
    
    async def _process_pdf_file(
        self, 
        file_path: str, 
        original_filename: str,
        document_type: str,
        company: str,
        industry: str,
        report_year: str
    ) -> List[Dict]:
        """Process PDF file using the configured chunking method."""
        try:
            # Process PDF with Document Intelligence
            with open(file_path, "rb") as file:
                poller = self.doc_intelligence_client.begin_analyze_document(
                    # model_id="prebuilt-layout",
                    # body=file, 
                    # output_content_format=DocumentContentFormat.MARKDOWN
                    "prebuilt-layout",
                    AnalyzeDocumentRequest(bytes_source=file.read()),
                    output_content_format=DocumentContentFormat.MARKDOWN,
                )
            
            result: AnalyzeResult = poller.result()

            print(result.content_format) 
            print(result.content[:3000]) # Print first 2000 characters of result for debugging
            page_count = len(result.pages) if result.pages else 0
            
            logger.info(f"Document Intelligence processing complete: {page_count} pages")
            
            # Route to appropriate chunking method based on configuration
            if self.processing_method == "semantic":
                documents = await self._process_pdf_with_semantic_chunking(
                    file_path, original_filename, document_type, company, industry, report_year, result
                )
            elif self.processing_method == "page":
                documents = await self._process_pdf_with_page_chunking(
                    file_path, original_filename, document_type, company, industry, report_year, result
                )
            else:
                # Fallback to semantic
                logger.warning(f"Unknown processing method: {self.processing_method}. Using semantic chunking.")
                documents = await self._process_pdf_with_semantic_chunking(
                    file_path, original_filename, document_type, company, industry, report_year, result
                )
            
            logger.info(f"Processed {len(documents)} chunks using {self.processing_method} chunking from {page_count} pages")
            
            # Log token distribution for analysis
            if documents:
                token_counts = [get_token_numbers(doc["content"], "gpt-4") for doc in documents]
                avg_tokens = sum(token_counts) / len(token_counts)
                min_tokens = min(token_counts)
                max_tokens = max(token_counts)
                logger.info(f"Token distribution - Avg: {avg_tokens:.1f}, Min: {min_tokens}, Max: {max_tokens}")
            
            return documents
            
        except Exception as e:
            logger.error(f"Error processing PDF file {file_path}: {e}")
            return []

    async def _upload_documents_to_vector_db(self, documents: List[Dict]) -> bool:
        """Upload processed documents to Azure AI Search."""
        try:
            if not documents:
                return False
            
            # Upload documents in batches
            batch_size = 50
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                
                try:
                    result = self.search_client.upload_documents(documents=batch)
                    logger.info(f"Uploaded batch {i//batch_size + 1} with {len(batch)} documents")
                    
                    # Check for any failed uploads
                    for res in result:
                        if not res.succeeded:
                            logger.error(f"Failed to upload document {res.key}: {res.error_message}")
                            
                except Exception as e:
                    logger.error(f"Error uploading batch {i//batch_size + 1}: {e}")
                    return False
            
            logger.info(f"Successfully uploaded {len(documents)} documents to vector database")
            return True
        except Exception as e:
            logger.error(f"Error processing PDF file {e}")
            return []

    @kernel_function(
        description="Handle user file upload requests and provide guidance",
        name="handle_upload_request"
    )
    async def handle_upload_request(self, user_message: str) -> str:
        """
        Detect and handle file upload requests from users.
        
        Args:
            user_message: User's message content
            
        Returns:
            JSON string with upload guidance
        """
        # Check for upload-related keywords
        upload_keywords = [
            "upload", "업로드", "파일 올리기", "문서 올리기", 
            "file upload", "add document", "add file"
        ]
        
        message_lower = user_message.lower()
        wants_upload = any(keyword in message_lower for keyword in upload_keywords)
        
        if wants_upload:
            response = {
                "status": "upload_requested",
                "message": """📁 **File Upload Available**

I can help you upload and process documents for AI analysis!

**Supported Files:** PDF, DOCX, TXT (up to 10 files)
**Process:** Files are chunked by page and stored for semantic search

**To Upload:**
1. Use your application's file upload interface
2. Select up to 10 files
3. Provide metadata (optional): document type, company, industry, year
4. Files will be processed in background
5. You'll get confirmation when ready

**After Upload:**
I can search, analyze, and answer questions about your documents.

Ready to upload files?""",
                "action_required": "file_upload"
            }
        else:
            response = {
                "status": "no_upload_needed",
                "message": "No file upload detected in your message.",
                "action_required": None
            }
        
        return json.dumps(response)
    
    @kernel_function(
        description="Upload and process documents for AI Search. Call this when files need to be processed as unstructured document data.",
        name="upload_files"  # 함수명을 upload_files로 변경 (main.py에서 호출하는 이름과 맞춤)
    )
    async def upload_files(
        self,
        file_paths: str,  # JSON string of file paths
        file_names: str = "",  # JSON string of original filenames 추가
        document_type: str = "GENERAL",
        company: str = "",
        industry: str = "",
        report_year: str = "",
        force_upload: str = "false"
    ) -> str:
        """Upload and process multiple files for document search."""
        try:
            # Parse file paths and names from JSON strings
            try:
                file_paths_list = json.loads(file_paths)
                file_names_list = json.loads(file_names) if file_names else []
            except json.JSONDecodeError:
                # If not JSON, assume single file path
                file_paths_list = [file_paths]
                file_names_list = [Path(file_paths).name]
        
            # Ensure we have matching file names
            if len(file_names_list) != len(file_paths_list):
                file_names_list = [Path(fp).name for fp in file_paths_list]
        
            logger.info(f"Starting upload for {len(file_paths_list)} files")
        
            # Convert string boolean to actual boolean
            force_upload_bool = force_upload.lower() == "true"
        
            results = []
            total_documents = 0
        
            for file_path, original_filename in zip(file_paths_list, file_names_list):
                try:
                    # Check if file exists
                    if not Path(file_path).exists():
                        results.append({
                            "file": original_filename,  # 실제 파일명으로 표시
                            "status": "error",
                            "message": "File not found"
                        })
                        continue
                
                    # Generate file hash for duplicate check
                    file_hash = self._generate_doc_hash(file_path)
                
                    # Check if file already exists in vector DB
                    if not force_upload_bool and await self._file_exists_in_vector_db(file_hash):
                        results.append({
                            "file": original_filename,
                            "status": "skipped",
                            "message": "File already exists in vector database"
                        })
                        continue
                
                    # Process file based on extension
                    file_extension = Path(original_filename).suffix.lower()
                
                    if file_extension == '.pdf':
                        documents = await self._process_pdf_file(
                            file_path, original_filename, document_type, company, industry, report_year
                        )
                    else:
                        results.append({
                            "file": original_filename,
                            "status": "error",
                            "message": f"Unsupported file type: {file_extension}"
                        })
                        continue
                
                    if documents:
                        # Upload to vector database
                        if await self._upload_documents_to_vector_db(documents):
                            results.append({
                                "file": original_filename,
                                "status": "success",
                                "message": f"Successfully uploaded {len(documents)} chunks",
                                "chunks_count": len(documents)
                            })
                            total_documents += len(documents)
                        else:
                            results.append({
                                "file": original_filename,
                                "status": "error",
                                "message": "Failed to upload to vector database"
                            })
                    else:
                        results.append({
                            "file": original_filename,
                            "status": "error",
                            "message": "No content could be extracted from file"
                        })
                        
                except Exception as e:
                    logger.error(f"Error processing file {original_filename}: {e}")
                    results.append({
                        "file": original_filename,
                        "status": "error",
                        "message": str(e)
                    })
        
            # Prepare final response
            successful_uploads = [r for r in results if r["status"] == "success"]
        
            response = {
                "status": "completed",
                "total_files": len(file_paths_list),
                "successful_uploads": len(successful_uploads),
                "total_documents_uploaded": total_documents,
                "results": results
            }
        
            logger.info(f"Upload completed: {len(successful_uploads)}/{len(file_paths_list)} files successful")
            return json.dumps(response)
            
        except Exception as e:
            error_msg = f"Error in file upload: {str(e)}"
            logger.error(error_msg)
            return json.dumps({
                "status": "error", 
                "message": error_msg,
                "results": []
            })
    
    
    
    @kernel_function(
        description="Check if documents exist in the vector database",
        name="check_docs_status"
    )
    async def check_docs_status(self, file_names: str) -> str:
        """
        Check the status of documents in the vector database.

        Args:
            file_names: Comma-separated string of file names to check
            
        Returns:
            JSON string with file status information
        """
        try:
            # Parse file names
            files_list = [name.strip() for name in file_names.split(",")]
            
            results = {}
            
            for file_name in files_list:
                try:
                    # Search for documents with this file name
                    search_results = self.search_client.search(
                        search_text="*",
                        filter=f"file_name eq '{file_name}'",
                        top=1
                    )
                    
                    exists = False
                    for _ in search_results:
                        exists = True
                        break
                    
                    results[file_name] = {
                        "exists": exists,
                        "status": "found" if exists else "not_found"
                    }
                    
                except Exception as e:
                    logger.error(f"Error checking file {file_name}: {e}")
                    results[file_name] = {
                        "exists": False,
                        "status": "error",
                        "error": str(e)
                    }
            
            existing_files = [f for f, info in results.items() if info["exists"]]
            missing_files = [f for f, info in results.items() if not info["exists"]]
            
            response = {
                "status": "success",
                "total_checked": len(files_list),
                "existing_files": existing_files,
                "missing_files": missing_files,
                "results": results
            }
            
            return json.dumps(response)
            
        except Exception as e:
            logger.error(f"Error checking file status: {e}")
            return json.dumps({
                "status": "error",
                "message": str(e)
            })
    
    @kernel_function(
        description="Format upload completion notification for users",
        name="notify_upload_completion"
    )
    async def notify_upload_completion(self, upload_results: str) -> str:
        """
        Process and format upload completion notification.
        
        Args:
            upload_results: JSON string containing upload results
            
        Returns:
            JSON string with formatted completion message
        """
        try:
            results = json.loads(upload_results)
            
            total_files = results.get("total_files", 0)
            successful = results.get("successful_uploads", 0)
            failed = total_files - successful
            
            if successful > 0:
                message = f"""🎉 **Upload Complete!**

📊 **Summary:**
• Files Processed: {total_files}
• Successfully Uploaded: {successful}
• Failed: {failed}

✅ **Your documents are ready for AI search!**

**What's next:**
• Ask questions about your documents
• Request analysis or summaries
• Search for specific information
• Compare data across documents

**Example questions:**
• "What are the key findings in my documents?"
• "Summarize the main points from the uploaded reports"
• "Find information about revenue growth"

How can I help analyze your documents?"""
            else:
                message = f"""⚠️ **Upload Issues**

📊 **Summary:**
• Total Files: {total_files}
• Failed: {failed}

**Common issues:**
• Unsupported file format (only PDF, DOCX, TXT)
• File corruption or encryption
• Network issues

**Next steps:**
• Check file formats
• Try smaller files
• Upload one file at a time

Would you like to try again?"""
            
            response = {
                "status": "completed",
                "message": message,
                "successful_uploads": successful,
                "failed_uploads": failed,
                "ready_for_analysis": successful > 0
            }
            
            return json.dumps(response)
            
        except Exception as e:
            logger.error(f"Error formatting completion message: {e}")
            return json.dumps({
                "status": "error",
                "message": "Error processing upload notification."
            })
