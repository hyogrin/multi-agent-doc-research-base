"""
Unified File Upload Plugin

This plugin handles the complete file upload workflow:
- File upload request handling
- Document processing and chunking
- Vector storage in Azure AI Search
- Upload status tracking and notification

All functionalities are consolidated into a single, simple plugin.
"""

import json
import logging
import hashlib
import os
from typing import Dict, List
from pathlib import Path

from semantic_kernel.functions import kernel_function
from config.config import Settings
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import DocumentContentFormat
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import AsyncAzureOpenAI

logger = logging.getLogger(__name__)


class UnifiedFileUploadPlugin:
    """
    Unified plugin for complete file upload and processing workflow.
    Handles everything from upload requests to vector storage.
    """
    
    def __init__(self):
        """Initialize the UnifiedFileUploadPlugin with Azure services."""
        self.settings = Settings()

        
        # Initialize Azure AI Search client
        self.search_client = SearchClient(
            endpoint=self.settings.AZURE_AI_SEARCH_ENDPOINT,
            index_name=self.settings.AZURE_AI_SEARCH_INDEX_NAME,
            credential=AzureKeyCredential(self.settings.AZURE_AI_SEARCH_API_KEY)
        )
        
        # Initialize Document Intelligence client
        self.doc_intelligence_client = DocumentIntelligenceClient(
            endpoint=os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"),
            credential=AzureKeyCredential(os.getenv("AZURE_DOCUMENT_INTELLIGENCE_API_KEY"))
        )
        
        # Initialize OpenAI client for embeddings
        self.openai_client = AsyncAzureOpenAI(
            api_key=self.settings.AZURE_OPENAI_API_KEY,
            api_version=self.settings.AZURE_OPENAI_API_VERSION,
            azure_endpoint=self.settings.AZURE_OPENAI_ENDPOINT
        )
        
        # Initialize text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
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
    
    def _generate_file_hash(self, file_path: str) -> str:
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
    
    async def _process_pdf_file(
        self, 
        file_path: str, 
        document_type: str,
        company: str,
        industry: str,
        report_year: str
    ) -> List[Dict]:
        """Process PDF file into chunks using Document Intelligence and prepare for vector storage."""
        try:
            # Process PDF with Document Intelligence
            with open(file_path, "rb") as file:
                poller = self.doc_intelligence_client.begin_analyze_document(
                    model_id="prebuilt-layout",
                    body=file, 
                    output_content_format=DocumentContentFormat.MARKDOWN
                )
            
            result = poller.result()
            
            # Extract content and metadata
            content = result.content if result.content else ""
            page_count = len(result.pages) if result.pages else 0
            
            logger.info(f"Processed PDF with Document Intelligence: {page_count} pages")
            
            # Generate file hash for duplicate detection
            file_hash = self._generate_file_hash(file_path)
            file_name = Path(file_path).name
            
            documents = []
            
            # Process paragraphs from Document Intelligence
            if result.paragraphs:
                for para_num, paragraph in enumerate(result.paragraphs, 1):
                    para_content = paragraph.content
                    page_number = paragraph.bounding_regions[0].page_number if paragraph.bounding_regions else 1
                    
                    # Split paragraph content into chunks if it's too long
                    chunks = self.text_splitter.split_text(para_content)
                    
                    # Process each chunk
                    for chunk_num, chunk_text in enumerate(chunks, 1):
                        if len(chunk_text.strip()) < 50:  # Skip very short chunks
                            continue
                        
                        try:
                            # Generate embedding for chunk
                            embedding = await self._get_embedding(chunk_text)
                            
                            # Create document object for vector storage
                            doc_id = f"{file_hash}_{para_num}_{chunk_num}"
                            
                            document = {
                                "id": doc_id,
                                "content": chunk_text,
                                "content_vector": embedding,
                                "title": f"{file_name} - Para {para_num}, Chunk {chunk_num}",
                                "file_name": file_name,
                                "file_hash": file_hash,
                                "page_number": page_number,
                                "paragraph_number": para_num,
                                "chunk_number": chunk_num,
                                "document_type": document_type,
                                "company": company,
                                "industry": industry,
                                "report_year": report_year,
                                "source": file_path,
                                "metadata": json.dumps({
                                    "total_pages": page_count,
                                    "total_paragraphs": len(result.paragraphs),
                                    "total_chunks_in_paragraph": len(chunks),
                                    "file_size": Path(file_path).stat().st_size,
                                    "processing_method": "document_intelligence"
                                })
                            }
                            
                            documents.append(document)
                            
                        except Exception as e:
                            logger.error(f"Error processing chunk {chunk_num} of paragraph {para_num}: {e}")
                            continue
            else:
                # Fallback: if no paragraphs, process the entire content
                logger.warning("No paragraphs found, processing entire content")
                chunks = self.text_splitter.split_text(content)
                
                for chunk_num, chunk_text in enumerate(chunks, 1):
                    if len(chunk_text.strip()) < 50:
                        continue
                    
                    try:
                        embedding = await self._get_embedding(chunk_text)
                        doc_id = f"{file_hash}_content_{chunk_num}"
                        
                        document = {
                            "id": doc_id,
                            "content": chunk_text,
                            "content_vector": embedding,
                            "title": f"{file_name} - Content Chunk {chunk_num}",
                            "file_name": file_name,
                            "file_hash": file_hash,
                            "page_number": 1,
                            "paragraph_number": 0,
                            "chunk_number": chunk_num,
                            "document_type": document_type,
                            "company": company,
                            "industry": industry,
                            "report_year": report_year,
                            "source": file_path,
                            "metadata": json.dumps({
                                "total_pages": page_count,
                                "total_chunks": len(chunks),
                                "file_size": Path(file_path).stat().st_size,
                                "processing_method": "document_intelligence_fallback"
                            })
                        }
                        
                        documents.append(document)
                        
                    except Exception as e:
                        logger.error(f"Error processing content chunk {chunk_num}: {e}")
                        continue
            
            logger.info(f"Processed {len(documents)} chunks from {page_count} pages using Document Intelligence")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing PDF file {file_path} with Document Intelligence: {e}")
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
            logger.error(f"Error uploading to vector database: {e}")
            return False
    
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
        description="Upload and process files for document search. Call this when files need to be processed.",
        name="upload_files"
    )
    async def upload_files(
        self,
        file_paths: str,  # JSON string of file paths
        document_type: str = "GENERAL",
        company: str = "",
        industry: str = "",
        report_year: str = "",
        force_upload: str = "false"
    ) -> str:
        """
        Upload and process multiple files for document search.
        
        Args:
            file_paths: JSON string containing list of file paths to upload
            document_type: Type of document (IR_REPORT, MANUAL, GENERAL, etc.)
            company: Company name associated with the documents
            industry: Industry category
            report_year: Year of the report
            force_upload: Whether to force upload even if file exists ("true"/"false")
            
        Returns:
            JSON string with upload results
        """
        try:
            # Parse file paths from JSON string
            try:
                file_paths_list = json.loads(file_paths)
            except json.JSONDecodeError:
                # If not JSON, assume single file path
                file_paths_list = [file_paths]
            
            logger.info(f"Starting upload for {len(file_paths_list)} files")
            
            # Convert string boolean to actual boolean
            force_upload_bool = force_upload.lower() == "true"
            
            results = []
            total_documents = 0
            
            for file_path in file_paths_list:
                try:
                    # Check if file exists
                    if not Path(file_path).exists():
                        results.append({
                            "file": file_path,
                            "status": "error",
                            "message": "File not found"
                        })
                        continue
                    
                    # Generate file hash for duplicate check
                    file_hash = self._generate_file_hash(file_path)
                    
                    # Check if file already exists in vector DB
                    if not force_upload_bool and await self._file_exists_in_vector_db(file_hash):
                        results.append({
                            "file": file_path,
                            "status": "skipped",
                            "message": "File already exists in vector database"
                        })
                        continue
                    
                    # Process file based on extension
                    file_extension = Path(file_path).suffix.lower()
                    
                    if file_extension == '.pdf':
                        documents = await self._process_pdf_file(
                            file_path, document_type, company, industry, report_year
                        )
                    else:
                        results.append({
                            "file": file_path,
                            "status": "error",
                            "message": f"Unsupported file type: {file_extension}"
                        })
                        continue
                    
                    if documents:
                        # Upload to vector database
                        if await self._upload_documents_to_vector_db(documents):
                            results.append({
                                "file": file_path,
                                "status": "success",
                                "message": f"Successfully uploaded {len(documents)} chunks",
                                "chunks_count": len(documents)
                            })
                            total_documents += len(documents)
                        else:
                            results.append({
                                "file": file_path,
                                "status": "error",
                                "message": "Failed to upload to vector database"
                            })
                    else:
                        results.append({
                            "file": file_path,
                            "status": "error",
                            "message": "No content could be extracted from file"
                        })
                        
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                    results.append({
                        "file": file_path,
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
        description="Check if files exist in the vector database",
        name="check_files_status"
    )
    async def check_files_status(self, file_names: str) -> str:
        """
        Check the status of files in the vector database.
        
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
