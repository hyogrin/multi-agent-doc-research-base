"""
Document Intelligence Upload Plugin for PDF Processing and AI Search Integration

This plugin handles PDF file processing using Azure Document Intelligence,
converts content to markdown format, and uploads to Azure AI Search with
metadata for IR reports and market research documents.
"""

import os
import json
import hashlib
import datetime
from typing import List, Dict, Any, Optional, BinaryIO
import re
from collections import Counter

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, DocumentContentFormat
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from openai import AzureOpenAI



class UploadDocPlugin:
    """Plugin for uploading and processing PDF documents with Document Intelligence."""
    
    def __init__(self):
        """Initialize the upload document plugin with required clients."""
        # Document Intelligence setup
        self.doc_intelligence_endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        self.doc_intelligence_key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_API_KEY")
        
        # AI Search setup
        self.search_endpoint = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
        self.search_key = os.getenv("AZURE_AI_SEARCH_API_KEY")
        self.index_name = os.getenv("AZURE_AI_SEARCH_INDEX_NAME")
        
        # OpenAI setup for embeddings
        self.openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.openai_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
        self.openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")

        # Initialize clients
        self._init_clients()
    
    def _init_clients(self):
        """Initialize Azure clients."""
        # Document Intelligence client
        self.doc_intelligence_client = DocumentIntelligenceClient(
            endpoint=self.doc_intelligence_endpoint,
            credential=AzureKeyCredential(self.doc_intelligence_key)
        )
        
        # Search client
        from azure.identity import DefaultAzureCredential
        if self.search_key:
            search_credential = AzureKeyCredential(self.search_key)
        else:
            search_credential = DefaultAzureCredential()
            
        self.search_client = SearchClient(
            endpoint=self.search_endpoint,
            index_name=self.index_name,
            credential=search_credential
        )
        
        # OpenAI client
        self.openai_client = AzureOpenAI(
            api_version=self.openai_api_version,
            azure_endpoint=self.openai_endpoint,
            api_key=self.openai_key
        )
    
    def upload_document(
        self,
        file_path: str = None,
        file_content: BinaryIO = None,
        file_name: str = None,
        doc_name: Optional[str] = None,
        document_type: str = "IR_REPORT",
        industry: Optional[str] = None,
        company: Optional[str] = None,
        report_year: Optional[str] = None,
        author: Optional[str] = None,
        force_upload: bool = False
    ) -> Dict[str, Any]:
        """
        Upload and process PDF document using Document Intelligence and store in Azure AI Search.
        
        Args:
            file_path: Path to the PDF file
            file_content: Binary content of the file (alternative to file_path)
            file_name: Name of the file (required if using file_content)
            doc_name: Document name (defaults to filename)
            document_type: Type of document (IR_REPORT, MARKET_RESEARCH, etc.)
            industry: Industry category
            company: Company name
            report_year: Year of the report
            author: Document author
            force_upload: Force upload even if document exists
        
        Returns:
            Dict containing upload status and document metadata
        """
        try:
            # Determine file source and name
            if file_path:
                file_name = os.path.basename(file_path)
                doc_id = hashlib.md5(file_path.encode()).hexdigest()
            elif file_content and file_name:
                doc_id = hashlib.md5(file_name.encode()).hexdigest()
            else:
                return {
                    "status": "error",
                    "message": "Either file_path or (file_content + file_name) must be provided"
                }
            
            # Check if document already exists
            if not force_upload:
                try:
                    existing_doc = self.search_client.get_document(key=doc_id)
                    return {
                        "status": "already_exists",
                        "doc_id": doc_id,
                        "message": f"Document {file_name} already exists in the index",
                        "existing_doc": dict(existing_doc)
                    }
                except Exception:
                    # Document doesn't exist, proceed with upload
                    pass
            
            print(f"Processing document: {file_name}")
            
            # Step 1: Process PDF with Document Intelligence
            with open(file_path, "rb") as file:
                file_bytes = file.read()
                poller = self.doc_intelligence_client.begin_analyze_document(
                    model_id="prebuilt-layout",
                    body=file, 
                    output_content_format=DocumentContentFormat.MARKDOWN
                )

            # Analyze document using the layout model to extract markdown
            poller = self.doc_intelligence_client.begin_analyze_document(
                model_id="prebuilt-layout",
                body=file, 
                output_content_format=DocumentContentFormat.MARKDOWN
            )
            
            result = poller.result()
            
            # Extract content and metadata
            content = result.content if result.content else ""
            content_ko = content  # TODO consider to use gpt models for translation
            page_count = len(result.pages) if result.pages else 0
            
            # TODO Consider page level chunking (can be extended by length/token if needed)
                # chunks = []
                # for p in result.paragraphs or []:
                #     page = p.bounding_regions[0].page_number if p.bounding_regions else None
                #     chunks.append({
                #         "page": page,
                #         "content": p.content,
                #         "offset": p.spans[0].offset,
                #         "length": p.spans[0].length
                #     })
                # #
            
            # Step 2: Generate embeddings and process content
            summary = self._create_summary(content)
            keywords = self._extract_keywords_from_content(content)
            
            # Generate embeddings
            content_vector = self._generate_embedding(content[:8000])  # Limit for embedding
            summary_vector = self._generate_embedding(summary)
            
            # Step 3: Prepare document metadata
            if file_path:
                file_stats = os.stat(file_path)
                file_size = file_stats.st_size
                last_modified = datetime.datetime.fromtimestamp(
                    file_stats.st_mtime, datetime.timezone.utc
                ).isoformat()
            else:
                file_size = len(file_bytes)
                last_modified = datetime.datetime.now(datetime.timezone.utc).isoformat()
            
            current_time = datetime.datetime.now(datetime.timezone.utc)
            
            # TODO semantic tagging with gpt models
            document_tags = []
            target_audience = []
            topics = []
            
            document = {
                "docId": doc_id,
                "docName": doc_name or os.path.splitext(file_name)[0],
                "fileName": file_name,
                "fileSize": file_size,
                "uploadDate": current_time.isoformat(),
                "lastModified": last_modified,
                "fileType": "PDF",
                "content": content,
                "contentKo": content_ko,  
                "documentType": document_type,
                "industry": industry or "Unknown",
                "company": company or "Unknown",
                "reportYear": report_year or str(current_time.year),
                "summary": summary,
                "keywords": keywords,
                "pageCount": page_count,
                "author": author or "Unknown",
                "sourceUrl": "",
                "sourcePath": file_path or f"uploaded_{file_name}",
                "contentVector": content_vector,
                "summaryVector": summary_vector,
                "documentTags": document_tags or [],
                "targetAudience": target_audience or [],
                "topics": topics or [],
            }
            
            # Step 4: write to local json for debugging
            with open("document.json", "w", encoding="utf-8") as f:
                json.dump(document, f, ensure_ascii=False, indent=2)

            # Step 5: Upload to Azure AI Search
            self.search_client.upload_documents([document])
            
            print(f"✅ Document {file_name} uploaded successfully!")
            
            return {
                "status": "success",
                "doc_id": doc_id,
                "doc_name": document["docName"],
                "message": f"Document {file_name} processed and uploaded successfully",
                "page_count": page_count,
                "content_length": len(content),
                "metadata": {
                    "document_type": document_type,
                    "industry": industry,
                    "company": company,
                    "report_year": report_year
                }
            }
            
        except Exception as e:
            error_msg = f"Error processing document {file_name if file_name else 'unknown'}: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "status": "error",
                "message": error_msg
            }
    
    def _create_summary(self, content: str, max_length: int = 2000) -> str:
        """
        Create a summary from content.
        TODO consider to use gpt models
        """
        if len(content) <= max_length:
            return content
        
        # Simple summary - take first paragraphs up to max_length
        paragraphs = content.split('\n\n')
        summary = ""
        for paragraph in paragraphs:
            if len(summary + paragraph) <= max_length - 3:
                summary += paragraph + "\n\n"
            else:
                break
        
        return summary.strip() + "..." if summary else content[:max_length] + "..."
    
    def _extract_keywords_from_content(self, content: str, max_keywords: int = 20) -> str:
        """
        Extract keywords from document content.
        TODO consider to use gpt models
        """
        # Simple keyword extraction
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        }
        
        # Extract words (letters only, minimum 3 characters)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
        
        # Filter out stop words and count frequency
        meaningful_words = [word for word in words if word not in stop_words]
        word_counts = Counter(meaningful_words)
        
        # Get top keywords
        top_keywords = [word for word, _ in word_counts.most_common(max_keywords)]
        
        return ", ".join(top_keywords)
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using Azure OpenAI."""
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model=self.embedding_deployment
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Warning: Failed to generate embedding: {str(e)}")
            # Return a zero vector as fallback
            return [0.0] * 1536  # Default embedding dimension
    
    def check_document_exists(self, file_path: str = None, file_name: str = None) -> Dict[str, Any]:
        """Check if a document already exists in the index."""
        try:
            if file_path:
                doc_id = hashlib.md5(file_path.encode()).hexdigest()
            elif file_name:
                doc_id = hashlib.md5(file_name.encode()).hexdigest()
            else:
                return {"status": "error", "message": "Either file_path or file_name required"}
            
            document = self.search_client.get_document(key=doc_id)
            return {
                "status": "exists",
                "doc_id": doc_id,
                "document": dict(document)
            }
        except Exception:
            return {
                "status": "not_found",
                "doc_id": doc_id if 'doc_id' in locals() else None
            }


# Convenience functions for backward compatibility
def upload_doc_plugin(file_path: str, **kwargs) -> Dict[str, Any]:
    """Convenience function to upload a document."""
    plugin = UploadDocPlugin()
    return plugin.upload_document(file_path=file_path, **kwargs)


def check_document_exists(file_path: str) -> Dict[str, Any]:
    """Convenience function to check if document exists."""
    plugin = UploadDocPlugin()
    return plugin.check_document_exists(file_path=file_path)
