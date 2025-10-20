"""
AI Search Executor for Document Retrieval using Microsoft Agent Framework.

This executor provides various search capabilities for documents stored in Azure AI Search,
including hybrid search, semantic search, vector search, and traditional text search.
Migrated from services_sk/ai_search_plugin.py.
"""

import os
import json
import base64
import logging
from typing import List, Dict, Any, Optional, Annotated

from agent_framework import Executor, WorkflowContext, handler, ai_function
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import (
    VectorizedQuery,
    QueryType,
    QueryCaptionType,
    QueryAnswerType
)
from openai import AzureOpenAI
from i18n.locale_msg import LOCALE_MESSAGES

logger = logging.getLogger(__name__)


class AISearchExecutor(Executor):
    """
    Executor for searching documents in Azure AI Search with multiple search methods.
    
    This executor supports:
    - Hybrid search (text + vector)
    - Semantic search
    - Vector search
    - Traditional text search
    """
    
    def __init__(
        self,
        id: str,
        search_endpoint: Optional[str] = None,
        search_key: Optional[str] = None,
        index_name: Optional[str] = None,
        openai_endpoint: Optional[str] = None,
        openai_key: Optional[str] = None,
        embedding_deployment: Optional[str] = None,
        openai_api_version: Optional[str] = None,
    ):
        """
        Initialize the AI search executor with required clients.
        
        Args:
            id: Executor ID
            search_endpoint: Azure AI Search endpoint
            search_key: Azure AI Search API key
            index_name: Azure AI Search index name
            openai_endpoint: Azure OpenAI endpoint for embeddings
            openai_key: Azure OpenAI API key
            embedding_deployment: Azure OpenAI embedding deployment name
            openai_api_version: Azure OpenAI API version
        """
        super().__init__(id=id)
        
        # AI Search setup
        self.search_endpoint = search_endpoint or os.getenv("AZURE_AI_SEARCH_ENDPOINT")
        self.search_key = search_key or os.getenv("AZURE_AI_SEARCH_API_KEY")
        self.index_name = index_name or os.getenv("AZURE_AI_SEARCH_INDEX_NAME")
        
        # OpenAI setup for embeddings
        self.openai_endpoint = openai_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.openai_key = openai_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.embedding_deployment = embedding_deployment or os.getenv(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"
        )
        self.openai_api_version = openai_api_version or os.getenv("AZURE_OPENAI_API_VERSION")
        
        # Initialize clients
        self._init_clients()
        
        logger.info(f"AISearchExecutor initialized with index: {self.index_name}")
    
    def _init_clients(self):
        """Initialize Azure clients."""
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
    
    @handler
    async def search_documents(
        self,
        search_data: Dict[str, Any],
        ctx: WorkflowContext[Dict[str, Any], str]  # Added str for yield_output
    ) -> None:
        """
        Search documents in Azure AI Search.
        
        Args:
            search_data: Dictionary with search parameters:
                - query: Search query string
                - search_type: "hybrid", "semantic", "vector", or "text"
                - filters: Optional OData filter expression
                - top_k: Number of results (default: 5)
                - include_content: Include full content (default: True)
                - document_type: Filter by document type
                - industry: Filter by industry
                - company: Filter by company
                - report_year: Filter by report year
            ctx: Workflow context for sending results
        """
        try:
            # Get metadata for verbose and locale
            metadata = search_data.get("metadata", {})
            locale = metadata.get("locale", "ko-KR")
            verbose = metadata.get("verbose", False)
            LOCALE_MSG = LOCALE_MESSAGES.get(locale, LOCALE_MESSAGES["ko-KR"])
            
            query = search_data.get("query", "")
            search_type = search_data.get("search_type", "hybrid")
            filters = search_data.get("filters")
            top_k = search_data.get("top_k", 5)
            include_content = search_data.get("include_content", True)
            document_type = search_data.get("document_type")
            industry = search_data.get("industry")
            company = search_data.get("company")
            report_year = search_data.get("report_year")
            
            # ✅ Yield starting message
            await ctx.yield_output(f"data: ### {LOCALE_MSG['searching_ai_search']}\n\n")
            
            logger.info(f"[AISearchExecutor] Searching for: {query} (type: {search_type})")
            
            # Generate query vector
            query_vector = self._generate_embedding(query)
            
            # Build filter expression
            filter_expression = self._build_filters(
                filters, document_type, industry, company, report_year
            )
            
            # Execute search
            search_results = self._execute_search(
                query=query,
                query_vector=query_vector,
                search_type=search_type,
                filter_expression=filter_expression,
                top_k=top_k,
                include_content=include_content
            )
            
            # Process results
            documents = self._process_search_results(search_results, include_content)
            
            result_data = {
                "query": query,
                "search_type": search_type,
                "documents": documents,
                "total_results": len(documents)
            }
            
            logger.info(f"[AISearchExecutor] Found {len(documents)} documents")
            
            # ✅ Yield completion message (SK compatible format with results)
            if verbose and documents:
                results_str = json.dumps(result_data, ensure_ascii=False, indent=2)
                truncated = results_str[:200] + "... [truncated for display]" if len(results_str) > 200 else results_str
                encoded_code = base64.b64encode(f"```json\n{truncated}\n```".encode('utf-8')).decode('utf-8')
                await ctx.yield_output(f"data: {LOCALE_MSG['ai_search_context_done']}#code#{encoded_code}\n\n")
            else:
                await ctx.yield_output(f"data: ### {LOCALE_MSG['ai_search_context_done']}\n\n")
            
            # Send results to next executor
            await ctx.send_message({
                **search_data,
                "ai_search_results": result_data
            })
            
        except Exception as e:
            error_msg = f"AI Search failed: {str(e)}"
            logger.error(f"[AISearchExecutor] {error_msg}")
            await ctx.send_message({
                **search_data,
                "ai_search_results": {
                    "error": error_msg,
                    "query": search_data.get("query", ""),
                    "documents": []
                }
            })
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using Azure OpenAI."""
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model=self.embedding_deployment
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return []
    
    def _build_filters(
        self,
        filters: Optional[str],
        document_type: Optional[str],
        industry: Optional[str],
        company: Optional[str],
        report_year: Optional[str]
    ) -> Optional[str]:
        """Build OData filter expression."""
        filter_parts = []
        
        if filters:
            filter_parts.append(filters)
        if document_type:
            filter_parts.append(f"document_type eq '{document_type}'")
        if industry:
            filter_parts.append(f"industry eq '{industry}'")
        if company:
            filter_parts.append(f"company eq '{company}'")
        if report_year:
            filter_parts.append(f"report_year eq '{report_year}'")
        
        return " and ".join(filter_parts) if filter_parts else None
    
    def _execute_search(
        self,
        query: str,
        query_vector: List[float],
        search_type: str,
        filter_expression: Optional[str],
        top_k: int,
        include_content: bool
    ):
        """Execute search based on search type."""
        select_fields = self._get_select_fields(include_content)

        vector_queries = [
            VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=top_k,
                fields="content_vector"
            ),
            VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=top_k,
                fields="summary_vector"
            )
        ]
        
        if search_type == "hybrid":
            # Hybrid search: text + vector
            return self.search_client.search(
                search_text=query,
                vector_queries=vector_queries,
                filter=filter_expression,
                select=select_fields,
                top=top_k,
                query_type=QueryType.SEMANTIC,
                semantic_configuration_name="default"
            )
            
        elif search_type == "semantic":
            # Semantic search with captions
            return self.search_client.search(
                search_text=query,
                filter=filter_expression,
                select=select_fields,
                top=top_k,
                query_type=QueryType.SEMANTIC,
                semantic_configuration_name="default",
                query_caption=QueryCaptionType.EXTRACTIVE,
                query_answer=QueryAnswerType.EXTRACTIVE
            )
            
        elif search_type == "vector":
            # Pure vector search
            return self.search_client.search(
                search_text=None,
                vector_queries=vector_queries,
                filter=filter_expression,
                select=select_fields,
                top=top_k
            )
            
        elif search_type == "text":
            # Traditional text search
            return self.search_client.search(
                search_text=query,
                filter=filter_expression,
                select=select_fields,
                top=top_k
            )
        
        else:
            raise ValueError(f"Unknown search type: {search_type}")
    
    def _get_select_fields(self, include_content: bool) -> str:
        """Get select fields for search query."""
        base_fields = "docId,title,file_name,summary,document_type,industry,company,report_year,page_number,upload_date,keywords"
        
        if include_content:
            return f"{base_fields},content"
        else:
            return base_fields
    
    def _process_search_results(self, search_results, include_content: bool) -> List[Dict[str, Any]]:
        """Process search results into a standardized format."""
        documents = []
        
        for result in search_results:
            doc = {
                "docId": result.get("docId"),
                "title": result.get("title"),
                "file_name": result.get("file_name"),
                "summary": result.get("summary"),
                "document_type": result.get("document_type"),
                "industry": result.get("industry"),
                "company": result.get("company"),
                "report_year": result.get("report_year"),
                "page_number": result.get("page_number"),
                "upload_date": result.get("upload_date"),
                "keywords": result.get("keywords"),
                "score": result.get("@search.score"),
            }
            
            if include_content:
                doc["content"] = result.get("content")
            
            # Add semantic captions if available
            if hasattr(result, "@search.captions"):
                doc["captions"] = [
                    {"text": caption.text, "highlights": caption.highlights}
                    for caption in result["@search.captions"]
                ]
            
            documents.append(doc)
        
        return documents


# AI Function for document search (can be used as a tool by ChatAgents)
@ai_function(description="Search documents in Azure AI Search")
def search_documents(
    query: Annotated[str, "The search query"],
    search_type: Annotated[str, "Search type: hybrid, semantic, vector, or text"] = "hybrid",
    top_k: Annotated[int, "Number of results to return"] = 5,
) -> str:
    """
    Search documents in Azure AI Search.
    
    Args:
        query: The search query
        search_type: Type of search to perform
        top_k: Number of results to return
        
    Returns:
        JSON string containing search results
    """
    executor = AISearchExecutor(id="ai_search")
    
    # Create a simple context that collects results
    class SimpleContext:
        def __init__(self):
            self.results = None
        
        async def emit(self, data):
            self.results = data
    
    context = SimpleContext()
    
    # Run the search
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    search_data = {
        "query": query,
        "search_type": search_type,
        "top_k": top_k
    }
    
    loop.run_until_complete(executor.search_documents(search_data, context))
    
    return json.dumps(context.results, ensure_ascii=False, indent=2)
