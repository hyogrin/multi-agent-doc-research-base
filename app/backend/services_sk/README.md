# Document Intelligence and AI Search Integration

This module provides a comprehensive solution for processing PDF documents using Azure Document Intelligence and storing them in Azure AI Search for intelligent retrieval.

## Features

- **PDF Processing**: Automatically converts PDF files to markdown using Azure Document Intelligence
- **Smart Upload Management**: Checks for existing documents to avoid duplicates
- **Multiple Search Methods**: Hybrid, semantic, vector, and text search capabilities
- **Rich Metadata Support**: Designed for IR reports and market research documents
- **Batch Processing**: Upload multiple documents efficiently
- **Comprehensive Filtering**: Filter by company, industry, document type, and more

## Components

### 1. upload_doc_plugin.py
Handles PDF document processing and upload to Azure AI Search using Document Intelligence.

**Key Features:**
- Document Intelligence integration for PDF-to-markdown conversion
- Automatic embedding generation
- Duplicate detection
- Metadata extraction and enrichment

### 2. ai_search_plugin.py
Provides advanced search capabilities for documents in Azure AI Search.

**Key Features:**
- Hybrid search (combines text and vector search)
- Semantic search with ranking
- Vector similarity search
- Traditional text search
- Advanced filtering and faceting

### 3. document_processing_service.py
Unified service that combines upload and search functionality for complete workflows.

**Key Features:**
- End-to-end document processing
- Batch document upload
- Search with summarization
- Index statistics and management

## Setup

### Environment Variables

Add these variables to your `.env` file:

```bash
# Azure AI Search
AZURE_AI_SEARCH_ENDPOINT=https://your-search-service.search.windows.net
AZURE_AI_SEARCH_API_KEY=your-search-api-key
AZURE_SEARCH_INDEX_NAME=document_inquiry_index

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-openai-service.openai.azure.com
AZURE_OPENAI_API_KEY=your-openai-api-key
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-3-small
AZURE_OPENAI_EMBEDDING_DIMENSIONS=1536
AZURE_OPENAI_API_VERSION=2024-12-01-preview

# Azure Document Intelligence
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://your-document-intelligence.cognitiveservices.azure.com
AZURE_DOCUMENT_INTELLIGENCE_API_KEY=your-document-intelligence-api-key
```

### Required Packages

```bash
pip install azure-ai-documentintelligence azure-search-documents azure-identity openai
```

## Usage Examples

### 1. Upload a Single Document

```python
from services_sk.upload_doc_plugin import upload_doc_plugin

# Upload a PDF document
result = upload_doc_plugin(
    file_path="/path/to/your/report.pdf",
    doc_name="Q3 2024 Financial Report",
    document_type="IR_REPORT",
    industry="Technology",
    company="Samsung Electronics",
    report_year="2024",
    author="IR Team",
    target_audience=["investors", "analysts", "stakeholders"],
    topics=["financial_performance", "quarterly_results", "revenue"],
    document_tags=["earnings", "Q3", "financial"]
)

print(f"Upload status: {result['status']}")
```

### 2. Search Documents

```python
from services_sk.ai_search_plugin import ai_search_plugin

# Hybrid search
results = ai_search_plugin(
    query="Samsung financial performance Q3 2024",
    search_type="hybrid",
    company="Samsung Electronics",
    top_k=5
)

print(f"Found {results['total_results']} documents")
for doc in results['documents']:
    print(f"- {doc['docName']} (Score: {doc.get('search_score', 'N/A')})")
```

### 3. Complete Workflow

```python
from services_sk.document_processing_service import DocumentProcessingService

service = DocumentProcessingService()

# Process PDF and search in one call
result = service.process_pdf_and_search(
    file_path="/path/to/your/report.pdf",
    query="What were the key financial highlights?",
    doc_name="Q3 2024 Earnings Report",
    company="Samsung Electronics",
    industry="Technology",
    search_type="hybrid"
)

print(f"Workflow status: {result['workflow_status']}")
if result['search_results']:
    print(f"Found {result['search_results']['total_results']} relevant documents")
```

### 4. Batch Upload

```python
from services_sk.document_processing_service import DocumentProcessingService

service = DocumentProcessingService()

file_paths = [
    "/path/to/report1.pdf",
    "/path/to/report2.pdf",
    "/path/to/report3.pdf"
]

metadata = [
    {"company": "Samsung", "document_type": "IR_REPORT", "report_year": "2024"},
    {"company": "LG", "document_type": "MARKET_RESEARCH", "report_year": "2024"},
    {"company": "SK", "document_type": "IR_REPORT", "report_year": "2024"}
]

results = service.upload_multiple_documents(
    file_paths=file_paths,
    document_metadata=metadata,
    batch_size=2
)

print(f"Successful uploads: {len(results['successful_uploads'])}")
```

### 5. Search with Filters

```python
from services_sk.ai_search_plugin import AISearchPlugin

plugin = AISearchPlugin()

# Search with multiple filters
results = plugin.search_documents(
    query="revenue growth and market analysis",
    search_type="semantic",
    document_type="IR_REPORT",
    company="Samsung Electronics",
    report_year="2024",
    topics=["financial_performance", "growth"],
    top_k=3
)

print(f"Found {results['total_results']} matching documents")
```

## Document Schema

The system uses the following schema for documents:

```json
{
  "docId": "unique_document_id",
  "docName": "Human-readable document name",
  "fileName": "original_file_name.pdf",
  "fileSize": 1234567,
  "uploadDate": "2024-01-01T00:00:00Z",
  "lastModified": "2024-01-01T00:00:00Z",
  "fileType": "PDF",
  "content": "Full document content in markdown",
  "contentKo": "Korean content (if applicable)",
  "documentType": "IR_REPORT|MARKET_RESEARCH|WHITEPAPER",
  "industry": "Technology|Finance|Healthcare|etc",
  "company": "Company name",
  "reportYear": "2024",
  "summary": "Document summary",
  "keywords": "extracted, keywords, from, content",
  "pageCount": 25,
  "author": "Document author",
  "sourceUrl": "Original URL (if applicable)",
  "sourcePath": "File system path",
  "contentVector": [0.1, 0.2, ...],
  "summaryVector": [0.1, 0.2, ...],
  "documentTags": ["tag1", "tag2"],
  "targetAudience": ["investors", "analysts"],
  "topics": ["topic1", "topic2"]
}
```

## Search Types

### 1. Hybrid Search
Combines text and vector search for best relevance:
```python
ai_search_plugin(query="financial performance", search_type="hybrid")
```

### 2. Semantic Search
Uses semantic understanding for better context matching:
```python
ai_search_plugin(query="revenue growth", search_type="semantic")
```

### 3. Vector Search
Pure similarity search using embeddings:
```python
ai_search_plugin(query="market analysis", search_type="vector")
```

### 4. Text Search
Traditional keyword-based search:
```python
ai_search_plugin(query="Q3 earnings", search_type="text")
```

## Filtering Options

You can filter documents by:
- `document_type`: IR_REPORT, MARKET_RESEARCH, etc.
- `industry`: Technology, Finance, Healthcare, etc.
- `company`: Company name
- `report_year`: Year of the report
- `target_audience`: Target audience tags
- `topics`: Document topics

## Best Practices

1. **Document Naming**: Use descriptive names for documents to improve searchability
2. **Metadata Completeness**: Provide as much metadata as possible for better filtering
3. **Batch Processing**: Use batch upload for multiple documents to improve efficiency
4. **Search Type Selection**: 
   - Use **hybrid** for general queries
   - Use **semantic** for conceptual searches
   - Use **vector** for similarity-based searches
   - Use **text** for exact keyword matches
5. **Content Preparation**: Ensure PDFs are text-based (not scanned images) for best results

## Troubleshooting

### Common Issues

1. **Document Intelligence Errors**
   - Ensure your PDF is text-based, not a scanned image
   - Check Document Intelligence service limits and quotas
   - Verify endpoint and API key configuration

2. **Search Errors**
   - Ensure the AI Search index exists and is properly configured
   - Check search service permissions and authentication
   - Verify embedding model deployment is active

3. **Upload Failures**
   - Check file permissions and accessibility
   - Verify all required environment variables are set
   - Ensure sufficient storage quota in AI Search service

### Error Handling

All functions return status information in their response:
```python
result = upload_doc_plugin(file_path="document.pdf")
if result["status"] == "error":
    print(f"Error: {result['message']}")
```

## Performance Considerations

- **Document Size**: Large documents may take longer to process
- **Batch Size**: Adjust batch size based on service limits and performance
- **Embedding Generation**: Vector generation adds processing time but improves search quality
- **Index Size**: Larger indexes may have slightly slower search performance

## Integration with Existing Systems

This module is designed to integrate seamlessly with existing chatbot and document inquiry systems. The plugins can be called from orchestration services or used in API endpoints for document management workflows.
