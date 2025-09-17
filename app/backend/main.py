from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import logging
import os
import tempfile
import uuid
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Optional
from utils.enum import SearchEngine
from config.config import Settings
from model.models import ChatRequest, ChatResponse, PlanSearchRequest, FileUploadResponse
from services.orchestrator import Orchestrator
from services.plan_executor import PlanExecutor
from services_sk.plan_search_executor_sk import PlanSearchExecutorSK
from services_sk.unified_file_upload_plugin import UnifiedFileUploadPlugin
from services.search_crawler import GoogleSearchCrawler, BingSearchCrawler
from services.bing_grounding_search import BingGroundingSearch, BingGroundingCrawler
from services.query_rewriter import QueryRewriter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Microsoft General Inquiry Chatbot",
    description="AI-powered chatbot for Microsoft product inquiries using Azure OpenAI",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

settings = Settings()

# Global upload status tracking (in production, use Redis or database)
upload_status_tracker: Dict[str, Dict] = {}

@app.router.lifespan_context
async def lifespan(app: FastAPI):
    logger.info("Starting up Microsoft Chatbot API...")
    
    yield
    
    logger.info("Shutting down Microsoft Chatbot API...")


@app.get("/health")
async def health_check():
    return {"status": "ok"}


async def update_upload_status(upload_id: str, status: str, message: str = "", progress: int = 0, file_results: List = None):
    """Update upload status in tracker"""
    if upload_id in upload_status_tracker:
        upload_status_tracker[upload_id].update({
            "status": status,
            "message": message,
            "progress": progress,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "file_results": file_results or upload_status_tracker[upload_id].get("file_results", [])
        })

async def process_uploaded_files_background(
    upload_id: str,
    file_paths: List[str],
    file_names: List[str],  # 실제 파일명 리스트 추가
    document_type: str,
    company: str,
    industry: str,
    report_year: str,
    force_upload: bool
):
    """Background task to process uploaded files with status tracking."""
    try:
        import json
        
        # Update status to processing
        await update_upload_status(upload_id, "processing", "파일 처리 중...", 10)
        
        upload_plugin = UnifiedFileUploadPlugin()
        
        # Update status before upload
        await update_upload_status(upload_id, "processing", "벡터 데이터베이스에 업로드 중...", 50)
        
        # Create file mapping for actual filenames
        file_mapping = {}
        for temp_path, original_name in zip(file_paths, file_names):
            file_mapping[temp_path] = original_name
        
        result = await upload_plugin.upload_files(
            file_paths=json.dumps(file_paths),
            file_names=json.dumps(file_names),  # 실제 파일명도 전달
            document_type=document_type,
            company=company,
            industry=industry,
            report_year=report_year,
            force_upload=str(force_upload).lower()
        )
        
        logger.info(f"Background file processing completed: {result}")
        
        # Parse result and update status
        if isinstance(result, str):
            import json
            result_data = json.loads(result)
        else:
            result_data = result
        
        if result_data.get("status") == "completed":
            await update_upload_status(
                upload_id, 
                "completed", 
                f"업로드 완료! {result_data.get('successful_uploads', 0)}개 파일 성공", 
                100,
                result_data.get("results", [])
            )
        else:
            await update_upload_status(
                upload_id, 
                "error", 
                f"업로드 실패: {result_data.get('message', 'Unknown error')}", 
                100,
                result_data.get("results", [])
            )
        
        # Clean up temporary files
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.warning(f"Failed to remove temp file {file_path}: {e}")
                
    except Exception as e:
        logger.error(f"Background file processing failed: {str(e)}")
        await update_upload_status(upload_id, "error", f"처리 중 오류 발생: {str(e)}", 100)


@app.post("/upload_documents", response_model=FileUploadResponse)
async def upload_documents_endpoint(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    document_type: str = Form("IR_REPORT"),
    company: str = Form(None),
    industry: str = Form(None),
    report_year: str = Form(None),
    force_upload: bool = Form(False)
):
    """
    Upload multiple documents for processing and vector storage.
    Documents are processed in the background after initial validation.
    """
    # Generate unique upload ID
    upload_id = str(uuid.uuid4())
    
    # Validate file count
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 files allowed per upload"
        )
    
    # Validate file types
    allowed_extensions = {'.pdf', '.docx', '.txt'}
    uploaded_files = []
    temp_file_paths = []
    
    try:
        temp_file_paths = []
        original_filenames = []  # 실제 파일명 저장
        
        for file in files:
            # Check file extension
            file_ext = os.path.splitext(file.filename)[1].lower()
            if file_ext not in allowed_extensions:
                raise HTTPException(
                    status_code=400,
                    detail=f"File type {file_ext} not supported. Allowed: {', '.join(allowed_extensions)}"
                )
            
            # Save original filename
            original_filenames.append(file.filename)
            
            # Save file to temporary location
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
            temp_file_path = temp_file.name
            
            # Read and save file content
            content = await file.read()
            temp_file.write(content)
            temp_file.close()
            
            temp_file_paths.append(temp_file_path)
            uploaded_files.append({
                "filename": file.filename,
                "file_path": temp_file_path,
                "file_size": len(content),
                "content_type": file.content_type
            })
        
        # Initialize upload status
        upload_status_tracker[upload_id] = {
            "upload_id": upload_id,
            "status": "initialized",
            "message": f"파일 {len(files)}개 업로드 시작",
            "progress": 0,
            "total_files": len(files),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "files": [f["filename"] for f in uploaded_files],
            "file_results": []
        }
        
        # Add background task for processing
        background_tasks.add_task(
            process_uploaded_files_background,
            upload_id=upload_id,
            file_paths=temp_file_paths,
            file_names=original_filenames,  # 실제 파일명 전달
            document_type=document_type,
            company=company,
            industry=industry,
            report_year=report_year,
            force_upload=force_upload
        )
        
        return FileUploadResponse(
            status="processing",
            total_files=len(files),
            successful_uploads=0,
            failed_uploads=0,
            results=[],
            message=f"Files uploaded successfully. Processing {len(files)} files in background.",
            upload_id=upload_id  # Return upload_id for status tracking
        )
        
    except Exception as e:
        # Clean up temp files on error
        for temp_path in temp_file_paths:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass
        
        logger.error(f"File upload error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"File upload failed: {str(e)}"
        )

@app.get("/upload_status/{upload_id}")
async def get_upload_status(upload_id: str):
    """Get upload status by upload ID"""
    if upload_id not in upload_status_tracker:
        raise HTTPException(
            status_code=404,
            detail="Upload ID not found"
        )
    
    return upload_status_tracker[upload_id]

@app.get("/upload_status")
async def list_upload_status():
    """Get all upload statuses"""
    return {
        "total_uploads": len(upload_status_tracker),
        "uploads": list(upload_status_tracker.values())
    }

@app.get("/plan_search")
async def plan_search_info():
    """Get information about the plan_search endpoint"""
    return {
        "endpoint": "/plan_search",
        "method": "POST",
        "description": "Plan Search endpoint for processing chat requests",
        "required_fields": ["messages"],
        "optional_fields": ["max_tokens", "temperature", "query_rewrite", "planning", "search_engine", "stream", "locale"],
        "example_usage": "Send POST request with JSON payload containing messages array",
        "error": "This endpoint only accepts POST requests. Please use POST method with proper JSON payload."
    }

@app.post("/plan_search", response_model=ChatResponse)
async def plan_search_endpoint(
    request: PlanSearchRequest, 
):
    plan_search_executor = None
    try:
        plan_search_executor = PlanSearchExecutorSK(settings)
        
        if request.stream:
            return StreamingResponse(
                plan_search_executor.generate_response(
                    request.messages,
                    request.max_tokens,
                    request.temperature,
                    request.query_rewrite,
                    request.planning,
                    request.search_engine,  
                    stream=True,
                    elapsed_time=True,
                    locale=request.locale,
                    include_web_search=request.include_web_search,
                    include_ytb_search=request.include_ytb_search,
                    include_mcp_server=request.include_mcp_server,
                    include_ai_search=request.include_ai_search,
                    verbose=request.verbose,
                ),
                media_type="text/event-stream"
            )
        
        response_generator = plan_search_executor.generate_response(
            request.messages,
            request.max_tokens,
            request.temperature,
            request.query_rewrite,
            request.planning,
            request.search_engine,
            stream=False,
            elapsed_time=True,
            locale=request.locale,
            include_web_search=request.include_web_search,
            include_ytb_search=request.include_ytb_search,
            include_mcp_server=request.include_mcp_server,
            include_ai_search=request.include_ai_search,
            verbose=request.verbose,
        )
        
        response = await response_generator.__anext__()
        
        return ChatResponse(
            message=response,
            success=True
        )
    except Exception as e:
        logger.error(f"Error processing risk search request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate risk analysis response: {str(e)}"
        )
    


@app.post("/deep_search", response_model=ChatResponse)
async def deep_search_endpoint(
    request: ChatRequest, 
):
    try:
        plan_executor = PlanExecutor(settings)
        
        redis_config = {
            "host": settings.REDIS_HOST,
            "port": settings.REDIS_PORT,
            "password": settings.REDIS_PASSWORD,
            "db": settings.REDIS_DB,
            "decode_responses": True
        }
        
        search_crawler = None
        
        if request.search_engine == SearchEngine.GOOGLE_SEARCH_CRAWLING:
            search_crawler = GoogleSearchCrawler(redis_config=redis_config)
        elif request.search_engine == SearchEngine.BING_SEARCH_CRAWLING:
            search_crawler = BingSearchCrawler(redis_config=redis_config)
        elif request.search_engine == SearchEngine.BING_GROUNDING_CRAWLING:
            search_crawler = BingGroundingCrawler(redis_config=redis_config)    
        
        query_rewriter = QueryRewriter(client=plan_executor.client, settings=settings)
        plan_executor.query_rewriter = query_rewriter
        
        if request.stream:
            return StreamingResponse(
                plan_executor.generate_response(
                    request.messages,
                    request.max_tokens,
                    request.temperature,
                    request.query_rewrite,
                    request.search_engine,
                    search_crawler=search_crawler,
                    stream=True,
                    elapsed_time=True,
                    locale=request.locale
                ),
                media_type="text/event-stream"
            )
        
        response_generator = plan_executor.generate_response(
            request.messages,
            request.max_tokens,
            request.temperature,
            request.query_rewrite,
            request.search_engine,
            search_crawler=search_crawler,
            stream=False,
            elapsed_time=True,
            locale=request.locale
        )
        
        response = await response_generator.__anext__()
        
        return ChatResponse(
            message=response,
            success=True
        )
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate response: {str(e)}"
        )
    

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest, 
):
    try:
        orchestrator = Orchestrator(settings)
        
        redis_config = {
            "host": settings.REDIS_HOST,
            "port": settings.REDIS_PORT,
            "password": settings.REDIS_PASSWORD,
            "db": settings.REDIS_DB,
            "decode_responses": True
        }

        #TODO : Refactor to use orchestrator.search_crawler
        search_crawler = None
        
        logger.debug(f"request.search_engine: {request.search_engine}")
        if request.search_engine == SearchEngine.GOOGLE_SEARCH_CRAWLING:
            search_crawler = GoogleSearchCrawler(redis_config=redis_config)
        elif request.search_engine == SearchEngine.BING_SEARCH_CRAWLING:
            search_crawler = BingSearchCrawler(redis_config=redis_config)
        elif request.search_engine == SearchEngine.BING_GROUNDING_CRAWLING:
            search_crawler = BingGroundingCrawler(redis_config=redis_config)

        bing_grounding_search = BingGroundingSearch(redis_config=redis_config)
        query_rewriter = QueryRewriter(client=orchestrator.client, settings=settings)

        orchestrator.bing_grounding_search = bing_grounding_search
        orchestrator.query_rewriter = query_rewriter
        
        if request.stream:
            return StreamingResponse(
                orchestrator.generate_response(
                    request.messages,
                    request.max_tokens,
                    request.temperature,
                    request.query_rewrite,
                    request.search_engine,
                    search_crawler=search_crawler,
                    stream=True,
                    elapsed_time=True,
                    locale=request.locale
                ),
                media_type="text/event-stream"
            )
        
        response_generator = orchestrator.generate_response(
            request.messages,
            request.max_tokens,
            request.temperature,
            request.query_rewrite,
            request.search_engine,
            search_crawler=search_crawler, 
            stream=False,
            elapsed_time=True,
            locale=request.locale
        )
        
        response = await response_generator.__anext__()
        
        return ChatResponse(
            message=response,
            success=True
        )
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate response: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again later."}
    )
