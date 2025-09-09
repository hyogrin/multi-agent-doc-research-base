from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import logging
from utils.enum import SearchEngine
from config.config import Settings
from model.models import ChatRequest, ChatResponse, PlanSearchRequest
from services.orchestrator import Orchestrator
from services.plan_executor import PlanExecutor
from services_sk.plan_search_executor_sk import PlanSearchExecutorSK
from services_sk.plan_search_executor_sk_parallel import PlanSearchExecutorSKParallel
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

@app.router.lifespan_context
async def lifespan(app: FastAPI):
    logger.info("Starting up Microsoft Chatbot API...")
    
    yield
    
    logger.info("Shutting down Microsoft Chatbot API...")


@app.get("/health")
async def health_check():
    return {"status": "ok"}

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
    
@app.post("/plan_search_parallel", response_model=ChatResponse)
async def plan_search_parallel_endpoint(
    request: PlanSearchRequest, 
):
    plan_search_executor = None
    try:
        plan_search_executor = PlanSearchExecutorSKParallel(settings)
        
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
