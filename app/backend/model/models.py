from typing import List, Optional
from pydantic import BaseModel, Field
from utils.enum import SearchEngine


class ChatMessage(BaseModel):
    """Chat message model with role and content"""
    role: str = Field(..., description="Role of the message sender (system, user, assistant)")
    content: str = Field(..., description="Content of the message")
    files: Optional[List[dict]] = Field(None, description="Optional list of uploaded files")


class ChatRequest(BaseModel):
    """Chat request model with messages and optional parameters"""
    messages: List[ChatMessage] = Field(
        ..., 
        description="List of chat messages in the conversation history"
    )
    max_tokens: Optional[int] = Field(
        None, 
        description="Maximum number of tokens to generate"
    )
    temperature: Optional[float] = Field(
        None, 
        description="Temperature for response generation (0.0 to 1.0)"
    )
    stream: bool = Field(
        False,
        description="Enable streaming response"
    )
    query_rewrite: bool = Field(
        True,
        description="Enable query rewriting for better search results"
    )
    plan_execute: bool = Field(
        False,
        description="Enable plan and execute mode for complex queries"
    )
    search_engine: str = Field(
        SearchEngine.GOOGLE_SEARCH_CRAWLING,
        description="Search engine to use for retrieving information"
    )
    locale: str = Field(
        "ko-KR",
        description="Locale for the response"
    )


class PlanSearchRequest(BaseModel):
    """Risk search request model with additional DART-specific parameters"""
    messages: List[ChatMessage] = Field(
        ..., 
        description="List of chat messages in the conversation history"
    )
    max_tokens: Optional[int] = Field(
        None, 
        description="Maximum number of tokens to generate"
    )
    temperature: Optional[float] = Field(
        None, 
        description="Temperature for response generation (0.0 to 1.0)"
    )
    stream: bool = Field(
        False,
        description="Enable streaming response"
    )
    query_rewrite: bool = Field(
        True,
        description="Enable query rewriting for better search results"
    )
    planning: bool = Field(
        True,
        description="Enable search planning for better search results"
    )
    search_engine: SearchEngine = Field(
        SearchEngine.BING_SEARCH_CRAWLING,
        description="Search engine to use for retrieving information"
    )
    locale: str = Field(
        "ko-KR",
        description="Locale for the response"
    )
    include_web_search: bool = Field(
        True,
        description="Include web search results"
    )
    include_ytb_search: bool = Field(
        True,
        description="Include YouTube search results"
    )
    include_mcp_server: bool = Field(
        True,
        description="Include MCP server integration"
    )
    include_ai_search: bool = Field(
        True,
        description="Include AI search results from uploaded documents"
    )
    verbose: bool = Field(
        True,
        description="Verbose mode for detailed output"
    )
    

class ChatResponse(BaseModel):
    """Chat response model with message and success status"""
    message: str = Field(..., description="Response message from the chatbot")
    success: bool = Field(..., description="Indicates if the request was successful")
