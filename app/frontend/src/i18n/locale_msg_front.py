# locale_msg_front.py

UI_TEXT = {
    "en-US": {
        "title": "# Microsoft Plan and Search Chat",
        "select_agent_mode": "### Select Agentic Mode",
        "query_rewrite_title": "#### Query Rewrite",
        "query_rewrite_desc": "GPT rewrites and responds with detailed questions about Microsoft.",
        "web_search_title": "#### Web Search",
        "web_search_desc": "GPT includes asynchronous web search results.",
        "planning_title": "#### Plan & Execute",
        "planning_desc": "GPT plans and executes for complex queries.",
        "ytb_search_title": "#### YouTube Search",
        "ytb_search_desc": "GPT includes asynchronous YouTube search results.",
        "mcp_title": "#### enable MCP Server",
        "mcp_desc": "MCP Server is used for YouTube searches.",
        "ai_search_title": "#### enable AI Search",
        "ai_search_desc": "AI Search is used for processing documents.",
        "verbose_title": "#### Verbose Mode", 
        "verbose_desc": "GPT will provide more detailed responses",
        "search_engine_title": "#### Search Engine",
        "search_engine_desc": "Search engines use crawling except for Grounding Gen",
        "enable_label": "Enable",
        "send_button": "Send",
        "clear_chat_button": "Clear Chat",
        "try_prompts": "### Try following prompts",
        "connecting_api": "⟳ Connecting to API...",
        "searching_response": "⟳ Searching response...",
        "processing_message": "Processing message...",
        "language_toggle": "Language / 언어",
        "analyzing": "Analyzing question...",
        "analyze_complete": "Question analysis complete",
        "plan_done": "Search plan complete.",
        "searching": "Searching Web...",
        "search_done": "Web Search complete.",
        "search_planning": "Search Planning...",
        "searching_YouTube": "Searching YouTube...",
        "YouTube_done": "YouTube Search complete.",
        "answering": "Generating answer...",
        "search_and_answer": "Searching and generating answer...",
    },
    "ko-KR": {
        "title": "# Microsoft Plan and Search Chat",
        "select_agent_mode": "### 에이전트 모드 선택",
        "query_rewrite_title": "#### 쿼리 재작성",
        "query_rewrite_desc": "GPT가 마이크로소프트에 관한 디테일한 질문으로 재작성하여 응답합니다",
        "web_search_title": "#### 웹 검색",
        "web_search_desc": "GPT가 비동기 웹 검색 결과를 포함합니다",
        "planning_title": "#### 검색 계획",
        "planning_desc": "복잡한 쿼리에 대해 GPT가 계획하고 실행합니다",
        "ytb_search_title": "#### YouTube 검색",
        "ytb_search_desc": "비동기 YouTube 검색 결과를 포함합니다",
        "mcp_title": "#### MCP서버 사용여부 ",
        "mcp_desc": "YouTube 검색 시 MCP서버를 사용합니다",
        "ai_search_title": "#### AI Search 사용여부",
        "ai_search_desc": "문서 처리를 위해 AI Search를 사용합니다",
        "verbose_title": "#### 상세 모드",
        "verbose_desc": "GPT가 더 자세한 응답을 제공합니다",
        "search_engine_title": "#### 검색 엔진",
        "search_engine_desc": "Grounding Gen을 제외한 검색 엔진은 크롤링을 사용합니다",
        "enable_label": "Enable",
        "send_button": "전송",
        "clear_chat_button": "채팅 지우기",
        "try_prompts": "### 다음 프롬프트를 시도해보세요",
        "connecting_api": "⟳ API 연결 중...",
        "searching_response": "⟳ 응답 검색 중...",
        "processing_message": "메시지 처리 중...",
        "language_toggle": "Language / 언어",
        "analyzing": "질문 분석 중...",
        "analyze_complete": "질문 분석 완료",
        "plan_done": "검색 계획 수립 완료.",
        "searching": "웹 검색 중...",
        "search_done": "웹 검색 완료.",
        "search_planning": "검색 계획 수립 중...",
        "searching_YouTube": "YouTube에서 검색 중...",
        "YouTube_done": "YouTube 검색 완료.",
        "answering": "답변 생성 중...",
        "search_and_answer": "웹 검색 및 답변 생성 중...",
    }
}

EXAMPLE_PROMPTS = {
    "en-US": {
        "question_Microsoft": {
            "title": "Microsoft General Questions",
            "description": "Have a general question about Microsoft",
            "prompt": "How is the Microsoft stock mood these days?"
        },
        "product_info": {
            "title": "Questions about Product", 
            "description": "Ask questions about various Microsoft products",
            "prompt": "Can you recommend a Surface model other than 12inches from Microsoft?"
        },
        "recommendation": {
            "title": "Recommendation",
            "description": "Ask recommendation for a product line", 
            "prompt": "Can you recommend a good office software for my parents?"
        },
        "comparison": {
            "title": "Comparison",
            "description": "Ask for a comparison between two different models",
            "prompt": "Can you compare Samsung Galaxy Book and Microsoft Surface Pro?"
        },
        "support_questions": {
            "title": "Support Questions",
            "description": "Ask for support-related inquiries",
            "prompt": "What is the warranty period for Microsoft products?"
        },
        "tools": {
            "title": "Tool (Time & Weather)",
            "description": "Ask time and weather information from external APIs",
            "prompt": "What time is it in New York?"
        }
    },
    "ko-KR": {
        "question_Microsoft": {
            "title": "마이크로소프트 일반 질문",
            "description": "마이크로소프트에 대한 일반적인 질문하기",
            "prompt": "마이크로소프트 주식 분위기가 요즘 어때?"
        },
        "product_info": {
            "title": "제품 질문",
            "description": "다양한 마이크로소프트 제품에 대한 질문하기",
            "prompt": "마이크로소프트 제품 중 12인치가 아닌 서피스 모델 추천해줘"
        },
        "recommendation": {
            "title": "추천",
            "description": "제품 라인에 대한 추천 요청하기",
            "prompt": "부모님께 좋은 오피스 소프트웨어를 추천해줄 수 있어?"
        },
        "comparison": {
            "title": "비교",
            "description": "두 가지 다른 모델 간의 비교 요청하기",
            "prompt": "삼성노트북과 마이크로소프트 서피스를 비교해줄 수 있어?"
        },
        "support_questions": {
            "title": "지원 질문",
            "description": "지원 관련 문의하기",
            "prompt": "마이크로소프트 제품의 보증 기간은 어떻게 되나요?"
        },
        "tools": {
            "title": "도구 (시간 및 날씨)",
            "description": "외부 API에서 시간 및 날씨 정보 요청하기",
            "prompt": "뉴욕은 지금 몇 시야?"
        }
    }
}