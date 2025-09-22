import chainlit as cl
import requests
import os
import sys
import json
import logging
import base64
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from i18n.locale_msg_front import UI_TEXT, EXAMPLE_PROMPTS
from pathlib import Path
from io import BytesIO

# Configuration from environment variables
AUTH_USERNAME = os.getenv("AUTH_USERNAME", "ms_user")
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD", "msuser123")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#@cl.password_auth_callback
def auth_callback(username: str, password: str):
    """Simple password authentication - fixed version"""
    try:
        logger.info(f"🔐 Authentication attempt - username: {username}")
        
        # MS 사용자 인증
        if username == AUTH_USERNAME and password == AUTH_PASSWORD:
            logger.info("✅ MS user authentication successful")
            return cl.User(
                identifier="ms_user",
                metadata={
                    "role": "user",
                    "name": "Microsoft User",
                    "login_time": datetime.now().isoformat()
                }
            )
        
        # 관리자 인증
        elif username == "admin" and password == ADMIN_PASSWORD:
            logger.info("✅ Admin authentication successful")
            return cl.User(
                identifier="admin",
                metadata={
                    "role": "admin", 
                    "name": "Administrator",
                    "login_time": datetime.now().isoformat()
                }
            )
        
        logger.warning(f"❌ Authentication failed for user: {username}")
        return None
        
    except Exception as e:
        logger.error(f"❌ Authentication error: {e}")
        return None

# Load environment variables
SK_API_URL = os.getenv("SK_API_URL", "http://localhost:8000/plan_search")
# Derive upload endpoint from SK_API_URL
UPLOAD_API_URL = os.getenv("UPLOAD_API_URL", SK_API_URL.rsplit("/", 1)[0] + "/upload_documents")
# Status check endpoint
UPLOAD_STATUS_URL = os.getenv("UPLOAD_STATUS_URL", SK_API_URL.rsplit("/", 1)[0] + "/upload_status")

# Global variable to track active uploads
active_uploads = {}  # { upload_id: { files: [...], message: cl.Message, task: asyncio.Task } }


# Define the search engines
SEARCH_ENGINES = {
    "Bing Search": "bing_search_crawling",
    "Grounding Gen": "grounding_bing"
}

# Internationalization constants
SUPPORTED_LANGUAGES = {
    "en-US": "English",
    "ko-KR": "한국어"
}

class ChatSettings:
    """Chat settings for managing user preferences"""
    def __init__(self):
        self.query_rewrite = True
        self.web_search = False
        self.planning = False
        self.ytb_search = False
        self.mcp_server = False
        self.ai_search = True
        self.verbose = True
        self.search_engine = list(SEARCH_ENGINES.values())[0]
        self.language = "ko-KR"
        self.max_tokens = 4000
        self.temperature = 0.7

def get_current_prompt(lang: str, category: str) -> str:
    """Get current prompt text for a category in the specified language"""
    return EXAMPLE_PROMPTS[lang][category]["prompt"]

def get_starter_label(lang: str, category: str) -> str:
    """Get starter label for a category in the specified language"""
    return EXAMPLE_PROMPTS[lang][category]["title"]

def get_starters_for_language(language: str):
    """Get starters for a specific language"""
    starters = []
    
    categories = ["question_Microsoft", "product_info", "recommendation", "comparison", "support_questions", "tools"]
    logger.info(f"Getting starters for language: {language}")
    logger.info(f"Available categories in EXAMPLE_PROMPTS: {list(EXAMPLE_PROMPTS.get(language, {}).keys())}")
    
    for category in categories:
        if category in EXAMPLE_PROMPTS[language]:
            if category == "question_Microsoft":
                emoji="📈" 
                image="/public/images/1f4c8_color.png"
            elif category == "product_info":
                emoji="✅"
                image="/public/images/2705_flat.png"
            elif category == "recommendation":
                emoji="💡"
                image="/public/images/1f4a1_color.png"
            elif category == "comparison":
                emoji="📚"
                image="/public/images/1f4da_color.png"
            elif category == "support_questions":
                emoji="👨‍💻"
                image="/public/images/1f468-1f4bb_flat.png"
            elif category == "tools":
                emoji="🛠"
                image="/public/images/1f6e0_color.png"
                        
            starter = cl.Starter(
                label=get_starter_label(language, category),
                message=get_current_prompt(language, category),
                icon=image
            )
            starters.append(starter)
            logger.info(f"Added starter: {category} - {starter.label}")
    return starters  # ensure starters list is returned

async def check_upload_status_once(upload_id: str) -> dict | None:
    """단발성 업로드 상태 조회 (폴링 루프 내부/액션 버튼에서 호출)"""
    try:
        session = requests.Session()
        resp = session.get(f"{UPLOAD_STATUS_URL}/{upload_id}", timeout=(10, 30))
        if not resp.ok:
            return None
        return resp.json()
    except Exception as e:
        logger.warning(f"[upload:{upload_id}] 상태 조회 실패: {e}")
        return None

async def poll_upload_status_loop(upload_id: str, msg: cl.Message, interval: float = 3.0):
    """주기적으로 상태를 폴링해서 동일 메시지를 갱신"""
    try:
        while True:
            status_data = await check_upload_status_once(upload_id)
            if not status_data:
                msg.content = f"⚠️ 업로드 ID {upload_id[:8]} 상태 조회 실패. 재시도 중..."
                await msg.update()
                await asyncio.sleep(interval)
                continue

            status = status_data.get("status", "unknown")
            message = status_data.get("message", "")
            progress = int(status_data.get("progress", 0))
            file_results = status_data.get("file_results", [])

            if status == "processing":
                green_blocks = progress // 10
                progress_bar = "🟩" * green_blocks + "⬜" * (10 - green_blocks)
                msg.content = (
                    f"📤 **업로드 진행 중** (ID: {upload_id[:8]})\n"
                    f"{message}\n\n"
                    f"진행률: {progress}%\n{progress_bar}"
                )
                await msg.update()
            elif status == "completed":
                success_cnt = len([r for r in file_results if r.get("status") == "success"])
                fail_cnt = len([r for r in file_results if r.get("status") == "error"])
                msg.content = (
                    f"✅ **업로드 완료** (ID: {upload_id[:8]})\n"
                    f"{message}\n\n"
                    f"📄 성공: {success_cnt} / 실패: {fail_cnt}\n"
                    f"💡 이제 문서에 대해 질문해보세요!"
                )
                await msg.update()
                # 예시 질문 자동 전송 (1회)
                entry = active_uploads.get(upload_id, {})
                if not entry.get("examples_sent"):
                    await send_example_questions(upload_id)
                    entry["examples_sent"] = True
                    active_uploads[upload_id] = entry
                break
            elif status == "error":
                msg.content = f"❌ **업로드 실패** (ID: {upload_id[:8]})\n{message}"
                await msg.update()
                break
            else:
                msg.content = f"ℹ️ 알 수 없는 상태 ({status}) - 재시도 중..."
                await msg.update()

            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        logger.info(f"[upload:{upload_id}] 폴링 태스크 취소됨")
    except Exception as e:
        logger.error(f"[upload:{upload_id}] 폴링 중 예외: {e}")
        msg.content += f"\n\n⚠️ 상태 업데이트 중 오류 발생: {e}"
        await msg.update()
    finally:
        # 완료/오류/취소 시 registry 정리
        entry = active_uploads.get(upload_id)
        if entry:
            entry["task"] = None

async def send_example_questions(upload_id: str):
    """업로드 완료 후 문서 기반 예시 질문 1회 자동 전송"""
    entry = active_uploads.get(upload_id)
    if not entry:
        return
    files = entry.get("files", [])
    # 세션에서 언어 가져오기
    settings = cl.user_session.get("settings")
    language = getattr(settings, "language", "ko-KR") if settings else "ko-KR"

    # 파일명 기반 간단한 도메인 추론 (예: 재무/IR 관련)
    lower_names = " ".join(files).lower()
    is_finance = any(k in lower_names for k in ["ir", "earnings", "financial", "재무", "실적", "분기", "annual", "report"])

    # 파일 표시 (최대 3개)
    display_files = files[:3]
    file_line = ""
    if display_files:
        if language.startswith("ko"):
            file_line = "📂 대상 파일: " + ", ".join(display_files)
        else:
            file_line = "📂 Files: " + ", ".join(display_files)

    if language.startswith("ko"):
        header = "💡 **이 문서로 질문 예시**"
        if is_finance:
            examples = [
                "이 보고서의 핵심 재무 지표를 요약해줘",
                "전년 대비 변화율이 큰 항목 3가지를 알려줘",
                "경영진 코멘트(또는 전망) 부분만 뽑아 정리해줘",
                "매출/영업이익/순이익 추이를 표로 만들어줘",
                "위험 요인(Risk factor)이나 경고 신호가 있으면 정리해줘"
            ]
        else:
            examples = [
                "이 문서의 핵심 내용을 5줄로 요약해줘",
                "가장 중요한 인사이트 3가지만 뽑아줘",
                "문서에 등장하는 주요 개념/용어를 설명과 함께 정리해줘",
                "이 문서가 다루는 문제와 제안된 해결책을 정리해줘",
                "추가로 조사하면 좋을 관련 주제 5가지를 제안해줘"
            ]
        follow = "다른 형태의 분석이나 비교가 필요하면 자연어로 자유롭게 질문해주세요."
    else:
        header = "💡 **Example Questions for These Documents**"
        if is_finance:
            examples = [
                "Summarize the key financial indicators from this report.",
                "List top 3 metrics with largest YoY change.",
                "Extract and summarize management outlook or guidance.",
                "Create a table of revenue / operating income / net income trends.",
                "Highlight any risk factors or warning signals mentioned."
            ]
        else:
            examples = [
                "Summarize the core points in 5 concise bullet lines.",
                "List the top 3 most important insights with brief rationale.",
                "Extract key concepts/terms and explain each briefly.",
                "Summarize the problem addressed and proposed solution.",
                "Suggest 5 related follow-up research questions."
            ]
        follow = "Feel free to ask for any other analysis or comparison you need."

    bullets = "\n".join(f"• {q}" for q in examples)
    content = f"{header}\n\n{file_line}\n\n{bullets}\n\n{follow}"
    await cl.Message(content=content).send()

def start_progress_tracker(upload_id: str, files: List[str], base_message: cl.Message):
    """비동기 폴링 태스크 시작 및 registry 저장"""
    if upload_id in active_uploads and active_uploads[upload_id].get("task"):
        logger.info(f"[upload:{upload_id}] 기존 폴링 태스크 재사용")
        return
    task = asyncio.create_task(poll_upload_status_loop(upload_id, base_message))
    active_uploads[upload_id] = {
        "files": files,
        "message": base_message,
        "task": task, 
        "examples_sent": False   # 추가
    }
    logger.info(f"[upload:{upload_id}] 업로드 상태 추적 시작 (files={files})")

async def check_upload_status(upload_id: str, status_message: cl.Message = None):
    """Check upload status and update message"""
    try:
        session = requests.Session()
        response = session.get(f"{UPLOAD_STATUS_URL}/{upload_id}", timeout=30)
        
        if response.ok:
            status_data = response.json()
            status = status_data.get("status", "unknown")
            message = status_data.get("message", "")
            progress = status_data.get("progress", 0)
            
            # Create progress message with emoji
            if status == "processing":
                progress_bar = "🟩" * (progress // 10) + "⬜" * (10 - progress // 10)
                content = f"📤 **업로드 진행 상황**\n\n{message}\n\n진행률: {progress}%\n{progress_bar}"
                
                if status_message:
                    status_message.content = content
                    await status_message.update()
                
                # Continue checking if still processing
                if progress < 100:
                    await asyncio.sleep(2)  # Wait 2 seconds before next check
                    return await check_upload_status(upload_id, status_message)
                    
            elif status == "completed":
                file_results = status_data.get("file_results", [])
                successful_count = len([r for r in file_results if r.get("status") == "success"])
                
                content = f"✅ **업로드 완료!**\n\n{message}\n\n성공한 파일: {successful_count}개\n\n💡 이제 업로드된 문서에 대해 질문해보세요!"
                
                if status_message:
                    status_message.content = content
                    await status_message.update()
                    
            elif status == "error":
                content = f"❌ **업로드 실패**\n\n{message}\n\n다시 시도해보세요."
                
                if status_message:
                    status_message.content = content
                    await status_message.update()
            
            # Remove from active uploads when complete
            if status in ["completed", "error"] and upload_id in active_uploads:
                del active_uploads[upload_id]
                
            return status_data
            
    except Exception as e:
        logger.error(f"Error checking upload status: {e}")
        if status_message:
            status_message.content = f"❌ **상태 확인 실패**: {str(e)}"
            await status_message.update()
    
    return None

async def handle_file_upload(files, settings=None, document_type: str = "IR_REPORT", company: str = None, industry: str = None, report_year: str = None, force_upload: bool = False):
    """Unified file upload handler for all file types"""
    try:
        # Initial upload message
        status_message = cl.Message(content="📤 **파일 업로드 중...**\n\n파일을 서버에 업로드하고 있습니다...")
        await status_message.send()
        
        # Process and validate files
        files_payload, valid_files = [], []
        MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
        allowed_extensions = {'.pdf', '.docx', '.txt'}
        
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(max_retries=3)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        for att in files:
            # Get filename - handle Chainlit file objects properly
            filename = None
            if hasattr(att, 'name'):
                filename = att.name
            elif hasattr(att, 'filename'):
                filename = att.filename
            elif isinstance(att, dict) and 'name' in att:
                filename = att['name']
            else:
                filename = "unknown_file"
            
            logger.info(f"Processing file: {filename}")
            
            # Check file extension
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext not in allowed_extensions:
                await cl.Message(content=f"❌ **지원하지 않는 파일 형식**: {filename}\n\n지원 형식: PDF, DOCX, TXT").send()
                continue
            
            file_bytes = None
            content_type = "application/octet-stream"

            # Get file content - handle Chainlit file objects properly
            if hasattr(att, "content") and att.content:
                file_bytes = att.content
                content_type = getattr(att, "mime", getattr(att, "content_type", content_type))
            elif hasattr(att, "path") and att.path:
                # Read from file path
                try:
                    with open(att.path, "rb") as f:
                        file_bytes = f.read()
                except Exception as e:
                    await cl.Message(content=f"❌ **파일 읽기 실패**: {filename} - {e}").send()
                    continue
            elif isinstance(att, dict) and ("content" in att or "data" in att):
                b64 = att.get("content") or att.get("data")
                try:
                    file_bytes = base64.b64decode(b64)
                except Exception:
                    file_bytes = b""
                content_type = att.get("content_type", content_type)
            elif hasattr(att, "url"):
                url = getattr(att, "url")
                try:
                    r = session.get(url, timeout=30)
                    r.raise_for_status()
                    file_bytes = r.content
                    content_type = r.headers.get("Content-Type", content_type)
                except Exception as e:
                    await cl.Message(content=f"❌ **파일 다운로드 실패**: {filename} - {e}").send()
                    continue
            else:
                logger.warning(f"Cannot process file: {filename} - unsupported format")
                await cl.Message(content=f"❌ **파일 처리 불가**: {filename} - 지원하지 않는 형식").send()
                continue

            # Check if we got file content
            if not file_bytes:
                await cl.Message(content=f"❌ **파일 내용 없음**: {filename}").send()
                continue

            # Check file size
            if len(file_bytes) > MAX_FILE_SIZE:
                await cl.Message(content=f"❌ **파일 크기 초과**: {filename}\n\n최대 크기: 50MB").send()
                continue

            # Add to upload payload
            files_payload.append(("files", (filename, BytesIO(file_bytes), content_type)))
            valid_files.append(filename)
            logger.info(f"Added file to upload: {filename} ({len(file_bytes)} bytes)")

        if not files_payload:
            status_message.content = "❌ **업로드할 유효한 파일이 없습니다.**"
            await status_message.update()
            return False

        # Check file count limit
        if len(files_payload) > 10:
            status_message.content = "❌ **파일 개수 초과**: 최대 10개 파일만 업로드 가능합니다."
            await status_message.update()
            return False

        # Update message with file list
        status_message.content = f"📤 **파일 업로드 중...**\n\n업로드할 파일 ({len(valid_files)}개):\n" + "\n".join([f"• {f}" for f in valid_files])
        await status_message.update()

        # Prepare form data
        data = {
            "document_type": document_type,
            "company": company or "",
            "industry": industry or "",
            "report_year": report_year or "",
            "force_upload": str(force_upload).lower()
        }

        # Upload files
        resp = session.post(UPLOAD_API_URL, files=files_payload, data=data, timeout=120)
        
        if resp.ok:
            try:
                resp_json = resp.json()
                upload_id = resp_json.get("upload_id")
                
                if upload_id:
                    # Start tracking upload status
                    start_progress_tracker(upload_id, valid_files, status_message)
                    return True
                else:
                    message = resp_json.get("message", "업로드 완료")
                    status_message.content = f"✅ **업로드 응답**: {message}"
                    await status_message.update()
                    return True
                    
            except Exception as e:
                status_message.content = f"✅ **업로드 완료**: {resp.text}"
                await status_message.update()
                return True
        else:
            status_message.content = f"❌ **업로드 실패**: {resp.status_code} - {resp.text}"
            await status_message.update()
            return False

    except Exception as e:
        await cl.Message(content=f"❌ **업로드 오류**: {str(e)}").send()
        logger.error(f"Upload error: {e}")
        return False
        error_msg = f"Upload error: {str(e)}"
        await cl.Message(content=error_msg).send()
        logger.error(f"Upload error: {e}")

@cl.set_chat_profiles
async def chat_profile():
    """Set up chat profiles for different languages"""
    return [
        cl.ChatProfile(
            name="Korean",
            markdown_description="## Plan Search Chat",
            icon="/public/images/ai_foundry_icon_small.png",
            starters=get_starters_for_language("ko-KR")
        ),
        cl.ChatProfile(
            name="English", 
            markdown_description="## Plan Search Chat",
            icon="/public/images/ai_foundry_icon_small.png",
            starters=get_starters_for_language("en-US")
        ),
        
    ]

@cl.on_chat_start
async def start():
    """Initialize chat session with user welcome"""
    # Simplified start: rely on global config for file upload (no modal AskFileMessage)
    
    # 사용자 정보 가져오기
    user = cl.user_session.get("user")
    
    # 사용자 환영 메시지
    if user:
        user_role = user.metadata.get("role", "user")
        
        # 관리자 권한이 있는 경우 추가 메시지
        if user_role == "admin":
            await cl.Message(content="🔧 **Admin Access Granted**\nYou have administrator privileges.").send()
    
    # Get current chat profile
    profile = cl.user_session.get("chat_profile", "Korean")
    language = "ko-KR" if profile == "Korean" else "en-US"
    
    # Initialize chat settings
    settings = ChatSettings()
    settings.language = language
    cl.user_session.set("settings", settings)
    
    # Set up chat settings UI
    ui_text = UI_TEXT[language]
    
    # Create settings components
    settings_components = [
        cl.input_widget.Switch(
            id="query_rewrite",
            label=ui_text["query_rewrite_title"],
            initial=True,
            tooltip=ui_text["query_rewrite_desc"]
        ),
        cl.input_widget.Switch(
            id="web_search",
            label=ui_text["web_search_title"],
            initial=False,
            tooltip=ui_text["web_search_desc"]
        ),
        cl.input_widget.Switch(
            id="planning",
            label=ui_text["planning_title"],
            initial=False,
            tooltip=ui_text["planning_desc"]
        ),
        cl.input_widget.Switch(
            id="ytb_search",
            label=ui_text["ytb_search_title"],
            initial=False,
            tooltip=ui_text["ytb_search_desc"]
        ),
        cl.input_widget.Switch(
            id="mcp",
            label=ui_text["mcp_title"],
            initial=False,
            tooltip=ui_text["mcp_desc"]
        ),
        cl.input_widget.Switch(
            id="ai_search",
            label=ui_text["ai_search_title"],
            initial=True,
            tooltip=ui_text["ai_search_desc"]
        ),
        cl.input_widget.Switch(
            id="verbose",
            label=ui_text["verbose_title"],
            initial=True,
            tooltip=ui_text["verbose_desc"]
        ),
        cl.input_widget.Select(
            id="search_engine",
            label=ui_text["search_engine_title"],
            values=list(SEARCH_ENGINES.keys()),
            initial_index=0,
            tooltip=ui_text["search_engine_desc"]
        ),
        cl.input_widget.Switch(
            id="show_starters",
            label="📋 Show Quick Start Options",
            initial=False,
            tooltip="Toggle to show/hide quick start prompts"
        ),
        cl.input_widget.Slider(
            id="max_tokens",
            label="Max Tokens",
            initial=4000,
            min=1000,
            max=8000,
            step=500,
            tooltip="Maximum number of tokens in response"
        ),
        cl.input_widget.Slider(
            id="temperature",
            label="Temperature",
            initial=0.7,
            min=0.0,
            max=1.0,
            step=0.1,
            tooltip="Controls randomness in response generation"
        )

        
    ]
    
    # Send settings to user
    await cl.ChatSettings(settings_components).send()
    
    
    # Set first message flag
    cl.user_session.set("first_message", False)

    # Display file upload information with clear instructions
    welcome_msg = f"""
🎉 **Plan Search Chat에 오신 것을 환영합니다!**

📁 **파일 업로드 기능이 활성화되었습니다**

� **파일 업로드 방법:**
1. 채팅 입력창 위의 **파일 첨부** 버튼을 클릭하세요
2. 업로드할 파일을 선택하세요 (드래그&드롭도 가능)
3. 파일이 자동으로 Knowledge Base에 추가됩니다

✅ **지원 파일 형식:** PDF, DOCX, TXT  
📊 **업로드 제한:** 최대 10개 파일, 각각 50MB 이하  
🔍 **처리 과정:** 업로드된 파일은 AI 검색을 위해 벡터화됩니다

💬 **질문하기:** 파일 업로드 후 관련 질문을 해보세요!
"""
    
    await cl.Message(content=welcome_msg).send()
    

@cl.on_settings_update
async def setup_agent(settings_dict: Dict[str, Any]):
    """Simplified settings update"""
    settings = cl.user_session.get("settings", ChatSettings())
    
    # Update settings with simple mapping
    for key, value in settings_dict.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
    
    # Handle special cases
    if "search_engine" in settings_dict:
        search_engine_name = settings_dict["search_engine"]
        settings.search_engine = SEARCH_ENGINES.get(search_engine_name, list(SEARCH_ENGINES.values())[0])

    # Check if user wants to show starters
    show_starters = settings_dict.get("show_starters", True)
    if show_starters:
        # Re-send starters
        current_profile = cl.user_session.get("chat_profile", "English")
        language = "ko-KR" if current_profile == "Korean" else "en-US"
        starters = get_starters_for_language(language)
        
        # Send starters as a message with action buttons
        starters_message = "📋 **Quick Start Options:**\n\n"
        actions = []
        
        for i, starter in enumerate(starters):
            actions.append(
                cl.Action(
                    name=f"starter_{i}",
                    payload={"message": starter.message, "label": starter.label},
                    label=starter.label,
                    description=f"Use starter: {starter.label}"
                )
            )
        
        await cl.Message(content=starters_message, actions=actions).send()
    cl.user_session.set("settings", settings)
    await cl.Message(content="⚙️ Settings updated successfully!").send()

async def safe_stream_token(msg: cl.Message, content: str) -> bool:
    """Safely stream token with connection check"""
    try:
        await msg.stream_token(content)
        return True
    except Exception as e:
        logger.warning(f"Failed to stream token: {str(e)}")
        return False

async def safe_send_step(step: cl.Step) -> bool:
    """Safely send step with connection check"""
    try:
        await step.send()
        return True
    except Exception as e:
        logger.warning(f"Failed to send step: {str(e)}")
        return False

async def safe_update_message(msg: cl.Message) -> bool:
    """Safely update message with connection check"""
    try:
        await msg.update()
        return True
    except Exception as e:
        logger.warning(f"Failed to update message: {str(e)}")
        return False

def decode_step_content(content: str) -> tuple[str, str, str]:
    """
    Decode step content that may contain code or input data
    Returns: (step_name, code_content, description)
    """
    step_name = content
    code_content = ""
    description = ""
    
    logger.info(f"Decoding step content: {content}")
    
    # Check for code content (Base64 encoded)
    if '#code#' in content:
        parts = content.split('#code#')
        step_name = parts[0]
        if len(parts) > 1:
            try:
                encoded_code = parts[1]
                logger.info(f"Found encoded code: {encoded_code[:50]}...")
                code_content = base64.b64decode(encoded_code).decode('utf-8')
                logger.info(f"Decoded code: {code_content[:100]}...")
            except Exception as e:
                logger.warning(f"Failed to decode code content: {e}")
                code_content = parts[1]  # fallback to raw content
    
    # Check for input description
    if '#input#' in step_name:
        parts = step_name.split('#input#')
        step_name = parts[0]
        if len(parts) > 1:
            description = parts[1].strip()
    
    logger.info(f"Decoded result - step_name: {step_name}, code_length: {len(code_content)}, description: {description}")
    
    return step_name, code_content, description

async def stream_chat_with_api(message: str, settings: ChatSettings) -> None:
    """Stream-enabled chat function that yields partial updates using Chainlit's Step API"""
    if not message or message.strip() == "":
        return
    
    # Get conversation history
    message_history = cl.chat_context.to_openai()
    
    # Helper function to clean text content
    def clean_response_text(text: str) -> str:
        """Clean response text to prevent unwanted markdown formatting"""
        # Replace ~~ with == to avoid strikethrough
        cleaned_text = text.replace("~~", "==")
        # You can add more replacements here if needed
        # cleaned_text = cleaned_text.replace("**", "*")  # Convert bold to italic if needed
        return cleaned_text
    
    # Prepare the API payload
    payload = {
        "messages": message_history[-10:],
        "max_tokens": settings.max_tokens,
        "temperature": settings.temperature,
        "query_rewrite": settings.query_rewrite,
        "planning": settings.planning,
        "include_web_search": settings.web_search,
        "include_ytb_search": settings.ytb_search,
        "include_mcp_server": settings.mcp_server,
        "include_ai_search": settings.ai_search,
        "search_engine": settings.search_engine,
        "stream": True,
        "locale": settings.language,
        "verbose": settings.verbose,
    }
    
    # Debug logging
    logger.info(f"API Payload: query_rewrite={settings.query_rewrite}, web_search={settings.web_search}, planning={settings.planning},"
          f"ytb_search={settings.ytb_search}, mcp_server={settings.mcp_server}, ai_search={settings.ai_search}, search_engine={settings.search_engine}, "
          f"max_tokens={settings.max_tokens}, temperature={settings.temperature}, "
          f"language={settings.language}, verbose={settings.verbose}")
    
    # Create message for streaming response
    ui_text = UI_TEXT[settings.language]
    msg = cl.Message(content="")
    await msg.send()
    
    try:
        # Set up session with retry capability
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(max_retries=3)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        api_url = SK_API_URL
        
        # Create step for API call with detailed information
        async with cl.Step(name="API Request", type="run") as step:
            step.input = {
                "endpoint": api_url,
                "query_rewrite": settings.query_rewrite,
                "planning": settings.planning,
                "web_search": settings.web_search,
                "ytb_search": settings.ytb_search,
                "mcp_server": settings.mcp_server,
                "ai_search": settings.ai_search,
                "search_engine": settings.search_engine,
                "verbose": settings.verbose,
                "locale": settings.language,
            }
            
            # Make request with stream=True
            response = session.post(
                api_url,
                json=payload,
                timeout=(5, 120),
                stream=True,
                headers={"Accept": "text/event-stream"}
            )
            
            step.output = f"Response status: {response.status_code}"
            
            logger.info(f"Response status: {response.status_code}, Content-Type: {response.headers.get('Content-Type', 'unknown')}")
            
            if response.status_code == 200:
                content_type = response.headers.get('Content-Type', '')
                
                if 'text/event-stream' in content_type:
                    # Process Server-Sent Events (SSE) with tool calling steps
                    async with cl.Step(name="Processing Response", type="tool") as process_step:
                        process_step.input = "Processing streaming response..."
                        
                        accumulated_content = ""
                        current_tool_step = None
                        tool_steps = {}
                        
                        logger.info("Starting SSE processing loop...")
                        for line in response.iter_lines():
                            if not line:
                                continue
                            
                            # Decode the line
                            line = line.decode('utf-8')
                            logger.info(f"SSE line received: {line}")
                            
                            # Skip SSE comments and empty lines
                            if line.startswith(':') or not line.strip():
                                continue
                            
                            # Handle SSE format (data: prefix)
                            if line.startswith('data: '):
                                line = line[6:].strip()  # Remove the 'data: ' prefix
                                
                                # Status message handling - create tool steps for different operations
                                if line.startswith('### '):
                                    step_content = line[4:]
                                    
                                    # Complete previous step if exists
                                    if current_tool_step:
                                        current_tool_step.output = "✅ Completed"
                                        await safe_send_step(current_tool_step)
                                    
                                    # Decode step content (name, code, description)
                                    step_name, code_content, description = decode_step_content(step_content)
                                    
                                    # Create new step for each tool operation with appropriate types
                                    step_type = "tool"
                                    step_icon = "🔧"
                                    
                                    # Determine step type and icon based on step name
                                    step_name_lower = step_name.lower()
                                    try:
                                        if ui_text.get("analyzing", "").lower() in step_name_lower:
                                            step_type = "intent"
                                            step_icon = "🧠"
                                        elif ui_text.get("analyze_complete", "").lower() in step_name_lower:
                                            step_type = "intent"
                                            step_icon = "🧠"
                                        elif ui_text.get("search_planning", "").lower() in step_name_lower:
                                            step_type = "planning"
                                            step_icon = "📋"
                                        elif ui_text.get("plan_done", "").lower() in step_name_lower:
                                            step_type = "planning"
                                            step_icon = "📋"
                                        elif ui_text.get("searching", "").lower() in step_name_lower:
                                            step_type = "retrieval"
                                            step_icon = "🌐"
                                        elif ui_text.get("search_done", "").lower() in step_name_lower:
                                            step_type = "retrieval"
                                            step_icon = "🌐"                                            
                                        elif ui_text.get("searching_YouTube", "").lower() in step_name_lower:
                                            step_type = "retrieval"
                                            step_icon = "🎬"
                                        elif ui_text.get("YouTube_done", "").lower() in step_name_lower:
                                            step_type = "retrieval"
                                            step_icon = "🎬"                                            
                                        elif ui_text.get("answering", "").lower() in step_name_lower:
                                            step_type = "llm"
                                            step_icon = "✏️"
                                        elif ui_text.get("search_and_answer", "").lower() in step_name_lower:
                                            step_type = "llm"
                                            step_icon = "✏️"
                                        elif "context information" in step_name_lower:
                                            step_type = "tool"
                                            step_icon = "📃"
                                    except KeyError as e:
                                        logger.warning(f"Missing UI text key: {e}")
                                    
                                    current_tool_step = cl.Step(
                                        name=f"{step_icon} {step_name}", 
                                        type=step_type
                                    )
                                    
                                    # Set input based on available content
                                    if code_content:
                                        # Display code with syntax highlighting
                                        current_tool_step.input = f"```python\n{code_content}\n```"
                                    elif description:
                                        # Display description
                                        current_tool_step.input = description
                                    else:
                                        # Default message
                                        current_tool_step.input = f"Executing: {step_name}"
                                    
                                    if not await safe_send_step(current_tool_step):
                                        logger.warning(f"Failed to send tool step: {step_name}")
                                        break  # Exit if connection is lost
                                    
                                    # Store step for later reference
                                    tool_steps[step_name] = current_tool_step
                            else:
                                # Regular content - clean and accumulate and stream
                                cleaned_line = clean_response_text(line)  # Clean the line before processing
                                
                                if accumulated_content:
                                    # Apply formatting rules for line breaks
                                    if cleaned_line.startswith(('•', '-', '#', '1.', '2.', '3.')) or accumulated_content.endswith(('.', '!', '?', ':')):
                                        accumulated_content += "\n\n" + cleaned_line
                                    else:
                                        accumulated_content += "\n" + cleaned_line
                                else:
                                    accumulated_content = cleaned_line
                                
                                # Stream update to UI safely with cleaned content
                                if not await safe_stream_token(msg, cleaned_line + "\n"):
                                    logger.warning("Stream connection lost, stopping streaming")
                                    break  # Exit if connection is lost
                        
                        # Close any remaining tool step
                        if current_tool_step:
                            current_tool_step.output = "✅ Completed"
                            await safe_send_step(current_tool_step)
                        
                        process_step.output = f"✅ Processed {len(accumulated_content)} characters across {len(tool_steps)} tool steps"
                
                else:
                    # Handle regular non-streaming response
                    async with cl.Step(name="Processing Non-Streaming Response", type="tool") as process_step:
                        logger.info("Not a chunked response, trying to process as regular response")
                        try:
                            chunks = []
                            for chunk in response.iter_content(chunk_size=None):
                                if chunk:
                                    chunks.append(chunk)
                            
                            if chunks:
                                response_text = b''.join(chunks).decode('utf-8', errors='replace')
                                cleaned_response = clean_response_text(response_text) # Clean the response
                                
                                # Try to parse as JSON first
                                try:
                                    response_data = json.loads(response_text)
                                    if isinstance(response_data, dict) and "content" in response_data:
                                        cleaned_content = clean_response_text(response_data["content"])
                                        await safe_stream_token(msg, cleaned_content)
                                        process_step.output = f"✅ Parsed JSON response with content: {cleaned_content[:50]}..."
                                    else:
                                        await safe_stream_token(msg, cleaned_response)
                                        process_step.output = "✅ JSON response without content field, using raw text"
                                except json.JSONDecodeError:
                                    # Not valid JSON, just use as text
                                    await safe_stream_token(msg, cleaned_response)
                                    process_step.output = "✅ Not a valid JSON response, using raw text"
                            else:
                                error_msg = "No response received from server."
                                await safe_stream_token(msg, error_msg)
                                process_step.output = error_msg
                        
                        except Exception as e:
                            error_msg = f"Error processing response: {str(e)}"
                            await safe_stream_token(msg, error_msg)
                            process_step.output = error_msg
            else:
                error_msg = f"Error: {response.status_code} - {response.text}"
                await safe_stream_token(msg, error_msg)
                step.output = error_msg
    
    except requests.exceptions.Timeout:
        error_msg = "Error: Request timed out. The server took too long to respond."
        await safe_stream_token(msg, error_msg)
        logger.error("Request timed out")
    except requests.exceptions.ConnectionError:
        error_msg = "Error: Connection failed. Please check if the API server is running."
        await safe_stream_token(msg, error_msg)
        logger.error("Connection error")
    except requests.exceptions.ChunkedEncodingError:
        error_msg = "Error: Connection interrupted while receiving data from the server."
        await safe_stream_token(msg, error_msg)
        logger.error("Chunked encoding error - connection interrupted")
    except requests.exceptions.RequestException as e:
        error_msg = f"Error connecting to the API: {str(e)}"
        await safe_stream_token(msg, error_msg)
        logger.error(f"Request exception: {type(e).__name__}: {str(e)}")
    except json.JSONDecodeError as e:
        error_msg = "Error: Received invalid JSON from the server."
        await safe_stream_token(msg, error_msg)
        logger.error(f"JSON decode error: {e}")
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        await safe_stream_token(msg, error_msg)
        logger.error(f"Unexpected error in stream_chat_with_api: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Finalize the message safely
    await safe_update_message(msg)
    logger.info("Streaming completed")


def clean_response_text(text: str) -> str:
    """Clean response text to prevent unwanted markdown formatting"""
    return text.replace("~~", "==")

def create_api_payload(settings: ChatSettings) -> dict:
    """Create API payload from settings"""
    message_history = cl.chat_context.to_openai()
    return {
        "messages": message_history[-10:],
        "max_tokens": settings.max_tokens,
        "temperature": settings.temperature,
        "query_rewrite": settings.query_rewrite,
        "planning": settings.planning,
        "include_web_search": settings.web_search,
        "include_ytb_search": settings.ytb_search,
        "include_mcp_server": settings.mcp_server,
        "include_ai_search": settings.ai_search,
        "search_engine": settings.search_engine,
        "stream": True,
        "locale": settings.language,
        "verbose": settings.verbose,
    }

async def handle_error_response(msg: cl.Message, error_type: str, error_msg: str):
    """Handle different types of errors uniformly"""
    full_msg = f"❌ **{error_type}**: {error_msg}"
    await safe_stream_token(msg, full_msg)
    logger.error(f"{error_type}: {error_msg}")

async def stream_chat_with_api(message: str, settings: ChatSettings) -> None:
    """Stream-enabled chat function that yields partial updates using Chainlit's Step API"""
    if not message or message.strip() == "":
        return
    
    # Get conversation history
    message_history = cl.chat_context.to_openai()
    
    # Helper function to clean text content
    def clean_response_text(text: str) -> str:
        """Clean response text to prevent unwanted markdown formatting"""
        # Replace ~~ with == to avoid strikethrough
        cleaned_text = text.replace("~~", "==")
        # You can add more replacements here if needed
        # cleaned_text = cleaned_text.replace("**", "*")  # Convert bold to italic if needed
        return cleaned_text
    
    # Prepare the API payload
    payload = {
        "messages": message_history[-10:],
        "max_tokens": settings.max_tokens,
        "temperature": settings.temperature,
        "query_rewrite": settings.query_rewrite,
        "planning": settings.planning,
        "include_web_search": settings.web_search,
        "include_ytb_search": settings.ytb_search,
        "include_mcp_server": settings.mcp_server,
        "include_ai_search": settings.ai_search,
        "search_engine": settings.search_engine,
        "stream": True,
        "locale": settings.language,
        "verbose": settings.verbose,
    }
    
    # Debug logging
    logger.info(f"API Payload: query_rewrite={settings.query_rewrite}, web_search={settings.web_search}, planning={settings.planning},"
          f"ytb_search={settings.ytb_search}, mcp_server={settings.mcp_server}, ai_search={settings.ai_search}, search_engine={settings.search_engine}, "
          f"max_tokens={settings.max_tokens}, temperature={settings.temperature}, "
          f"language={settings.language}, verbose={settings.verbose}")
    
    # Create message for streaming response
    ui_text = UI_TEXT[settings.language]
    msg = cl.Message(content="")
    await msg.send()
    
    try:
        # Set up session with retry capability
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(max_retries=3)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        api_url = SK_API_URL
        
        # Create step for API call with detailed information
        async with cl.Step(name="API Request", type="run") as step:
            step.input = {
                "endpoint": api_url,
                "query_rewrite": settings.query_rewrite,
                "planning": settings.planning,
                "web_search": settings.web_search,
                "ytb_search": settings.ytb_search,
                "mcp_server": settings.mcp_server,
                "ai_search": settings.ai_search,
                "search_engine": settings.search_engine,
                "verbose": settings.verbose,
                "locale": settings.language,
            }
            
            # Make request with stream=True
            response = session.post(
                api_url,
                json=payload,
                timeout=(5, 120),
                stream=True,
                headers={"Accept": "text/event-stream"}
            )
            
            step.output = f"Response status: {response.status_code}"
            
            logger.info(f"Response status: {response.status_code}, Content-Type: {response.headers.get('Content-Type', 'unknown')}")
            
            if response.status_code == 200:
                content_type = response.headers.get('Content-Type', '')
                
                if 'text/event-stream' in content_type:
                    # Process Server-Sent Events (SSE) with tool calling steps
                    async with cl.Step(name="Processing Response", type="tool") as process_step:
                        process_step.input = "Processing streaming response..."
                        
                        accumulated_content = ""
                        current_tool_step = None
                        tool_steps = {}
                        
                        logger.info("Starting SSE processing loop...")
                        for line in response.iter_lines():
                            if not line:
                                continue
                            
                            # Decode the line
                            line = line.decode('utf-8')
                            logger.info(f"SSE line received: {line}")
                            
                            # Skip SSE comments and empty lines
                            if line.startswith(':') or not line.strip():
                                continue
                            
                            # Handle SSE format (data: prefix)
                            if line.startswith('data: '):
                                line = line[6:].strip()  # Remove the 'data: ' prefix
                                
                                # Status message handling - create tool steps for different operations
                                if line.startswith('### '):
                                    step_content = line[4:]
                                    
                                    # Complete previous step if exists
                                    if current_tool_step:
                                        current_tool_step.output = "✅ Completed"
                                        await safe_send_step(current_tool_step)
                                    
                                    # Decode step content (name, code, description)
                                    step_name, code_content, description = decode_step_content(step_content)
                                    
                                    # Create new step for each tool operation with appropriate types
                                    step_type = "tool"
                                    step_icon = "🔧"
                                    
                                    # Determine step type and icon based on step name
                                    step_name_lower = step_name.lower()
                                    try:
                                        if ui_text.get("analyzing", "").lower() in step_name_lower:
                                            step_type = "intent"
                                            step_icon = "🧠"
                                        elif ui_text.get("analyze_complete", "").lower() in step_name_lower:
                                            step_type = "intent"
                                            step_icon = "🧠"
                                        elif ui_text.get("search_planning", "").lower() in step_name_lower:
                                            step_type = "planning"
                                            step_icon = "📋"
                                        elif ui_text.get("plan_done", "").lower() in step_name_lower:
                                            step_type = "planning"
                                            step_icon = "📋"
                                        elif ui_text.get("searching", "").lower() in step_name_lower:
                                            step_type = "retrieval"
                                            step_icon = "🌐"
                                        elif ui_text.get("search_done", "").lower() in step_name_lower:
                                            step_type = "retrieval"
                                            step_icon = "🌐"                                            
                                        elif ui_text.get("searching_YouTube", "").lower() in step_name_lower:
                                            step_type = "retrieval"
                                            step_icon = "🎬"
                                        elif ui_text.get("YouTube_done", "").lower() in step_name_lower:
                                            step_type = "retrieval"
                                            step_icon = "🎬"                                            
                                        elif ui_text.get("answering", "").lower() in step_name_lower:
                                            step_type = "llm"
                                            step_icon = "✏️"
                                        elif ui_text.get("search_and_answer", "").lower() in step_name_lower:
                                            step_type = "llm"
                                            step_icon = "✏️"
                                        elif "context information" in step_name_lower:
                                            step_type = "tool"
                                            step_icon = "📃"
                                    except KeyError as e:
                                        logger.warning(f"Missing UI text key: {e}")
                                    
                                    current_tool_step = cl.Step(
                                        name=f"{step_icon} {step_name}", 
                                        type=step_type
                                    )
                                    
                                    # Set input based on available content
                                    if code_content:
                                        # Display code with syntax highlighting
                                        current_tool_step.input = f"```python\n{code_content}\n```"
                                    elif description:
                                        # Display description
                                        current_tool_step.input = description
                                    else:
                                        # Default message
                                        current_tool_step.input = f"Executing: {step_name}"
                                    
                                    if not await safe_send_step(current_tool_step):
                                        logger.warning(f"Failed to send tool step: {step_name}")
                                        break  # Exit if connection is lost
                                    
                                    # Store step for later reference
                                    tool_steps[step_name] = current_tool_step
                            else:
                                # Regular content - clean and accumulate and stream
                                cleaned_line = clean_response_text(line)  # Clean the line before processing
                                
                                if accumulated_content:
                                    # Apply formatting rules for line breaks
                                    if cleaned_line.startswith(('•', '-', '#', '1.', '2.', '3.')) or accumulated_content.endswith(('.', '!', '?', ':')):
                                        accumulated_content += "\n\n" + cleaned_line
                                    else:
                                        accumulated_content += "\n" + cleaned_line
                                else:
                                    accumulated_content = cleaned_line
                                
                                # Stream update to UI safely with cleaned content
                                if not await safe_stream_token(msg, cleaned_line + "\n"):
                                    logger.warning("Stream connection lost, stopping streaming")
                                    break  # Exit if connection is lost
                        
                        # Close any remaining tool step
                        if current_tool_step:
                            current_tool_step.output = "✅ Completed"
                            await safe_send_step(current_tool_step)
                        
                        process_step.output = f"✅ Processed {len(accumulated_content)} characters across {len(tool_steps)} tool steps"
                
                else:
                    # Handle regular non-streaming response
                    async with cl.Step(name="Processing Non-Streaming Response", type="tool") as process_step:
                        logger.info("Not a chunked response, trying to process as regular response")
                        try:
                            chunks = []
                            for chunk in response.iter_content(chunk_size=None):
                                if chunk:
                                    chunks.append(chunk)
                            
                            if chunks:
                                response_text = b''.join(chunks).decode('utf-8', errors='replace')
                                cleaned_response = clean_response_text(response_text) # Clean the response
                                
                                # Try to parse as JSON first
                                try:
                                    response_data = json.loads(response_text)
                                    if isinstance(response_data, dict) and "content" in response_data:
                                        cleaned_content = clean_response_text(response_data["content"])
                                        await safe_stream_token(msg, cleaned_content)
                                        process_step.output = f"✅ Parsed JSON response with content: {cleaned_content[:50]}..."
                                    else:
                                        await safe_stream_token(msg, cleaned_response)
                                        process_step.output = "✅ JSON response without content field, using raw text"
                                except json.JSONDecodeError:
                                    # Not valid JSON, just use as text
                                    await safe_stream_token(msg, cleaned_response)
                                    process_step.output = "✅ Not a valid JSON response, using raw text"
                            else:
                                error_msg = "No response received from server."
                                await safe_stream_token(msg, error_msg)
                                process_step.output = error_msg
                        
                        except Exception as e:
                            error_msg = f"Error processing response: {str(e)}"
                            await safe_stream_token(msg, error_msg)
                            process_step.output = error_msg
            else:
                error_msg = f"Error: {response.status_code} - {response.text}"
                await safe_stream_token(msg, error_msg)
                step.output = error_msg
    
    except requests.exceptions.Timeout:
        error_msg = "Error: Request timed out. The server took too long to respond."
        await safe_stream_token(msg, error_msg)
        logger.error("Request timed out")
    except requests.exceptions.ConnectionError:
        error_msg = "Error: Connection failed. Please check if the API server is running."
        await safe_stream_token(msg, error_msg)
        logger.error("Connection error")
    except requests.exceptions.ChunkedEncodingError:
        error_msg = "Error: Connection interrupted while receiving data from the server."
        await safe_stream_token(msg, error_msg)
        logger.error("Chunked encoding error - connection interrupted")
    except requests.exceptions.RequestException as e:
        error_msg = f"Error connecting to the API: {str(e)}"
        await safe_stream_token(msg, error_msg)
        logger.error(f"Request exception: {type(e).__name__}: {str(e)}")
    except json.JSONDecodeError as e:
        error_msg = "Error: Received invalid JSON from the server."
        await safe_stream_token(msg, error_msg)
        logger.error(f"JSON decode error: {e}")
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        await safe_stream_token(msg, error_msg)
        logger.error(f"Unexpected error in stream_chat_with_api: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Finalize the message safely
    await safe_update_message(msg)
    logger.info("Streaming completed")

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages"""
    settings = cl.user_session.get("settings")
    if not settings:
        settings = ChatSettings()
        cl.user_session.set("settings", settings)
    
    # Check for file attachments (try multiple possible attributes)
    # Collect attachments uniformly
    attachments = (getattr(message, "elements", None) or 
                  getattr(message, "files", None) or 
                  getattr(message, "attachments", None))
    
    uploaded = None
   
    if attachments:
        uploaded = await handle_file_upload(attachments, settings)
        # If only files (no textual content), stop here
        if (not message.content) or (message.content.strip() == ""):
            return
    
    message_content = message.content
    
    # Provide light feedback if user sent text along with freshly uploaded files
    if uploaded and message_content:
        await cl.Message(content="📎 첨부한 파일을 처리한 후 답변을 생성합니다...").send()
        await stream_chat_with_api(message.content, settings)

    # Process the message with streaming
    await stream_chat_with_api(message.content, settings)

@cl.action_callback("clear_chat")
async def on_action(action: cl.Action):
    """Handle clear chat action"""
    # Clear the chat context
    cl.chat_context.clear()
    
    # Send confirmation
    await cl.Message(content="Chat history cleared!").send()
    
    # Return success
    return "Chat cleared successfully"


@cl.action_callback("help_action")
async def on_help_action(action: cl.Action):
    """Handle help action"""
    help_message = """
📖 **도움말**

🔹 **파일 업로드 방법:**
1️⃣ 채팅 입력창 하단의 📎 버튼 클릭 (있는 경우)
2️⃣ "파일업로드" 명령어 입력
3️⃣ 위의 "📎 파일 업로드" 버튼 클릭

🔹 **지원 파일 형식:** PDF, DOCX, TXT
🔹 **업로드 제한:** 최대 10개 파일, 각각 50MB 이하

🔹 **사용법:**
- 파일 업로드 후 관련 질문을 입력하세요
- 예: "이 문서의 주요 내용을 요약해주세요"

❓ **문제 해결:**
- 📎 버튼이 보이지 않으면 브라우저를 새로고침하거나 "파일업로드" 명령어를 사용하세요
"""
    
    await cl.Message(content=help_message).send()
    return "Help displayed"

@cl.action_callback("show_starters_action")
async def on_show_starters_action(action: cl.Action):
    """Handle show starters action"""
    current_profile = cl.user_session.get("chat_profile", "Korean")
    language = "ko-KR" if current_profile == "Korean" else "en-US"
    starters = get_starters_for_language(language)
    
    # Send starters as a message with action buttons
    starters_message = "📋 **Quick Start Options:**\n\n"
    actions = []
    
    for i, starter in enumerate(starters):
        # Get emoji from category mapping
        if i == 0:  # question_Microsoft
            emoji = "📈"
        elif i == 1:  # product_info
            emoji = "✅"
        elif i == 2:  # recommendation
            emoji = "💡"
        elif i == 3:  # comparison
            emoji = "📚"
        elif i == 4:  # support_questions
            emoji = "👨‍💻"
        elif i == 5:  # tools
            emoji = "🛠️"
        else:
            emoji = "🤖"
            
        starters_message += f"{emoji} **{starter.label}**\n"
        actions.append(
            cl.Action(
                name=f"starter_{i}",
                payload={"message": starter.message, "label": starter.label},
                label=f"{emoji} {starter.label}",
                description=f"Use starter: {starter.label}"
            )
        )
    
    await cl.Message(content=starters_message, actions=actions).send()
    return "Starters displayed"

@cl.action_callback("starter_0")
@cl.action_callback("starter_1")
@cl.action_callback("starter_2")
@cl.action_callback("starter_3")
@cl.action_callback("starter_4")
@cl.action_callback("starter_5")
async def on_starter_action(action: cl.Action):
    """Handle starter action clicks"""
    # Extract message from payload dictionary
    message_content = action.payload.get("message", "")
    starter_label = action.payload.get("label", "Unknown")
    
    logger.info(f"🎯 Starter action triggered: {action.name}")
    logger.info(f"📝 Message content: {message_content[:100]}...")
    logger.info(f"🏷️ Starter label: {starter_label}")
    
    # First, add the user message to chat history
    user_message = cl.Message(
        author="User",
        content=message_content,
        type="user_message"
    )
    await user_message.send()
    
    # Get current settings
    settings = cl.user_session.get("settings")
    if not settings:
        settings = ChatSettings()
        cl.user_session.set("settings", settings)
    
    # Process the starter message
    await stream_chat_with_api(message_content, settings)
    
    return f"Processing starter: {starter_label}"

@cl.action_callback("check_upload_status")
async def on_check_upload_status(action: cl.Action):
    """모든 활성 업로드의 최신 상태 요약 출력"""
    if not active_uploads:
        await cl.Message(content="📋 **현재 진행 중인 업로드가 없습니다.**").send()
        return "No active uploads"
    lines = ["📊 **현재 진행 중인 업로드 목록**\n"]
    for upload_id, info in active_uploads.items():
        state_line = ""
        # 메시지 객체의 최신 content 일부 활용
        msg_obj = info.get("message")
        preview = ""
        if msg_obj and getattr(msg_obj, "content", None):
            preview = msg_obj.content.splitlines()[0][:60]
        lines.append(f"• {upload_id[:8]} ({', '.join(info['files'])})")
        if preview:
            lines.append(f"  ↳ {preview}")
    await cl.Message(content="\n".join(lines)).send()
    return "Listed active uploads"
    
if __name__ == "__main__":
    cl.run()
