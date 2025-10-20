"""Group Chatting Executor using Microsoft Agent Framework.

Implements a writer/reviewer loop per sub-topic using Agent Framework patterns:
- ChatAgent for Writer and Reviewer roles
- Custom termination strategy (approval-based)
- Sequential turn-based conversation

This executor provides the same interface as the Semantic Kernel version
but uses Agent Framework's native group chat capabilities.

References:
- Multi-Agent Collaboration notebook (01.1_multi-agent-collabration.ipynb)
- Agent Framework Documentation: https://learn.microsoft.com/en-us/agent-framework/
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

from agent_framework import ChatAgent, ChatMessage, TextContent, Role, Executor, WorkflowContext, handler
from agent_framework.azure import AzureOpenAIChatClient
from langchain.prompts import load_prompt
from config.config import Settings
from i18n.locale_msg import LOCALE_MESSAGES
from utils.yield_message import send_step_with_code, send_step_with_input, send_step_with_code_and_input
from typing_extensions import Never



logger = logging.getLogger(__name__)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load prompt templates (reuse existing SK prompts)
RESEARCH_WRITER_PROMPT = load_prompt(
    os.path.join(current_dir, "..", "prompts", "research_writer_prompt.yaml"),
    encoding="utf-8",
)
RESEARCH_REVIEWER_PROMPT = load_prompt(
    os.path.join(current_dir, "..", "prompts", "research_reviewer_prompt.yaml"),
    encoding="utf-8",
)


class ApprovalTerminationStrategy:
    """Termination strategy that stops when approval is given."""
    
    def __init__(self, max_iterations: int = 10):
        self.max_iterations = max_iterations
        self.iteration_count = 0
    
    def should_terminate(self, messages: List[ChatMessage]) -> bool:
        """Check if the conversation should terminate."""
        self.iteration_count += 1
        
        # Terminate if max iterations reached
        if self.iteration_count >= self.max_iterations:
            logger.info(f"Reached maximum iterations ({self.max_iterations})")
            return True
        
        # Terminate if last message contains approval or final answer
        if messages and len(messages) > 0:
            last_message = messages[-1]
            if last_message.text:
                text_lower = last_message.text.lower()
                # Check for approval keywords
                if any(keyword in text_lower for keyword in ["approved", "approval granted", "final answer"]):
                    logger.info(f"Approval detected in iteration {self.iteration_count}")
                    return True
        
        return False
    
    def reset(self):
        """Reset the iteration counter."""
        self.iteration_count = 0


class GroupChattingExecutor:
    """Agent Framework-based group chatting executor for writer/reviewer collaboration."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        
        # Create Azure OpenAI chat client
        self.chat_client = AzureOpenAIChatClient(
            model_id=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
            endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
        )
        
        logger.info("[GroupChatExecutor] Initialized with Azure OpenAI")
    
    async def group_chat(
        self,
        sub_topic: str,
        question: str,
        sub_topic_contexts: str,
        locale: str = "ko-KR",
        max_rounds: int = 3,
        max_tokens: int = 8000,
        current_date: Optional[str] = None,
    ) -> str:
        """
        Execute writer/reviewer collaboration for a single sub-topic.
        
        Args:
            sub_topic: The research sub-topic to investigate
            question: The original user question
            sub_topic_contexts: Context information for the sub-topic
            locale: Language locale (default: ko-KR)
            max_rounds: Maximum conversation rounds (default: 3)
            max_tokens: Maximum tokens for response (default: 8000)
            current_date: Current date string (default: today)
        
        Returns:
            JSON string with conversation results
        """
        started_at = datetime.utcnow()
        if current_date is None:
            current_date = started_at.strftime("%Y-%m-%d")
        
        # Normalize inputs
        try:
            max_rounds = int(max_rounds)
        except Exception:
            max_rounds = 3
        if max_rounds < 1:
            max_rounds = 3
        
        try:
            max_tokens = int(max_tokens)
        except Exception:
            max_tokens = 8000
        
        # Make max_rounds odd so writer gets final word
        if max_rounds % 2 == 0:
            max_rounds += 1
        
        logger.info(
            f"[GroupChatAF] Starting sub_topic='{sub_topic}' rounds={max_rounds}"
        )
        
        # Context size guard
        MAX_CONTEXT_CHARS = 15000
        trimmed_context = sub_topic_contexts
        if len(trimmed_context) > MAX_CONTEXT_CHARS:
            trimmed_context = trimmed_context[:MAX_CONTEXT_CHARS] + "\n...[truncated]"
        
        # Build agent instructions
        writer_instructions = RESEARCH_WRITER_PROMPT.format(
            current_date=current_date,
            locale=locale,
            sub_topic=sub_topic,
            question=question,
            contexts=trimmed_context,
            max_tokens=max_tokens
        )
        
        reviewer_instructions = RESEARCH_REVIEWER_PROMPT.format(
            current_date=current_date,
            locale=locale,
            sub_topic=sub_topic,
            question=question,
            contexts=trimmed_context,
            max_tokens=max_tokens
        )
        
        # Add JSON output instructions
        json_instruction = "\n\nIMPORTANT: You must output ONLY valid JSON. No explanatory text before or after. Do not include markdown code blocks."
        
        # Create agents
        writer_agent = ChatAgent(
            chat_client=self.chat_client,
            name="Writer",
            instructions=writer_instructions + json_instruction
        )
        
        reviewer_agent = ChatAgent(
            chat_client=self.chat_client,
            name="Reviewer",
            instructions=reviewer_instructions + json_instruction
        )
        
        # Initialize termination strategy
        termination = ApprovalTerminationStrategy(max_iterations=max_rounds)
        
        # Create initial task message
        task = (
            f"Sub-topic: {sub_topic}\n"
            f"Original Question: {question}\n"
            f"Locale: {locale}\n"
            f"Date: {current_date}\n\n"
            f"Context (may be truncated):\n{trimmed_context}\n\n"
            "Follow the roles: Writer drafts/improves structured answer. "
            "Reviewer critiques/refines and if satisfactory provides final answer with approval."
        )
        
        # Message history
        messages: List[ChatMessage] = [
            ChatMessage(role=Role.USER, contents=[TextContent(text=task)])
        ]
        
        # Tracking
        conversation_history: List[Dict[str, Any]] = []
        current_agent = writer_agent
        error_info: Optional[str] = None
        final_answer = ""
        
        try:
            # Main conversation loop (Group Chat Pattern)
            iteration = 0
            while not termination.should_terminate(messages):
                iteration += 1
                logger.info(f"[GroupChatAF] Iteration {iteration}, Agent: {current_agent.name}")
                
                # Get agent response
                last_message_text = messages[-1].text if messages else task
                response = await current_agent.run(last_message_text)
                
                # Log response
                logger.info(
                    f"[GroupChatAF] {current_agent.name} response length: {len(response.text)}"
                )
                
                # Validate JSON
                is_valid_json = self._validate_json(response.text)
                if not is_valid_json:
                    logger.warning(
                        f"[GroupChatAF] {current_agent.name} produced non-JSON: {response.text[:100]}..."
                    )
                
                # Record message
                response_message = ChatMessage(
                    role=Role.ASSISTANT,
                    contents=[TextContent(text=response.text)],
                    author_name=current_agent.name
                )
                messages.append(response_message)
                
                conversation_history.append({
                    "idx": len(conversation_history),
                    "time": datetime.utcnow().isoformat(),
                    "role": "assistant",
                    "agent": current_agent.name,
                    "content": response.text,
                })
                
                # Switch agent (turn-based)
                current_agent = reviewer_agent if current_agent == writer_agent else writer_agent
            
            # Extract final answer (last message)
            if messages and len(messages) > 1:
                final_answer = messages[-1].text
                final_answer = self._clean_and_validate_json(final_answer)
            else:
                final_answer = "No response generated"
            
            logger.info(f"[GroupChatAF] Completed successfully, rounds: {iteration}")
            
        except Exception as e:
            error_info = f"{type(e).__name__}: {e}"
            logger.exception(f"[GroupChatAF] Error: {error_info}")
            final_answer = f"Orchestration failed: {str(e)}"
        
        # Build result
        result = {
            "status": "success" if error_info is None else "error",
            "sub_topic": str(sub_topic),
            "question": str(question),
            "final_answer": str(final_answer),
            "rounds_used": len(conversation_history),
            "max_rounds_requested": int(max_rounds),
            "started_at": started_at.isoformat(),
            "ended_at": datetime.utcnow().isoformat(),
        }
        
        if error_info:
            result["error"] = str(error_info)
        
        # Serialize result
        try:
            json_result = json.dumps(result, ensure_ascii=False)
            logger.info(f"[GroupChatAF] Result serialized, length: {len(json_result)}")
            return json_result
        except Exception as serial_error:
            logger.error(f"[GroupChatAF] Serialization error: {serial_error}")
            return json.dumps(
                {
                    "status": "error",
                    "sub_topic": str(sub_topic),
                    "error": f"serialization_failure: {str(serial_error)}",
                },
                ensure_ascii=False,
            )
    
    def _validate_json(self, content: str) -> bool:
        """Validate if content is valid JSON."""
        try:
            content_stripped = content.strip()
            if content_stripped.startswith('{') and content_stripped.endswith('}'):
                json.loads(content_stripped)
                return True
        except json.JSONDecodeError:
            pass
        return False
    
    def _clean_and_validate_json(self, content: str) -> str:
        """Clean and validate JSON response."""
        try:
            # Remove whitespace
            content = content.strip()
            
            # Remove markdown code blocks
            if content.startswith('```'):
                lines = content.split('\n')
                json_lines = []
                in_json = False
                for line in lines:
                    if line.strip().startswith('```'):
                        if not in_json:
                            in_json = True
                        else:
                            break
                    elif in_json:
                        json_lines.append(line)
                content = '\n'.join(json_lines).strip()
            
            # Extract JSON part (from first { to last })
            start_idx = content.find('{')
            end_idx = content.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                content = content[start_idx:end_idx + 1]
            
            # Validate JSON
            parsed = json.loads(content)
            
            # Re-serialize to clean format
            clean_json = json.dumps(parsed, ensure_ascii=False, separators=(',', ':'))
            
            logger.info("[GroupChatAF] Successfully cleaned and validated JSON")
            return clean_json
            
        except json.JSONDecodeError as e:
            logger.error(f"[GroupChatAF] JSON validation failed: {e}")
            logger.error(f"[GroupChatAF] Problematic content: {content[:500]}...")
            
            # Fallback JSON
            return json.dumps({
                "sub_topic": "Unknown",
                "final_answer": content[:1000] if content else "No response generated",
                "error": "json_parsing_failed"
            }, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"[GroupChatAF] Unexpected error in JSON cleaning: {e}")
            return json.dumps({
                "sub_topic": "Unknown",
                "final_answer": "Processing error occurred",
                "error": str(e)
            }, ensure_ascii=False)


class GroupChattingExecutorAF(Executor):
    """
    Agent Framework Executor for group chatting (writer/reviewer collaboration).
    
    This executor implements the same writer/reviewer loop as the standalone
    GroupChattingExecutor, but as a proper AF Executor that can be used in workflows.
    """
    
    def __init__(
        self,
        id: str,
        chat_client: AzureOpenAIChatClient,
        settings: Settings,
        context_max_chars: int = 20000,
        writer_parallel_limit: int = 4
    ):
        super().__init__(id=id)
        self.chat_client = chat_client
        self.settings = settings
        self.context_max_chars = context_max_chars
        
        logger.info(f"GroupChattingExecutorAF initialized")
    
    @handler
    async def run_group_chat(
        self,
        research_data: Dict[str, Any],
        ctx: WorkflowContext[Dict[str, Any], str]
    ) -> None:
        """
        Run group chat for each sub-topic with streaming progress.
        
        Args:
            research_data: Dictionary with research parameters
            ctx: Workflow context for sending results and yielding progress
        """
        try:
            # Extract parameters
            question = (
                research_data.get("question") or 
                research_data.get("enriched_query") or 
                research_data.get("original_query") or 
                ""
            )
            sub_topics = research_data.get("sub_topics", [])
            
            # Get metadata
            metadata = research_data.get("metadata", {})
            locale = metadata.get("locale", "ko-KR")
            verbose = metadata.get("verbose", False)
            max_tokens = metadata.get("max_tokens", 8000)
            
            logger.info(f"[GroupChattingExecutorAF] Starting group chat for {len(sub_topics)} sub-topics")
            
            # Get locale messages
            LOCALE_MSG = LOCALE_MESSAGES.get(locale, LOCALE_MESSAGES["ko-KR"])
            
            # If no sub_topics, create default
            if not sub_topics:
                sub_topics = [{
                    "sub_topic": "research report",
                    "queries": [question]
                }]
            
            # ✅ Yield starting message
            await ctx.yield_output(f"data: ### {LOCALE_MSG.get('start_research', 'Starting research')}...\n\n")
            
            all_results = []
            
            # Process each sub-topic
            for idx, sub_topic_data in enumerate(sub_topics, 1):
                sub_topic_name = sub_topic_data.get("sub_topic", f"Topic {idx}")
                sub_topic_queries = sub_topic_data.get("queries", [question])
                
                # Get context for this sub-topic
                sub_topic_contexts = research_data.get("sub_topic_contexts", {})
                if isinstance(sub_topic_contexts, dict):
                    context = sub_topic_contexts.get(sub_topic_name, "")
                else:
                    context = str(sub_topic_contexts)
                
                # ✅ Yield sub-topic start
                await ctx.yield_output(f"data: ### 📝 Writer phase for '{sub_topic_name}' [{idx}/{len(sub_topics)}]\n\n")
                
                # Execute group chat for this sub-topic
                result = await self._execute_single_sub_topic(
                    sub_topic=sub_topic_name,
                    question=", ".join(sub_topic_queries),
                    sub_topic_contexts=context,
                    locale=locale,
                    max_tokens=max_tokens,
                    ctx=ctx
                )
                
                all_results.append(result)
                
                # ✅ Yield completion for this sub-topic
                status = result.get("status", "unknown")
                if status == "success":
                    await ctx.yield_output(f"data: ### ✅ Completed '{sub_topic_name}' [{idx}/{len(sub_topics)}]\n\n")
                else:
                    await ctx.yield_output(f"data: ### ❌ Failed '{sub_topic_name}' [{idx}/{len(sub_topics)}]\n\n")
            
            # ✅ Yield overall completion
            success_count = sum(1 for r in all_results if r.get("status") == "success")
            await ctx.yield_output(f"data: ### ✅ Group chat completed: {success_count}/{len(all_results)} successful\n\n")
            
            # Send results to next executor
            await ctx.send_message({
                **research_data,
                "group_chat_results": {
                    "status": "success",
                    "results": all_results,
                    "total_sub_topics": len(sub_topics),
                    "successful": success_count
                }
            })
            
        except Exception as e:
            error_msg = f"Group chat failed: {str(e)}"
            logger.error(f"[GroupChattingExecutorAF] {error_msg}")
            await ctx.yield_output(f"data: ### ❌ {error_msg}\n\n")
            await ctx.send_message({
                **research_data,
                "group_chat_results": {
                    "status": "error",
                    "message": error_msg,
                    "results": []
                }
            })
    
    async def _execute_single_sub_topic(
        self,
        sub_topic: str,
        question: str,
        sub_topic_contexts: str,
        locale: str,
        max_tokens: int,
        ctx: WorkflowContext[Dict[str, Any], str]
    ) -> Dict[str, Any]:
        """Execute writer/reviewer loop for a single sub-topic."""
        
        started_at = datetime.utcnow()
        current_date = started_at.strftime("%Y-%m-%d")
        max_rounds = 3  # Fixed for now
        
        # Context size guard
        MAX_CONTEXT_CHARS = 15000
        trimmed_context = sub_topic_contexts
        if len(trimmed_context) > MAX_CONTEXT_CHARS:
            trimmed_context = trimmed_context[:MAX_CONTEXT_CHARS] + "\n...[truncated]"
        
        # Build agent instructions
        writer_instructions = RESEARCH_WRITER_PROMPT.format(
            current_date=current_date,
            locale=locale,
            sub_topic=sub_topic,
            question=question,
            contexts=trimmed_context,
            max_tokens=max_tokens
        )
        
        reviewer_instructions = RESEARCH_REVIEWER_PROMPT.format(
            current_date=current_date,
            locale=locale,
            sub_topic=sub_topic,
            question=question,
            contexts=trimmed_context,
            max_tokens=max_tokens
        )
        
        json_instruction = "\n\nIMPORTANT: You must output ONLY valid JSON."
        
        # Create agents
        writer_agent = ChatAgent(
            chat_client=self.chat_client,
            name="Writer",
            instructions=writer_instructions + json_instruction
        )
        
        reviewer_agent = ChatAgent(
            chat_client=self.chat_client,
            name="Reviewer",
            instructions=reviewer_instructions + json_instruction
        )
        
        # Initialize termination
        termination = ApprovalTerminationStrategy(max_iterations=max_rounds)
        
        # Create task
        task = (
            f"Sub-topic: {sub_topic}\n"
            f"Question: {question}\n"
            f"Context: {trimmed_context[:500]}...\n\n"
            "Writer: Draft answer. Reviewer: Review and approve if satisfactory."
        )
        
        messages: List[ChatMessage] = [
            ChatMessage(role=Role.USER, contents=[TextContent(text=task)])
        ]
        
        conversation_history = []
        current_agent = writer_agent
        error_info = None
        final_answer = ""
        writer_count = 0
        reviewer_count = 0
        
        try:
            iteration = 0
            while not termination.should_terminate(messages):
                iteration += 1
                is_writer = (current_agent == writer_agent)
                
                if is_writer:
                    writer_count += 1
                    # ✅ Yield writer progress
                    await ctx.yield_output(f"data: ✅ Writer round {writer_count}: {sub_topic}\n\n")
                else:
                    reviewer_count += 1
                    if reviewer_count == 1:
                        # ✅ Yield reviewer phase start
                        await ctx.yield_output(f"data: ### 🔍 Reviewer phase for '{sub_topic}'\n\n")
                    # ✅ Yield reviewer progress
                    await ctx.yield_output(f"data: 🔍 Reviewer round {reviewer_count}: {sub_topic}\n\n")
                
                # Get response
                last_message_text = messages[-1].text if messages else task
                response = await current_agent.run(last_message_text)
                
                # Record message
                response_message = ChatMessage(
                    role=Role.ASSISTANT,
                    contents=[TextContent(text=response.text)],
                    author_name=current_agent.name
                )
                messages.append(response_message)
                
                conversation_history.append({
                    "agent": current_agent.name,
                    "content": response.text,
                })
                
                # Check for approval
                if not is_writer and "approved" in response.text.lower():
                    await ctx.yield_output(f"data: ✅ '{sub_topic}' approved!\n\n")
                
                # Switch agent
                current_agent = reviewer_agent if current_agent == writer_agent else writer_agent
            
            # Extract final answer
            if messages and len(messages) > 1:
                final_answer = messages[-1].text
            
        except Exception as e:
            error_info = str(e)
            logger.error(f"Error in group chat for '{sub_topic}': {e}")
            await ctx.yield_output(f"data: ❌ Error: {e}\n\n")
        
        return {
            "status": "success" if error_info is None else "error",
            "sub_topic": sub_topic,
            "question": question,
            "final_answer": final_answer,
            "rounds_used": iteration,
            "writer_rounds": writer_count,
            "reviewer_rounds": reviewer_count,
            "error": error_info
        }
