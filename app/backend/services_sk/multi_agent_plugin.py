"""Multi-agent research plugin for orchestrating writer and reviewer workflows.

This plugin coordinates parallel writer drafts followed by sequential reviews
using Azure OpenAI completions and project-specific prompts.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pytz
from langchain.prompts import load_prompt
from openai import AsyncAzureOpenAI
from semantic_kernel.functions import kernel_function

from config.config import Settings

logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))

RESEARCH_WRITER_PROMPT = load_prompt(
    os.path.join(current_dir, "..", "prompts", "research_writer_prompt.yaml"),
    encoding="utf-8",
)
RESEARCH_REVIEWER_PROMPT = load_prompt(
    os.path.join(current_dir, "..", "prompts", "research_reviewer_prompt.yaml"),
    encoding="utf-8",
)


def _safe_int(value: Union[str, int, None], default: int) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _safe_bool(value: Union[str, bool, None], default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return default


def _safe_json_loads(payload: Optional[str]) -> Any:
    if not payload:
        return None
    try:
        return json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        return None


@dataclass
class SubTopicTask:
    sub_topic: str
    question: str
    contexts: str
    max_tokens: int


class MultiAgentPlugin:
    """Semantic Kernel plugin that runs writer and reviewer agents per sub-topic."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = AsyncAzureOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        )
        self.deployment_name = settings.AZURE_OPENAI_DEPLOYMENT_NAME
        self.default_max_tokens = settings.MAX_TOKENS
        self.context_max_chars = _safe_int(
            os.getenv("MULTI_AGENT_CONTEXT_MAX_CHARS"), 20000
        )
        self.writer_parallel_limit = max(
            1,
            _safe_int(os.getenv("MULTI_AGENT_WRITER_CONCURRENCY"), 4),
        )
        if isinstance(settings.TIME_ZONE, str):
            try:
                self.timezone = pytz.timezone(settings.TIME_ZONE)
            except Exception:
                self.timezone = pytz.UTC
        else:
            self.timezone = pytz.UTC

    @kernel_function(
        description=(
            "Run multi-agent research workflow: writers draft answers per sub-topic "
            "in parallel, reviewers validate sequentially, returning publish-ready "
            "results when all reviews pass."
        ),
        name="run_multi_agent",
    )
    async def run_multi_agent(
        self,
        question: str,
        sub_topic: Optional[str] = None,
        sub_topics: Optional[str] = None,
        sub_topic_contexts: Optional[str] = None,
        contexts: Optional[str] = None,
        locale: str = "ko-KR",
        max_tokens: Union[str, int] = 8000,
        current_date: Optional[str] = None,
        parallel: Union[str, bool, None] = None,
        parallel_limit: Union[str, int, None] = None,
        max_rounds: Optional[Union[str, int]] = None,  # kept for compatibility
    ) -> str:
        """Execute writer/reviewer pipeline and return aggregated JSON string."""

        normalized_max_tokens = _safe_int(max_tokens, self.default_max_tokens)
        parallel_writers = _safe_bool(parallel, True)
        effective_parallel_limit = max(
            1,
            _safe_int(parallel_limit, self.writer_parallel_limit),
        )
        current_date = self._normalize_current_date(current_date)

        tasks = self._normalize_tasks(
            question=question,
            sub_topic=sub_topic,
            sub_topics=sub_topics,
            sub_topic_contexts=sub_topic_contexts,
            contexts=contexts,
            max_tokens=normalized_max_tokens,
        )

        if not tasks:
            logger.warning("[MultiAgent] No valid sub-topics resolved; returning fallback response")
            return json.dumps(
                {
                    "status": "error",
                    "message": "No valid sub-topics provided",
                    "question": question,
                    "current_date": current_date,
                },
                ensure_ascii=False,
            )

        writer_results = await self._run_writers(
            tasks=tasks,
            locale=locale,
            current_date=current_date,
            parallel=parallel_writers,
            parallel_limit=effective_parallel_limit,
        )

        reviewer_results = await self._run_reviewers(
            writer_results=writer_results,
            locale=locale,
            current_date=current_date,
        )

        all_ready = all(
            r["review"].get("ready_to_publish")
            for r in reviewer_results
            if r["review"]["status"] == "success"
        ) and all(r["review"]["status"] == "success" for r in reviewer_results)

        final_payload: Dict[str, Any] = {
            "status": "success" if all_ready else "pending_review",
            "question": question,
            "current_date": current_date,
            "locale": locale,
            "all_ready_to_publish": all_ready,
            "sub_topic_results": reviewer_results,
        }

        if all_ready:
            final_answers = {
                item["sub_topic"]: item["review"]["parsed"].get(
                    "revised_answer_markdown"
                )
                for item in reviewer_results
            }
            final_payload["final_answer"] = json.dumps(
                {
                    "question": question,
                    "answers": final_answers,
                },
                ensure_ascii=False,
            )
        else:
            final_payload["final_answer"] = ""

        return json.dumps(final_payload, ensure_ascii=False)

    async def _run_writers(
        self,
        tasks: List[SubTopicTask],
        locale: str,
        current_date: str,
        parallel: bool,
        parallel_limit: int,
    ) -> List[Dict[str, Any]]:
        semaphore = asyncio.Semaphore(parallel_limit)

        async def _wrapped(task: SubTopicTask) -> Dict[str, Any]:
            async with semaphore:
                return await self._invoke_writer(task, locale, current_date)

        if parallel and len(tasks) > 1:
            results = await asyncio.gather(
                *[asyncio.create_task(_wrapped(task)) for task in tasks],
                return_exceptions=True,
            )
            normalized: List[Dict[str, Any]] = []
            for task, result in zip(tasks, results):
                if isinstance(result, Exception):
                    logger.exception(
                        "[MultiAgent] Writer task failed for sub-topic '%s'", task.sub_topic
                    )
                    normalized.append(
                        {
                            "sub_topic": task.sub_topic,
                            "question": task.question,
                            "writer": {
                                "status": "error",
                                "error": f"{type(result).__name__}: {result}",
                                "raw_output": "",
                                "parsed": None,
                            },
                            "task": task,
                        }
                    )
                else:
                    normalized.append({"sub_topic": task.sub_topic, **result, "task": task})
            return normalized

        sequential_results = []
        for task in tasks:
            try:
                result = await _wrapped(task)
                sequential_results.append({"sub_topic": task.sub_topic, **result, "task": task})
            except Exception as exc:
                logger.exception(
                    "[MultiAgent] Writer task failed for sub-topic '%s'", task.sub_topic
                )
                sequential_results.append(
                    {
                        "sub_topic": task.sub_topic,
                        "question": task.question,
                        "writer": {
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                            "raw_output": "",
                            "parsed": None,
                        },
                        "task": task,
                    }
                )
        return sequential_results

    async def _run_reviewers(
        self,
        writer_results: List[Dict[str, Any]],
        locale: str,
        current_date: str,
    ) -> List[Dict[str, Any]]:
        reviewed: List[Dict[str, Any]] = []
        for item in writer_results:
            writer_info = item.get("writer", {})
            task: SubTopicTask = item.get("task")
            if writer_info.get("status") != "success" or not writer_info.get("parsed"):
                reviewed.append(
                    {
                        "sub_topic": item.get("sub_topic"),
                        "question": item.get("question"),
                        "writer": writer_info,
                        "review": {
                            "status": "skipped",
                            "reason": "Writer output unavailable",
                            "parsed": {},
                        },
                    }
                )
                continue

            review_result = await self._invoke_reviewer(
                task=task,
                writer_parsed=writer_info["parsed"],
                locale=locale,
                current_date=current_date,
            )
            reviewed.append(
                {
                    "sub_topic": item.get("sub_topic"),
                    "question": item.get("question"),
                    "writer": writer_info,
                    "review": review_result,
                }
            )
        return reviewed

    async def _invoke_writer(
        self, task: SubTopicTask, locale: str, current_date: str
    ) -> Dict[str, Any]:
        prompt = RESEARCH_WRITER_PROMPT.format(
            current_date=current_date,
            locale=locale,
            sub_topic=task.sub_topic,
            question=task.question,
            contexts=task.contexts,
            max_tokens=task.max_tokens,
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": task.question},
        ]

        try:
            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=0.3,
                max_tokens=min(task.max_tokens, self.default_max_tokens),
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content.strip()
            parsed = json.loads(content)
            return {
                "question": task.question,
                "writer": {
                    "status": "success",
                    "raw_output": content,
                    "parsed": parsed,
                },
            }
        except Exception as exc:
            logger.exception(
                "[MultiAgent] Writer call failed for sub-topic '%s'", task.sub_topic
            )
            return {
                "question": task.question,
                "writer": {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "raw_output": "",
                    "parsed": None,
                },
            }

    async def _invoke_reviewer(
        self,
        task: SubTopicTask,
        writer_parsed: Dict[str, Any],
        locale: str,
        current_date: str,
    ) -> Dict[str, Any]:
        prompt = RESEARCH_REVIEWER_PROMPT.format(
            current_date=current_date,
            locale=locale,
            sub_topic=task.sub_topic,
            question=task.question,
            contexts=task.contexts,
            max_tokens=task.max_tokens,
        )

        user_message = (
            "Here is the writer draft JSON to review and improve:\n"
            f"{json.dumps(writer_parsed, ensure_ascii=False)}"
        )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_message},
        ]

        try:
            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=0.2,
                max_tokens=min(task.max_tokens, self.default_max_tokens),
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content.strip()
            parsed = json.loads(content)
            ready = bool(parsed.get("ready_to_publish"))
            return {
                "status": "success",
                "raw_output": content,
                "parsed": parsed,
                "ready_to_publish": ready,
            }
        except Exception as exc:
            logger.exception(
                "[MultiAgent] Reviewer call failed for sub-topic '%s'", task.sub_topic
            )
            return {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "parsed": {},
                "raw_output": "",
                "ready_to_publish": False,
            }

    def _normalize_tasks(
        self,
        question: str,
        sub_topic: Optional[str],
        sub_topics: Optional[str],
        sub_topic_contexts: Optional[str],
        contexts: Optional[str],
        max_tokens: int,
    ) -> List[SubTopicTask]:
        tasks: List[SubTopicTask] = []
        parsed_payload = _safe_json_loads(sub_topics)

        if isinstance(parsed_payload, dict) and "sub_topics" in parsed_payload:
            parsed_payload = parsed_payload.get("sub_topics")

        if isinstance(parsed_payload, list):
            for entry in parsed_payload:
                if not isinstance(entry, dict):
                    continue
                name = str(entry.get("sub_topic") or entry.get("name") or "sub-topic")
                entry_question = str(entry.get("question") or question or "")
                entry_context = str(
                    entry.get("contexts")
                    or entry.get("context")
                    or entry.get("sub_topic_contexts")
                    or ""
                )
                entry_tokens = _safe_int(entry.get("max_tokens"), max_tokens)
                tasks.append(
                    SubTopicTask(
                        sub_topic=name,
                        question=entry_question,
                        contexts=self._trim_context(entry_context),
                        max_tokens=entry_tokens,
                    )
                )

        if tasks:
            return tasks

        context_map = _safe_json_loads(sub_topic_contexts)
        if isinstance(context_map, dict) and context_map:
            for name, ctx in context_map.items():
                tasks.append(
                    SubTopicTask(
                        sub_topic=str(name),
                        question=question,
                        contexts=self._trim_context(str(ctx)),
                        max_tokens=max_tokens,
                    )
                )
            if tasks:
                return tasks

        single_context = sub_topic_contexts or contexts or ""
        resolved_sub_topic = sub_topic or "research sub-topic"
        tasks.append(
            SubTopicTask(
                sub_topic=resolved_sub_topic,
                question=question or resolved_sub_topic,
                contexts=self._trim_context(single_context),
                max_tokens=max_tokens,
            )
        )
        return tasks

    def _trim_context(self, context: str) -> str:
        if not context:
            return ""
        if len(context) <= self.context_max_chars:
            return context
        return context[: self.context_max_chars] + "\n...[truncated]"

    def _normalize_current_date(self, current_date: Optional[str]) -> str:
        if current_date:
            return current_date
        return datetime.now(tz=self.timezone).strftime("%Y-%m-%d")


async def run_multi_agent(**kwargs) -> str:
    """Convenience function for quick invocation with environment settings."""

    settings = Settings()  # Loads from environment / .env
    plugin = MultiAgentPlugin(settings)
    return await plugin.run_multi_agent(**kwargs)