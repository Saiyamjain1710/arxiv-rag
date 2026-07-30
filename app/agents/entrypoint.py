import asyncio
from app.agents.graph import rag_graph
from app.guardrails.input.token_budget import TokenBudgetManager
from app.guardrails.input.scope_classifier import ScopeClassifier
from app.guardrails.input.prompt_injection import PromptInjectionScanner
from app.config import settings

_budget = TokenBudgetManager(settings.max_tokens_per_request, settings.max_tokens_per_session)
_scope = ScopeClassifier(settings.guardrail_llm_model, settings.ollama_base_url)
_injection = PromptInjectionScanner(settings.guardrail_llm_model, settings.ollama_base_url)


async def _run_guardrails_concurrently(query: str):
    loop = asyncio.get_event_loop()
    injection_task = loop.run_in_executor(None, _injection.is_injection, query)
    scope_task = loop.run_in_executor(None, _scope.is_in_scope, query)
    return await asyncio.gather(injection_task, scope_task)


def answer_query(query: str, session_id: str, topic_filter: str | None = None) -> dict:
    ok, msg = _budget.check_and_consume(session_id, query)
    if not ok:
        return {"blocked": True, "reason": msg, "retrieved_chunks": []}

    is_injection, in_scope = asyncio.run(_run_guardrails_concurrently(query))

    if is_injection:
        return {"blocked": True, "reason": "Message blocked by content policy.", "retrieved_chunks": []}
    if not in_scope:
        return {"blocked": True, "reason": "This question is outside the assistant's supported topics.", "retrieved_chunks": []}

    result = rag_graph.invoke({"query": query, "session_id": session_id, "topic_filter": topic_filter})
    result["blocked"] = False
    return result