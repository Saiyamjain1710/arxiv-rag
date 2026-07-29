from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from app.config import settings

_llm = ChatOllama(model=settings.main_llm_model, base_url=settings.ollama_base_url, temperature=0.2)


async def stream_answer(query: str, system_prompt: str):
    """Async generator yielding text tokens as they're produced by Qwen 3."""
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=query)]
    async for chunk in _llm.astream(messages):
        if chunk.content:
            yield chunk.content