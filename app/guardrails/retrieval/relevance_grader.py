from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from app.config import settings

GRADE_PROMPT = ChatPromptTemplate.from_template(
    """You are grading whether a retrieved passage is relevant to a user's question.

Question: {query}

Passage:
{passage}

Is this passage relevant enough to help answer the question? Answer with only Yes or No."""
)


class RelevanceGrader:
    def __init__(self):
        self.llm = ChatOllama(model=settings.guardrail_llm_model, base_url=settings.ollama_base_url, temperature=0)
        self.chain = GRADE_PROMPT | self.llm

    def grade(self, query: str, passage: str) -> bool:
        result = self.chain.invoke({"query": query, "passage": passage[:2000]})
        return result.content.strip().lower().startswith("yes")

    def filter_relevant(self, query: str, candidates: list[dict]) -> list[dict]:
        return [c for c in candidates if self.grade(query, c["text"])]