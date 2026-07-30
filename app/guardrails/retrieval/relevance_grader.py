from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from app.config import settings

BATCH_GRADE_PROMPT = ChatPromptTemplate.from_template(
    """You are grading whether retrieved passages are relevant to a question.

Question: {query}

Passages:
{passages_block}

For each passage number, answer only "yes" or "no" on its own line, in order,
e.g.:
1: yes
2: no
3: yes"""
)


class RelevanceGrader:
    def __init__(self):
        self.llm = ChatOllama(model=settings.guardrail_llm_model, base_url=settings.ollama_base_url, temperature=0)
        self.chain = BATCH_GRADE_PROMPT | self.llm

    def filter_relevant(self, query: str, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []

        passages_block = "\n\n".join(f"{i+1}: {c['text'][:1000]}" for i, c in enumerate(candidates))
        result = self.chain.invoke({"query": query, "passages_block": passages_block})

        verdicts = {}
        for line in result.content.strip().splitlines():
            if ":" in line:
                num, verdict = line.split(":", 1)
                verdicts[num.strip()] = verdict.strip().lower().startswith("yes")

        return [c for i, c in enumerate(candidates) if verdicts.get(str(i + 1), True)]  # default keep if parsing fails