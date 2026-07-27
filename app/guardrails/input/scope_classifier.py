import yaml
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

SCOPE_PROMPT = ChatPromptTemplate.from_template(
    """You are a scope classifier for an academic research assistant that only
answers questions about these research topics: {topics}.

User question: "{query}"

Does this question relate to one of the allowed topics? Answer with only Yes or No."""
)


def load_topics(path: str = "configs/topics.yaml") -> list[str]:
    with open(path) as f:
        return yaml.safe_load(f)["topics"]


class ScopeClassifier:
    def __init__(self, model_name: str, base_url: str, topics_path: str = "configs/topics.yaml"):
        self.llm = ChatOllama(model=model_name, base_url=base_url, temperature=0)
        self.topics = load_topics(topics_path)

    def is_in_scope(self, query: str) -> bool:
        chain = SCOPE_PROMPT | self.llm
        result = chain.invoke({"topics": ", ".join(self.topics), "query": query})
        return result.content.strip().lower().startswith("yes")