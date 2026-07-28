from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    qdrant_url: str = "http://localhost:6333"
    ollama_base_url: str = "http://localhost:11434"
    main_llm_model: str = "qwen3:8b"
    guardrail_llm_model: str = "qwen3:1.7b"
    embedding_model: str = "BAAI/bge-m3"
    max_tokens_per_request: int = 2000
    max_tokens_per_session: int = 20000
    confidence_tau_high: float = 0.7
    confidence_tau_low: float = 0.4
    arxiv_max_fetch: int = 3

    class Config:
        env_file = ".env"


settings = Settings()