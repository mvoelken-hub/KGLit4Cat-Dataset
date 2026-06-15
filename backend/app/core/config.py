from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, computed_field
from pydantic_settings import BaseSettings

load_dotenv()


def _resolve_service_host(hostname: str | None, *, production_host: str, development_host: str, app_env: str) -> str:
    if hostname and hostname.strip().lower() not in {"localhost", "127.0.0.1", "::1"}:
        return hostname.strip()
    return production_host if app_env == "production" else development_host


class Settings(BaseSettings):
    app_env: str = "development"

    # Neo4j configuration
    neo4j_hostname: str | None = None
    neo4j_port: int = 7687
    neo4j_user: str = "neo4j"
    neo4j_password: str = "change-me"

    db_names: dict[str, str] = {  # WARNING: Neo4j Community Edition only supports one default database named "neo4j" dont add more unless using Enterprise Edition
        "default": "neo4j",
        "system": "system",
    }

    @computed_field
    @property
    def neo4j_uri(self) -> str:
        host = _resolve_service_host(self.neo4j_hostname, production_host="neo4j", development_host="localhost", app_env=self.app_env)
        return f"bolt://{host}:{self.neo4j_port}"

    @computed_field
    @property
    def neo4j_auth(self) -> tuple[str, str]:
        return self.neo4j_user, self.neo4j_password
    
    
    # Ollama configuration
    ollama_hostname: str | None = None
    ollama_port: int = 11433
    ollama_embed_model: str = "qwen3-embedding:0.6b"
    ollama_chat_model: str = "gemma4:31b-cloud"
    ollama_chat_tokenizer: str = ""
    hf_token: str = Field(default="", validation_alias="HF_TOKEN", exclude=True)
    ollama_embed_dimensions: int = 768
    embedding_batch_size: int = 32
    max_context_length: int = 8192
    ollama_timeout_seconds: int | None = Field(default=None, validation_alias="OLLAMA_TIMEOUT_SECONDS")
    extraction_vocab_query_concurrency: int = 4
    vocab_selection_llm_concurrency: int = 1
    vocab_selection_parallel_mode: str = "conservative"
    ollama_flash_attention: bool = False
    ollama_kv_cache_type: str = "f16"
    ollama_embed_num_gpu: int = -1  # -1 = auto (all layers on GPU), 0 = CPU only

    @computed_field
    @property
    def ollama_base_url(self) -> str:
        host = _resolve_service_host(self.ollama_hostname, production_host="ollama", development_host="localhost", app_env=self.app_env)
        return f"http://{host}:{self.ollama_port}"

    # Runtime directory configuration
    runtime_dir: Path = Path("./.runtime")

    @computed_field
    @property
    def uploads_dir(self) -> Path:
        return self.runtime_dir / "uploads"
    
    @computed_field
    @property
    def vocab_storage_dir(self) -> Path:
        return self.runtime_dir / "vocabs"

    @computed_field
    @property
    def dcat_profiles_dir(self) -> Path:
        return self.runtime_dir / "profiles"
    
    @computed_field
    @property
    def output_dir(self) -> Path:
        return self.runtime_dir / "output"


    # Vocabulary configuration
    rdf_skolem_prefix: str = "bnode"
    rdf_skolem_base: str = "http://example.org/.well-known/bnodes/"

    # Startup behavior
    skip_initial_vocab_import_override: bool | None = Field(default=None, validation_alias="SKIP_INITIAL_VOCAB_IMPORT")
    skip_model_pull_override: bool | None = Field(default=None, validation_alias="SKIP_MODEL_PULL")

    @computed_field
    @property
    def skip_initial_vocab_import(self) -> bool:
        if self.skip_initial_vocab_import_override is not None:
            return self.skip_initial_vocab_import_override
        return self.app_env != "production"

    @computed_field
    @property
    def skip_model_pull(self) -> bool:
        if self.skip_model_pull_override is not None:
            return self.skip_model_pull_override
        return self.app_env != "production"

settings = Settings()
