from pathlib import Path

from dotenv import load_dotenv
from pydantic import computed_field
from pydantic_settings import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    app_env: str = "development"

    # Neo4j configuration
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "change-me"

    db_names: dict[str, str] = {  # WARNING: Neo4j Community Edition only supports one default database named "neo4j" dont add more unless using Enterprise Edition
        "default": "neo4j",
        "system": "system",
    }

    @computed_field
    @property
    def neo4j_auth(self) -> tuple[str, str]:
        return self.neo4j_user, self.neo4j_password
    
    
    # Ollama configuration
    ollama_base_url: str = "http://localhost:11433"
    ollama_embed_model: str = "qwen3-embedding:0.6b"
    ollama_chat_model: str = "gemma4:31b-cloud" #TODO: implement signin logic when ollama container is started the first time
    ollama_embed_dimensions: int = 768
    embedding_batch_size: int = 32
    max_context_length: int = 64000

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


    # Workflow configuration
    semantic_chunking_threshold: int = 97
    combined_lines_buffer_size: int = 2
    num_chunks_per_turn: int = 5

    # Vocabulary configuration
    rdf_skolem_prefix: str = "bnode"
    rdf_skolem_base: str = "http://example.org/.well-known/bnodes/"

    # Startup behavior
    skip_initial_vocab_import: bool = True
    skip_model_pull: bool = False
    generate_missing_embeddings_on_startup: bool = False

    # Frontend configuration
    frontend_base_urls: str = "http://localhost:3000,http://127.0.0.1:3000"

    @computed_field
    @property
    def frontend_cors_origins(self) -> list[str]:
        return [item.strip() for item in self.frontend_base_urls.split(",") if item.strip()]


settings = Settings()
