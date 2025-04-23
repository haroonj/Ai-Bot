from typing import Optional

from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    # OpenAI
    openai_api_key: str = Field(..., validation_alias='OPENAI_API_KEY')
    embedding_model_name: str = "text-embedding-3-small"
    llm_model_name: str = "gpt-4o-mini"

    # PGVector
    postgres_host: str = Field("localhost", validation_alias='POSTGRES_HOST')
    postgres_port: int = Field(5333, validation_alias='POSTGRES_PORT')
    postgres_db: str = Field("ragbot", validation_alias='POSTGRES_DB')
    postgres_user: str = Field("raguser", validation_alias='POSTGRES_USER')
    postgres_password: str = Field("ragpass", validation_alias='POSTGRES_PASSWORD')
    vector_store_collection_name: str = Field("ecommerce_kb", validation_alias='VECTOR_STORE_COLLECTION_NAME')

    @property
    def pgvector_connection_string(self) -> str:
        return f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    mock_api_base_url: HttpUrl = Field("http://localhost:8001", validation_alias='MOCK_API_BASE_URL')

    # Langsmith (Optional)
    langchain_tracing_v2: Optional[str] = Field(None, validation_alias='LANGCHAIN_TRACING_V2')
    langchain_endpoint: Optional[str] = Field(None, validation_alias='LANGCHAIN_ENDPOINT')
    langchain_api_key: Optional[str] = Field(None, validation_alias='LANGCHAIN_API_KEY')
    langchain_project: Optional[str] = Field(None, validation_alias='LANGCHAIN_PROJECT')


settings = Settings()


def setup_langsmith():
    if (settings.langchain_tracing_v2 == "true" and
            settings.langchain_api_key and
            settings.langchain_project):
        import os
        os.environ["LANGCHAIN_TRACING_V2"] = settings.langchain_tracing_v2
        os.environ["LANGCHAIN_ENDPOINT"] = str(settings.langchain_endpoint)  # Ensure string
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
        print("Langsmith tracing enabled.")
    else:
        print("Langsmith tracing is not configured or disabled.")


setup_langsmith()
