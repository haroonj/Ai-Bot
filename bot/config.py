from typing import Optional
import os

from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    # OpenAI
    openai_api_key: str = Field(..., validation_alias='OPENAI_API_KEY')
    embedding_model_name: str = "text-embedding-3-small"
    llm_model_name: str = "gpt-4o-mini"

    # FAISS Index Path (relative to project root)
    # Ensure this path is writable by the user running the script/server
    # It's often best placed within the project directory for easy management.
    faiss_index_path: str = Field("./faiss_index", validation_alias='FAISS_INDEX_PATH')

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