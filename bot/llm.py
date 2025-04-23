from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .config import settings

llm = ChatOpenAI(
    model=settings.llm_model_name,
    temperature=0,
    api_key=settings.openai_api_key,
    max_tokens=500
)

embeddings = OpenAIEmbeddings(
    model=settings.embedding_model_name,
    api_key=settings.openai_api_key
)
