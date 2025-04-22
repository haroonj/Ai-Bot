# bot/llm.py

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from .config import settings

# Centralized LLM and Embedding instances
llm = ChatOpenAI(
    model=settings.llm_model_name,
    temperature=0, # Low temperature for predictable tool use/responses
    api_key=settings.openai_api_key,
    max_tokens=500
)

embeddings = OpenAIEmbeddings(
    model=settings.embedding_model_name,
    api_key=settings.openai_api_key
)

# If using function/tool calling, bind tools here or in the graph node
# Example: llm_with_tools = llm.bind_tools(tools_list)