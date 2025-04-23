# tests/conftest.py
import pytest
from unittest.mock import MagicMock, patch
import sys
import os
from pathlib import Path # Import for spec
# Import the actual classes/objects we intend to mock or use for spec
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
import httpx
from langgraph.graph import StateGraph # For runnable mock spec if needed
import bot # Import the top-level package to make submodules accessible for patch.object

# --- Define ALL Mock Objects/Classes Needed by Tests ---

# PGVector Mocks
mock_pgvector_instance = MagicMock(name="MockPGVectorInstance_Session")
MockPGVector_Session = MagicMock(name="MockPGVector_Session", return_value=mock_pgvector_instance)
mock_retriever_instance = MagicMock(name="MockRetrieverInstance_Session")
MockGetRetriever = MagicMock(name="MockGetRetriever_Factory_Session", return_value=mock_retriever_instance)

# Embeddings Mocks
mock_embeddings_instance = MagicMock(name="MockEmbeddingsInstance_Session")
MockOpenAIEmbeddings_Session = MagicMock(name="MockOpenAIEmbeddings_Session", return_value=mock_embeddings_instance)

# HTTPX Client Mocks
mock_httpx_client_instance = MagicMock(name="MockHTTPXClientInstance_Session")
MockHttpxClient_Session = MagicMock(name="MockHTTPXClient_Session", return_value=mock_httpx_client_instance)

# LLM Mocks
mock_llm_instance = MagicMock(name="MockLLMInstance_Session")
MockChatOpenAI_Session = MagicMock(name="MockChatOpenAI_Session", return_value=mock_llm_instance)

# Tool Function Mocks (These will replace the actual functions)
MockGetOrderStatus_Func = MagicMock(name="get_order_status_func")
MockGetTrackingInfo_Func = MagicMock(name="get_tracking_info_func")
MockGetOrderDetails_Func = MagicMock(name="get_order_details_func")
MockInitiateReturn_Func = MagicMock(name="initiate_return_request_func")
MockKBLookup_Func = MagicMock(name="knowledge_base_lookup_func")

# LangGraph Runnable Mock
mock_langgraph_runnable = MagicMock(name="MockLangGraphRunnable_Session") # This is the object we'll inject

# Loader/Splitter/Path Mocks
MockDirectoryLoader_Session = MagicMock(name="MockDirectoryLoader_Session")
mock_loader_instance = MockDirectoryLoader_Session.return_value
MockRecursiveCharacterTextSplitter_Session = MagicMock(name="MockSplitter_Session")
mock_splitter_instance = MockRecursiveCharacterTextSplitter_Session.return_value
MockPathInstance = MagicMock(spec=Path)
MockPath_Session = MagicMock(name="MockPath_Session", return_value=MockPathInstance)


# --- Global Patching Fixture ---
@pytest.fixture(scope='session', autouse=True)
def patch_external_dependencies(session_mocker):
    """
    Apply session-wide patches BEFORE any module imports them.
    """
    # print("Applying session-wide patches...") # Uncomment for debug

    # Patch Classes where they are originally defined
    session_mocker.patch('langchain_openai.ChatOpenAI', MockChatOpenAI_Session)
    session_mocker.patch('langchain_openai.OpenAIEmbeddings', MockOpenAIEmbeddings_Session)
    session_mocker.patch('langchain_community.vectorstores.pgvector.PGVector', MockPGVector_Session)
    session_mocker.patch('httpx.Client', MockHttpxClient_Session)
    session_mocker.patch('langchain_community.document_loaders.directory.Path', MockPath_Session, create=True)
    session_mocker.patch('langchain_community.document_loaders.DirectoryLoader', MockDirectoryLoader_Session)
    session_mocker.patch('langchain.text_splitter.RecursiveCharacterTextSplitter', MockRecursiveCharacterTextSplitter_Session)

    # Patch specific instances created in modules AFTER class patches are applied
    session_mocker.patch('bot.llm.llm', mock_llm_instance)
    session_mocker.patch('bot.llm.embeddings', mock_embeddings_instance)
    session_mocker.patch('bot.tools.client', mock_httpx_client_instance)
    session_mocker.patch('bot.tools.llm_with_tools', mock_llm_instance)
    session_mocker.patch('bot.vector_store.vector_store', mock_pgvector_instance)

    # Patch factory functions where they are defined
    session_mocker.patch('bot.vector_store.get_retriever', MockGetRetriever)
    session_mocker.patch('bot.tools.get_retriever', MockGetRetriever)

    # --- IMPORTANT CHANGE: Patch the variable directly ---
    # Instead of patching get_runnable, patch the actual variable holding the runnable in main.py
    session_mocker.patch('main.langgraph_runnable', mock_langgraph_runnable)
    # Keep the patch for bot.graph.app as well, just in case it's used elsewhere
    session_mocker.patch('bot.graph.app', mock_langgraph_runnable)
    print("Patched main.langgraph_runnable directly.")

    # Patch the actual tool functions in bot.tools with our specific mocks
    session_mocker.patch('bot.tools.get_order_status', MockGetOrderStatus_Func)
    session_mocker.patch('bot.tools.get_tracking_info', MockGetTrackingInfo_Func)
    session_mocker.patch('bot.tools.get_order_details', MockGetOrderDetails_Func)
    session_mocker.patch('bot.tools.initiate_return_request', MockInitiateReturn_Func)
    session_mocker.patch('bot.tools.knowledge_base_lookup', MockKBLookup_Func)

    # print("Finished applying session patches.")
    yield
    # print("Stopping session patches...") # Cleanup usually handled by pytest


# --- Fixture to reset mocks BEFORE EACH TEST ---
@pytest.fixture(autouse=True)
def reset_session_mocks():
     """Resets mock instances before each test."""
     # print("Resetting mocks...") # Debug print

     # Reset Instances and ensure methods exist
     mock_pgvector_instance.reset_mock()
     mock_pgvector_instance.as_retriever = MagicMock(return_value=mock_retriever_instance)
     mock_pgvector_instance.add_documents = MagicMock()

     mock_embeddings_instance.reset_mock()
     mock_embeddings_instance.embed_query = MagicMock(return_value=[0.1] * 1536)
     mock_embeddings_instance.embed_documents = MagicMock(return_value=[[0.1] * 1536])

     mock_httpx_client_instance.reset_mock()
     mock_httpx_client_instance.get = MagicMock()
     mock_httpx_client_instance.post = MagicMock()

     mock_llm_instance.reset_mock()
     mock_llm_instance.invoke = MagicMock()
     mock_llm_instance.bind_tools = MagicMock(return_value=mock_llm_instance)

     # Reset the MAIN runnable mock used by main.py
     mock_langgraph_runnable.reset_mock()
     mock_langgraph_runnable.invoke = MagicMock() # Ensure invoke exists and is fresh
     mock_langgraph_runnable.side_effect = None # Clear any side effects like exceptions

     mock_retriever_instance.reset_mock()
     mock_retriever_instance.invoke = MagicMock()

     # Reset Tool Function Mocks (defined above)
     MockGetOrderStatus_Func.reset_mock()
     MockGetOrderStatus_Func.invoke = MagicMock()
     MockGetTrackingInfo_Func.reset_mock()
     MockGetTrackingInfo_Func.invoke = MagicMock()
     MockGetOrderDetails_Func.reset_mock()
     MockGetOrderDetails_Func.invoke = MagicMock()
     MockInitiateReturn_Func.reset_mock()
     MockInitiateReturn_Func.invoke = MagicMock()
     MockKBLookup_Func.reset_mock()
     MockKBLookup_Func.invoke = MagicMock()

     # Reset Loader/Splitter/Path Mocks (Class-level and Instance)
     MockDirectoryLoader_Session.reset_mock()
     MockRecursiveCharacterTextSplitter_Session.reset_mock()
     MockPath_Session.reset_mock()
     mock_loader_instance.reset_mock()
     mock_loader_instance.load = MagicMock()
     mock_splitter_instance.reset_mock()
     mock_splitter_instance.split_documents = MagicMock()
     MockPath_Session.return_value = MockPathInstance # Re-assign instance
     MockPathInstance.reset_mock()
     MockPathInstance.exists = MagicMock(return_value=True) # Reset default behavior

     # Reset factory mocks
     MockGetRetriever.reset_mock()
     MockGetRetriever.return_value = mock_retriever_instance # Re-assign default

     # Reset Class-level mocks (usually just call counts)
     MockPGVector_Session.reset_mock()
     MockOpenAIEmbeddings_Session.reset_mock()
     MockHttpxClient_Session.reset_mock()
     MockChatOpenAI_Session.reset_mock()

# Ensure project root is in path for imports if running pytest from tests dir
# (Generally better to run pytest from root)
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# if project_root not in sys.path:
#      sys.path.insert(0, project_root)