# tests/conftest.py
import pytest
from unittest.mock import MagicMock, patch
import sys
import os
from pathlib import Path # Import for spec

# --- Define ALL Mock Objects/Classes Needed by Tests ---
mock_pgvector_instance = MagicMock(name="MockPGVectorInstance_Session")
MockPGVector_Session = MagicMock(name="MockPGVector_Session", return_value=mock_pgvector_instance)
mock_retriever_instance = MagicMock(name="MockRetrieverInstance_Session")
MockGetRetriever = MagicMock(name="MockGetRetriever_Factory_Session", return_value=mock_retriever_instance)

mock_embeddings_instance = MagicMock(name="MockEmbeddingsInstance_Session")
MockOpenAIEmbeddings_Session = MagicMock(name="MockOpenAIEmbeddings_Session", return_value=mock_embeddings_instance)

mock_httpx_client_instance = MagicMock(name="MockHTTPXClientInstance_Session")
MockHttpxClient_Session = MagicMock(name="MockHTTPXClient_Session", return_value=mock_httpx_client_instance)

mock_llm_instance = MagicMock(name="MockLLMInstance_Session")
MockChatOpenAI_Session = MagicMock(name="MockChatOpenAI_Session", return_value=mock_llm_instance)

MockGetOrderStatus = MagicMock(name="get_order_status")
MockGetTrackingInfo = MagicMock(name="get_tracking_info")
MockGetOrderDetails = MagicMock(name="get_order_details")
MockInitiateReturn = MagicMock(name="initiate_return_request")
MockKBLookup = MagicMock(name="knowledge_base_lookup")

mock_langgraph_runnable = MagicMock(name="MockLangGraphRunnable_Session")

MockDirectoryLoader = MagicMock(name="MockDirectoryLoader_Session")
mock_loader_instance = MockDirectoryLoader.return_value
MockRecursiveCharacterTextSplitter = MagicMock(name="MockSplitter_Session")
mock_splitter_instance = MockRecursiveCharacterTextSplitter.return_value
MockPathInstance = MagicMock(spec=Path)
MockPath = MagicMock(name="MockPath_Session", return_value=MockPathInstance)

@pytest.fixture(scope='session', autouse=True)
def patch_external_dependencies(session_mocker):
    """Apply session-wide patches BEFORE any module imports them."""
    # print("Applying session-wide patches...")

    # Patch Classes (These MUST run before the modules using them are imported)
    session_mocker.patch('langchain_openai.ChatOpenAI', MockChatOpenAI_Session)
    session_mocker.patch('langchain_openai.OpenAIEmbeddings', MockOpenAIEmbeddings_Session)
    session_mocker.patch('langchain_community.vectorstores.pgvector.PGVector', MockPGVector_Session)
    session_mocker.patch('httpx.Client', MockHttpxClient_Session)
    session_mocker.patch('langchain_community.document_loaders.directory.Path', MockPath, create=True)
    session_mocker.patch('langchain_community.document_loaders.DirectoryLoader', MockDirectoryLoader)
    session_mocker.patch('langchain.text_splitter.RecursiveCharacterTextSplitter', MockRecursiveCharacterTextSplitter)

    # Patch Instances/Functions *where they are used* (more robust)
    session_mocker.patch('bot.llm.llm', mock_llm_instance)
    session_mocker.patch('bot.llm.embeddings', mock_embeddings_instance)
    session_mocker.patch('bot.tools.client', mock_httpx_client_instance)
    session_mocker.patch('bot.tools.llm_with_tools', mock_llm_instance) # Use same mock for simplicity
    session_mocker.patch('bot.vector_store.vector_store', mock_pgvector_instance) # Patch the created instance
    session_mocker.patch('bot.vector_store.get_retriever', MockGetRetriever)
    session_mocker.patch('bot.tools.get_retriever', MockGetRetriever) # Patch where used in tools
    session_mocker.patch('main.get_runnable', return_value=mock_langgraph_runnable)
    session_mocker.patch('bot.graph.app', mock_langgraph_runnable) # Attempt patch here too

    # Patch tool functions where they are defined/imported if necessary
    # Note: This might conflict if bot.nodes also tries importing them before this runs
    # It's often better to mock them where they are *called* (e.g., inside execute_tool tests)
    # For now, keep global mocks for simplicity, ensure test_nodes patches override if needed.
    session_mocker.patch('bot.tools.get_order_status', MockGetOrderStatus)
    session_mocker.patch('bot.tools.get_tracking_info', MockGetTrackingInfo)
    session_mocker.patch('bot.tools.get_order_details', MockGetOrderDetails)
    session_mocker.patch('bot.tools.initiate_return_request', MockInitiateReturn)
    session_mocker.patch('bot.tools.knowledge_base_lookup', MockKBLookup)


    yield
    # print("Stopping session-wide patches...")

@pytest.fixture(autouse=True)
def reset_session_mocks():
     """Resets mocks before each test."""
     # print("Resetting mocks...") # Debug print

     # Reset Class Mocks (just call count, etc.)
     MockPGVector_Session.reset_mock()
     MockOpenAIEmbeddings_Session.reset_mock()
     MockHttpxClient_Session.reset_mock()
     MockChatOpenAI_Session.reset_mock()
     MockGetRetriever.reset_mock()
     MockDirectoryLoader.reset_mock()
     MockRecursiveCharacterTextSplitter.reset_mock()
     MockPath.reset_mock()

     # Reset Instance Mocks and ensure methods exist
     mock_pgvector_instance.reset_mock()
     mock_pgvector_instance.as_retriever = MagicMock(return_value=mock_retriever_instance)
     mock_pgvector_instance.add_documents = MagicMock()
     mock_pgvector_instance.embedding_function = mock_embeddings_instance # Re-assign if needed

     mock_embeddings_instance.reset_mock()
     mock_embeddings_instance.embed_query = MagicMock(return_value=[0.1] * 1536)
     mock_embeddings_instance.embed_documents = MagicMock(return_value=[[0.1] * 1536])

     mock_httpx_client_instance.reset_mock()
     mock_httpx_client_instance.get = MagicMock()
     mock_httpx_client_instance.post = MagicMock()

     mock_llm_instance.reset_mock()
     mock_llm_instance.invoke = MagicMock()
     mock_llm_instance.bind_tools = MagicMock(return_value=mock_llm_instance) # Mock bind_tools

     mock_langgraph_runnable.reset_mock()
     mock_langgraph_runnable.invoke = MagicMock()

     mock_retriever_instance.reset_mock()
     mock_retriever_instance.invoke = MagicMock()
     MockGetRetriever.return_value = mock_retriever_instance # Re-assign default

     # Reset Tool Function Mocks (ensure invoke exists)
     MockGetOrderStatus.reset_mock(); MockGetOrderStatus.invoke = MagicMock()
     MockGetTrackingInfo.reset_mock(); MockGetTrackingInfo.invoke = MagicMock()
     MockGetOrderDetails.reset_mock(); MockGetOrderDetails.invoke = MagicMock()
     MockInitiateReturn.reset_mock(); MockInitiateReturn.invoke = MagicMock()
     MockKBLookup.reset_mock(); MockKBLookup.invoke = MagicMock()

     # Reset Loader/Splitter/Path instances
     mock_loader_instance.reset_mock()
     mock_loader_instance.load = MagicMock()
     mock_splitter_instance.reset_mock()
     mock_splitter_instance.split_documents = MagicMock()
     MockPathInstance.reset_mock()
     MockPathInstance.exists = MagicMock(return_value=True) # Default path exists