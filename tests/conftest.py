from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Removed PGVector Mocks (assuming they were removed in previous FAISS step)
# mock_pgvector_instance = MagicMock(name="MockPGVectorInstance_Session")
# MockPGVector_Session = MagicMock(name="MockPGVector_Session", return_value=mock_pgvector_instance)

# FAISS / Retriever Mocks (Keep if using FAISS)
# It's better to mock the specific functions used (e.g., FAISS.load_local) if needed
mock_faiss_instance = MagicMock(name="MockFAISSInstance_Session")
MockFAISS_Session = MagicMock(name="MockFAISS_Session", return_value=mock_faiss_instance) # Mock the class constructor
MockFAISS_LoadLocal = MagicMock(name="MockFAISS_LoadLocal_Session", return_value=mock_faiss_instance) # Mock the load_local method

mock_retriever_instance = MagicMock(name="MockRetrieverInstance_Session")
MockGetRetriever = MagicMock(name="MockGetRetriever_Factory_Session", return_value=mock_retriever_instance)

# Embeddings Mocks
mock_embeddings_instance = MagicMock(name="MockEmbeddingsInstance_Session")
MockOpenAIEmbeddings_Session = MagicMock(name="MockOpenAIEmbeddings_Session", return_value=mock_embeddings_instance)

# Removed HTTPX Client Mocks
# mock_httpx_client_instance = MagicMock(name="MockHTTPXClientInstance_Session")
# MockHttpxClient_Session = MagicMock(name="MockHTTPXClient_Session", return_value=mock_httpx_client_instance)

# Mock Sample Data Functions
MockGetOrder_SampleData = MagicMock(name="MockGetOrder_SampleData_Session")
MockCreateReturn_SampleData = MagicMock(name="MockCreateReturn_SampleData_Session")

# LLM Mocks
mock_llm_instance = MagicMock(name="MockLLMInstance_Session")
MockChatOpenAI_Session = MagicMock(name="MockChatOpenAI_Session", return_value=mock_llm_instance)

# --- Tool Function Mocks (Assign .name attribute) ---
# These wrap the actual tool functions but are still useful for tracking calls
MockGetOrderStatus_Func = MagicMock(name="get_order_status_func")
MockGetOrderStatus_Func.name = "get_order_status"

MockGetTrackingInfo_Func = MagicMock(name="get_tracking_info_func")
MockGetTrackingInfo_Func.name = "get_tracking_info"

MockGetOrderDetails_Func = MagicMock(name="get_order_details_func")
MockGetOrderDetails_Func.name = "get_order_details"

MockInitiateReturn_Func = MagicMock(name="initiate_return_request_func")
MockInitiateReturn_Func.name = "initiate_return_request"

MockKBLookup_Func = MagicMock(name="knowledge_base_lookup_func")
MockKBLookup_Func.name = "knowledge_base_lookup"

# LangGraph Runnable Mock
mock_langgraph_runnable = MagicMock(name="MockLangGraphRunnable_Session")

# Loader/Splitter/Path Mocks
MockDirectoryLoader_Session = MagicMock(name="MockDirectoryLoader_Session")
mock_loader_instance = MockDirectoryLoader_Session.return_value
MockRecursiveCharacterTextSplitter_Session = MagicMock(name="MockSplitter_Session")
mock_splitter_instance = MockRecursiveCharacterTextSplitter_Session.return_value
MockPathInstance = MagicMock(spec=Path)
MockPath_Session = MagicMock(name="MockPath_Session", return_value=MockPathInstance)


@pytest.fixture(scope='session', autouse=True)
def patch_external_dependencies(session_mocker):
    # LLM and Embeddings
    session_mocker.patch('langchain_openai.ChatOpenAI', MockChatOpenAI_Session)
    session_mocker.patch('langchain_openai.OpenAIEmbeddings', MockOpenAIEmbeddings_Session)

    # Vector Store (FAISS) - Mock load_local and the class if needed elsewhere
    session_mocker.patch('langchain_community.vectorstores.FAISS', MockFAISS_Session)
    session_mocker.patch('langchain_community.vectorstores.FAISS.load_local', MockFAISS_LoadLocal)

    # Loaders and Splitters
    session_mocker.patch('langchain_community.document_loaders.directory.Path', MockPath_Session, create=True)
    session_mocker.patch('langchain_community.document_loaders.DirectoryLoader', MockDirectoryLoader_Session)
    session_mocker.patch('langchain.text_splitter.RecursiveCharacterTextSplitter', MockRecursiveCharacterTextSplitter_Session)

    # Patch internal modules/instances
    session_mocker.patch('bot.llm.llm', mock_llm_instance)
    session_mocker.patch('bot.llm.embeddings', mock_embeddings_instance)
    # Removed httpx client patching
    # session_mocker.patch('bot.tools.client', mock_httpx_client_instance)
    session_mocker.patch('bot.tools.llm_with_tools', mock_llm_instance)
    session_mocker.patch('bot.vector_store.vector_store', mock_faiss_instance) # Patch the global instance

    # Patch retriever factory function
    session_mocker.patch('bot.vector_store.get_retriever', MockGetRetriever)
    session_mocker.patch('bot.tools.get_retriever', MockGetRetriever)

    # Patch sample data functions used by tools
    session_mocker.patch('bot.tools.get_order', MockGetOrder_SampleData)
    session_mocker.patch('bot.tools.create_return', MockCreateReturn_SampleData)

    # Patch LangGraph runnable
    session_mocker.patch('main.langgraph_runnable', mock_langgraph_runnable)
    session_mocker.patch('bot.graph.app', mock_langgraph_runnable)

    # Patch tool functions themselves (optional, but useful for tracking)
    session_mocker.patch('bot.tools.get_order_status', MockGetOrderStatus_Func)
    session_mocker.patch('bot.tools.get_tracking_info', MockGetTrackingInfo_Func)
    session_mocker.patch('bot.tools.get_order_details', MockGetOrderDetails_Func)
    session_mocker.patch('bot.tools.initiate_return_request', MockInitiateReturn_Func)
    session_mocker.patch('bot.tools.knowledge_base_lookup', MockKBLookup_Func)

    # Patch the list of tools used by nodes/LLM binding
    mocked_tool_list = [
        MockGetOrderStatus_Func, MockGetTrackingInfo_Func, MockGetOrderDetails_Func,
        MockInitiateReturn_Func, MockKBLookup_Func
    ]
    session_mocker.patch('bot.nodes.available_tools', mocked_tool_list)
    session_mocker.patch('bot.tools.available_tools', mocked_tool_list)
    yield


@pytest.fixture(autouse=True)
def reset_session_mocks():
    # Reset FAISS mocks
    MockFAISS_Session.reset_mock()
    MockFAISS_LoadLocal.reset_mock()
    mock_faiss_instance.reset_mock()
    mock_faiss_instance.as_retriever = MagicMock(return_value=mock_retriever_instance)
    # Add other FAISS methods if needed, e.g., save_local, from_documents
    mock_faiss_instance.save_local = MagicMock()
    mock_faiss_instance.from_documents = MagicMock(return_value=mock_faiss_instance)


    # Reset Retriever mocks
    MockGetRetriever.reset_mock()
    MockGetRetriever.return_value = mock_retriever_instance
    mock_retriever_instance.reset_mock()
    mock_retriever_instance.invoke = MagicMock()

    # Reset Embeddings mocks
    mock_embeddings_instance.reset_mock()
    mock_embeddings_instance.embed_query = MagicMock(return_value=[0.1] * 1536)
    mock_embeddings_instance.embed_documents = MagicMock(return_value=[[0.1] * 1536])

    # Reset Sample Data mocks
    MockGetOrder_SampleData.reset_mock()
    MockGetOrder_SampleData.return_value = None # Default to not found
    MockCreateReturn_SampleData.reset_mock()
    MockCreateReturn_SampleData.return_value = (None, "Mock Error") # Default to error

    # Reset LLM mocks
    mock_llm_instance.reset_mock()
    mock_llm_instance.invoke = MagicMock()
    mock_llm_instance.bind_tools = MagicMock(return_value=mock_llm_instance)

    # Reset LangGraph mocks
    mock_langgraph_runnable.reset_mock()
    mock_langgraph_runnable.invoke = MagicMock()
    mock_langgraph_runnable.side_effect = None

    # Reset Tool function mocks
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

    # Reset Loader/Splitter/Path mocks
    MockDirectoryLoader_Session.reset_mock()
    MockRecursiveCharacterTextSplitter_Session.reset_mock()
    MockPath_Session.reset_mock()
    mock_loader_instance.reset_mock()
    mock_loader_instance.load = MagicMock()
    mock_splitter_instance.reset_mock()
    mock_splitter_instance.split_documents = MagicMock()
    MockPath_Session.return_value = MockPathInstance
    MockPathInstance.reset_mock()
    MockPathInstance.exists = MagicMock(return_value=True)
    MockPathInstance.is_dir = MagicMock(return_value=True) # Add is_dir mock

    # Reset Class constructor mocks
    MockOpenAIEmbeddings_Session.reset_mock()
    # MockHttpxClient_Session.reset_mock() # Removed
    MockChatOpenAI_Session.reset_mock()