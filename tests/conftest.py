# tests/conftest.py
from pathlib import Path
from unittest.mock import MagicMock
from typing import List, Optional

import pytest
from langchain_core.messages import BaseMessage, HumanMessage

# Import GraphState here if it's needed for the fixture type hint
try:
    from bot.state import GraphState
except ImportError:
    # Define a dummy type if bot.state isn't easily importable here
    GraphState = dict

# --- PGVector Mocks ---
mock_pgvector_instance = MagicMock(name="MockPGVectorInstance_Session")
MockPGVector_Session = MagicMock(name="MockPGVector_Session", return_value=mock_pgvector_instance)
mock_retriever_instance = MagicMock(name="MockRetrieverInstance_Session")
MockGetRetriever = MagicMock(name="MockGetRetriever_Factory_Session", return_value=mock_retriever_instance)

# --- Embeddings Mocks ---
mock_embeddings_instance = MagicMock(name="MockEmbeddingsInstance_Session")
MockOpenAIEmbeddings_Session = MagicMock(name="MockOpenAIEmbeddings_Session", return_value=mock_embeddings_instance)

# --- HTTPX Client Mocks ---
mock_httpx_client_instance = MagicMock(name="MockHTTPXClientInstance_Session")
MockHttpxClient_Session = MagicMock(name="MockHTTPXClient_Session", return_value=mock_httpx_client_instance)

# --- LLM Mocks (Revised for better test isolation) ---
@pytest.fixture(scope='session')
def _mock_llm_instance_session() -> MagicMock:
    """Internal fixture to create the LLM mock once per session."""
    instance = MagicMock(name="MockLLMInstance_Session")
    # Add a base invoke mock here if needed, but reset_mocks will overwrite it
    instance.invoke = MagicMock()
    instance.bind_tools = MagicMock(return_value=instance) # bind_tools returns self
    return instance

@pytest.fixture
def mock_llm_instance(_mock_llm_instance_session: MagicMock) -> MagicMock:
    """Provides the session-scoped LLM mock instance, reset for each test."""
    # Reset is handled by reset_mocks_before_each_test fixture
    return _mock_llm_instance_session

@pytest.fixture
def mock_llm_with_tools_instance(mock_llm_instance: MagicMock) -> MagicMock:
    """Provides the mock instance representing the LLM with tools bound."""
    # It's the same underlying mock object
    return mock_llm_instance
# --- End LLM Mocks ---

# --- Tool Function Mocks ---
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

# --- LangGraph Runnable Mock ---
mock_langgraph_runnable = MagicMock(name="MockLangGraphRunnable_Session")

# --- Loader/Splitter/Path Mocks ---
MockDirectoryLoader_Session = MagicMock(name="MockDirectoryLoader_Session")
mock_loader_instance = MockDirectoryLoader_Session.return_value
MockRecursiveCharacterTextSplitter_Session = MagicMock(name="MockSplitter_Session")
mock_splitter_instance = MockRecursiveCharacterTextSplitter_Session.return_value
MockPathInstance = MagicMock(spec=Path)
MockPath_Session = MagicMock(name="MockPath_Session", return_value=MockPathInstance)

# --- Autouse Fixtures ---
@pytest.fixture(scope='session', autouse=True)
def patch_external_dependencies(session_mocker, _mock_llm_instance_session: MagicMock): # Use internal session mock
    """Mocks external libraries and key module-level objects for the test session."""
    mock_llm_for_patching = _mock_llm_instance_session

    # Mock LLM/Embedding classes
    session_mocker.patch('langchain_openai.ChatOpenAI', MagicMock(return_value=mock_llm_for_patching))
    session_mocker.patch('langchain_openai.OpenAIEmbeddings', MockOpenAIEmbeddings_Session)

    # Mock Vector Store / Loaders
    session_mocker.patch('langchain_community.vectorstores.pgvector.PGVector', MockPGVector_Session)
    session_mocker.patch('langchain_community.document_loaders.directory.Path', MockPath_Session, create=True)
    session_mocker.patch('langchain_community.document_loaders.DirectoryLoader', MockDirectoryLoader_Session)
    session_mocker.patch('langchain.text_splitter.RecursiveCharacterTextSplitter', MockRecursiveCharacterTextSplitter_Session)

    # Mock HTTP Client
    session_mocker.patch('httpx.Client', MockHttpxClient_Session)

    # Patch module-level instances in the bot code
    session_mocker.patch('bot.llm.llm', mock_llm_for_patching)
    session_mocker.patch('bot.llm.embeddings', mock_embeddings_instance)
    session_mocker.patch('bot.tools.client', mock_httpx_client_instance)
    session_mocker.patch('bot.tools.llm_with_tools', mock_llm_for_patching) # Patches the instance used in classification
    session_mocker.patch('bot.vector_store.vector_store', mock_pgvector_instance)

    # Patch retriever factories
    session_mocker.patch('bot.vector_store.get_retriever', MockGetRetriever)
    session_mocker.patch('bot.tools.get_retriever', MockGetRetriever)

    # Patch graph runnable used by main.py and graph.py
    session_mocker.patch('main.langgraph_runnable', mock_langgraph_runnable)
    session_mocker.patch('bot.graph.app', mock_langgraph_runnable)

    # Patch the actual tool functions where they are defined/used
    session_mocker.patch('bot.tools.get_order_status', MockGetOrderStatus_Func)
    session_mocker.patch('bot.tools.get_tracking_info', MockGetTrackingInfo_Func)
    session_mocker.patch('bot.tools.get_order_details', MockGetOrderDetails_Func)
    session_mocker.patch('bot.tools.initiate_return_request', MockInitiateReturn_Func)
    session_mocker.patch('bot.tools.knowledge_base_lookup', MockKBLookup_Func)

    # Patch the list of tools where llm.bind_tools happens
    mocked_tool_list = [
        MockGetOrderStatus_Func, MockGetTrackingInfo_Func, MockGetOrderDetails_Func,
        MockInitiateReturn_Func, MockKBLookup_Func
    ]
    session_mocker.patch('bot.tools.available_tools', mocked_tool_list)

    yield # Tests run here


@pytest.fixture(autouse=True)
def reset_mocks_before_each_test(
    # Request the function-scoped fixtures derived from the session mock
    mock_llm_instance: MagicMock,
    mock_llm_with_tools_instance: MagicMock
):
    """Resets mock call counts and configurations before each test function."""
    # --- Reset Core Instances ---
    mock_pgvector_instance.reset_mock()
    mock_pgvector_instance.as_retriever = MagicMock(return_value=mock_retriever_instance)
    mock_pgvector_instance.add_documents = MagicMock()

    mock_embeddings_instance.reset_mock()
    mock_embeddings_instance.embed_query = MagicMock(return_value=[0.1] * 1536)
    mock_embeddings_instance.embed_documents = MagicMock(return_value=[[0.1] * 1536])

    mock_httpx_client_instance.reset_mock()
    mock_httpx_client_instance.get = MagicMock()
    mock_httpx_client_instance.post = MagicMock()

    # --- Reset LLM Mocks (Crucial Part) ---
    # Reset the main mock object's call history etc.
    mock_llm_instance.reset_mock()
    # Assign a *new* MagicMock to the invoke attribute for this specific test
    # This prevents leftover return_values/side_effects from interfering
    mock_llm_instance.invoke = MagicMock()
    # Ensure bind_tools still returns the (same) mock instance
    mock_llm_instance.bind_tools = MagicMock(return_value=mock_llm_with_tools_instance)

    # Also reset the llm_with_tools_instance mock if needed (often same as mock_llm_instance)
    mock_llm_with_tools_instance.reset_mock()
    mock_llm_with_tools_instance.invoke = MagicMock() # Give it a fresh invoke too
    # --- End LLM Reset ---

    mock_langgraph_runnable.reset_mock()
    mock_langgraph_runnable.invoke = MagicMock()
    mock_langgraph_runnable.side_effect = None # Clear any exceptions

    # Reset Retriever and Factory
    mock_retriever_instance.reset_mock()
    mock_retriever_instance.invoke = MagicMock()
    MockGetRetriever.reset_mock()
    MockGetRetriever.return_value = mock_retriever_instance

    # Reset Tool Function Mocks (invoke attribute)
    MockGetOrderStatus_Func.reset_mock(); MockGetOrderStatus_Func.invoke = MagicMock()
    MockGetTrackingInfo_Func.reset_mock(); MockGetTrackingInfo_Func.invoke = MagicMock()
    MockGetOrderDetails_Func.reset_mock(); MockGetOrderDetails_Func.invoke = MagicMock()
    MockInitiateReturn_Func.reset_mock(); MockInitiateReturn_Func.invoke = MagicMock()
    MockKBLookup_Func.reset_mock(); MockKBLookup_Func.invoke = MagicMock()

    # Reset Loader/Splitter Mocks
    MockDirectoryLoader_Session.reset_mock(); mock_loader_instance.reset_mock(); mock_loader_instance.load = MagicMock()
    MockRecursiveCharacterTextSplitter_Session.reset_mock(); mock_splitter_instance.reset_mock(); mock_splitter_instance.split_documents = MagicMock()
    MockPath_Session.reset_mock(); MockPathInstance.reset_mock(); MockPathInstance.exists = MagicMock(return_value=True)

    # Reset Session Mocks (Factories) - Generally less critical unless testing instantiation
    # MockPGVector_Session.reset_mock()
    # MockOpenAIEmbeddings_Session.reset_mock()
    # MockHttpxClient_Session.reset_mock()
    # MockChatOpenAI_Session.reset_mock()

# --- State Helper Fixtures ---
@pytest.fixture
def initial_state() -> GraphState:
    """Provides a clean, basic GraphState dictionary for tests."""
    from bot.state import GraphState
    return GraphState(
        messages=[],
        intent=None, order_id=None, item_sku_to_return=None, return_reason=None,
        needs_clarification=False, clarification_question=None, available_return_items=None,
        rag_context=None, api_response=None, tool_error=None, next_node=None,
        latest_ai_response=None,
        final_llm_response=None
    )

@pytest.fixture
def state_with_human_message(initial_state: GraphState) -> callable:
    """Factory fixture to create a state with a specific human message."""
    def _create_state(content: str, existing_messages: Optional[List[BaseMessage]] = None) -> GraphState:
        state = initial_state.copy() # Start fresh
        if existing_messages:
            state['messages'] = existing_messages[:] # Copy existing
        state['messages'].append(HumanMessage(content=content))
        return state
    return _create_state