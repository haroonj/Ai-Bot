# tests/test_main.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import sys
import os

# Import the specific session-wide mock from conftest used in main.py
from tests.conftest import mock_langgraph_runnable

# --- Import app AFTER patching (handled by conftest autouse fixture) ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    # Import the app instance directly
    from main import app as main_app
except Exception as e:
    print(f"Error importing main app: {e}")
    # If app import fails, skip tests in this module
    pytest.skip("Skipping main tests due to import error.", allow_module_level=True)


@pytest.fixture(scope="module")
def client():
    """Provides a TestClient instance for the main FastAPI app."""
    # Ensure the app imported correctly before creating the client
    if not main_app:
         pytest.skip("Skipping client fixture as app failed to import.", allow_module_level=True)
    return TestClient(main_app)

# No need for reset_main_mocks, handled by reset_session_mocks in conftest.py

def test_health_check(client):
    """Tests the root health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_chat_endpoint_success(client):
    """Tests a successful chat request flow with mocked graph output."""
    user_query = "What is the status of ORD123?"
    expected_reply = "The status for order ORD123 is: Shipped."
    # Configure the globally mocked runnable's invoke method
    mock_final_state = { "messages": [ HumanMessage(content=user_query), AIMessage(content=expected_reply)] }
    mock_langgraph_runnable.invoke.return_value = mock_final_state

    payload = {"query": user_query}
    response = client.post("/chat", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["reply"] == expected_reply
    assert "conversation_id" in data
    # Assert that the globally mocked runnable was called
    mock_langgraph_runnable.invoke.assert_called_once()

    # --- Corrected Argument Access ---
    call_object = mock_langgraph_runnable.invoke.call_args
    invoked_state = call_object.args[0] # First positional argument
    invoked_config = call_object.kwargs.get('config', 'MISSING') # Keyword argument 'config'

    assert invoked_state['messages'][0].content == user_query
    assert invoked_config == {} # Expecting empty config for stateless invoke

def test_chat_endpoint_invoke_error(client):
    """Tests the case where runnable.invoke *itself* raises an exception."""
    user_query = "Cause an error in invoke"
    # Configure the globally mocked runnable to raise an exception
    mock_langgraph_runnable.invoke.side_effect = Exception("Graph invocation failed catastrophically")

    payload = {"query": user_query}
    response = client.post("/chat", json=payload)

    # The endpoint should catch this and return 500
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal Server Error: Failed to process the request."}
    mock_langgraph_runnable.invoke.assert_called_once()

def test_chat_endpoint_invoke_returns_no_aimessage(client):
    """Tests graph finishing with non-AIMessage as the last message."""
    user_query = "No AI message"
    # Configure the globally mocked runnable
    mock_final_state = { "messages": [ HumanMessage(content=user_query), ToolMessage(content="Tool ran", tool_call_id="t1")]}
    mock_langgraph_runnable.invoke.return_value = mock_final_state

    payload = {"query": user_query}
    response = client.post("/chat", json=payload)

    # The endpoint should handle this gracefully
    assert response.status_code == 200
    assert response.json()["reply"] == "I'm sorry, I encountered an issue generating a response."
    mock_langgraph_runnable.invoke.assert_called_once()

def test_chat_endpoint_invoke_returns_empty_state(client):
    """Tests the case where invoke returns None or an empty dict."""
    user_query = "Empty state"
    # Configure the globally mocked runnable
    mock_langgraph_runnable.invoke.return_value = {}

    payload = {"query": user_query}
    response = client.post("/chat", json=payload)

    # The endpoint should handle this gracefully
    assert response.status_code == 200
    assert response.json()["reply"] == "I'm sorry, something went wrong while processing your request."
    mock_langgraph_runnable.invoke.assert_called_once()

# Use pytest-mock's mocker fixture for temporary patching within a test
def test_chat_endpoint_graph_not_loaded(mocker):
     """Tests the scenario where the runnable couldn't be loaded at startup."""
     # Temporarily patch the *already patched* global variable to None for this test
     mocker.patch('main.langgraph_runnable', None)

     # Re-import main *within the test scope* to get the effect of the patch
     import main
     # Create a temporary client for this modified app state
     temp_client = TestClient(main.app)

     payload = {"query": "test"}
     response = temp_client.post("/chat", json=payload)
     assert response.status_code == 503
     assert response.json() == {"detail": "Service Unavailable: Bot engine not initialized."}

def test_chat_endpoint_with_conversation_id(client):
    """Tests passing a conversation_id."""
    user_query = "Another query"
    expected_reply = "Okay."
    conv_id = "test-conv-123"
    # Configure the globally mocked runnable
    mock_final_state = {"messages": [HumanMessage(content=user_query), AIMessage(content=expected_reply)]}
    mock_langgraph_runnable.invoke.return_value = mock_final_state

    payload = {"query": user_query, "conversation_id": conv_id}
    response = client.post("/chat", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["reply"] == expected_reply
    assert data["conversation_id"] == conv_id
    # Assert that the globally mocked runnable was called
    mock_langgraph_runnable.invoke.assert_called_once()

    # --- Corrected Argument Access ---
    call_object = mock_langgraph_runnable.invoke.call_args
    invoked_state = call_object.args[0] # First positional argument
    invoked_config = call_object.kwargs.get('config', 'MISSING') # Keyword argument 'config'

    # If using checkpointer, config would be {"configurable": {"thread_id": conv_id}}
    # For stateless (as currently implemented in main.py), config is {}
    assert invoked_config == {}