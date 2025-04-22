# tests/test_main.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import sys
import os

# NOTE: get_runnable is mocked globally by tests/conftest.py
# Import the specific mock instance if needed for configuration/assertions
from tests.conftest import mock_langgraph_runnable

# --- Import app AFTER patching (handled by conftest autouse fixture) ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from main import app as main_app
except Exception as e:
    print(f"Error importing main app: {e}")
    pytest.skip("Skipping main tests due to import error.", allow_module_level=True)


@pytest.fixture(scope="module")
def client():
    """Provides a TestClient instance for the main FastAPI app."""
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
    mock_final_state = { "messages": [ HumanMessage(content=user_query), AIMessage(content=expected_reply)] }
    mock_langgraph_runnable.invoke.return_value = mock_final_state

    payload = {"query": user_query}
    response = client.post("/chat", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["reply"] == expected_reply
    assert "conversation_id" in data
    mock_langgraph_runnable.invoke.assert_called_once()
    call_args, call_kwargs = mock_langgraph_runnable.invoke.call_args
    invoked_state = call_args[0]
    invoked_config = call_args[1]
    assert invoked_state['messages'][0].content == user_query
    assert invoked_config == {}

def test_chat_endpoint_invoke_error(client):
    """Tests the case where runnable.invoke *itself* raises an exception."""
    user_query = "Cause an error in invoke"
    mock_langgraph_runnable.invoke.side_effect = Exception("Graph invocation failed catastrophically")

    payload = {"query": user_query}
    response = client.post("/chat", json=payload)

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal Server Error: Failed to process the request."}
    mock_langgraph_runnable.invoke.assert_called_once()

def test_chat_endpoint_invoke_returns_no_aimessage(client):
    """Tests graph finishing with non-AIMessage as the last message."""
    user_query = "No AI message"
    mock_final_state = { "messages": [ HumanMessage(content=user_query), ToolMessage(content="Tool ran", tool_call_id="t1")]}
    mock_langgraph_runnable.invoke.return_value = mock_final_state

    payload = {"query": user_query}
    response = client.post("/chat", json=payload)

    assert response.status_code == 200
    assert response.json()["reply"] == "I'm sorry, I encountered an issue generating a response."

def test_chat_endpoint_invoke_returns_empty_state(client):
    """Tests the case where invoke returns None or an empty dict."""
    user_query = "Empty state"
    mock_langgraph_runnable.invoke.return_value = {}

    payload = {"query": user_query}
    response = client.post("/chat", json=payload)

    assert response.status_code == 200
    assert response.json()["reply"] == "I'm sorry, something went wrong while processing your request."

@patch('main.langgraph_runnable', None)
def test_chat_endpoint_graph_not_loaded():
     """Tests the scenario where the runnable couldn't be loaded at startup."""
     with patch('main.get_runnable', return_value=None):
          import main
          import importlib
          importlib.reload(main)
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
    mock_final_state = {"messages": [HumanMessage(content=user_query), AIMessage(content=expected_reply)]}
    mock_langgraph_runnable.invoke.return_value = mock_final_state

    payload = {"query": user_query, "conversation_id": conv_id}
    response = client.post("/chat", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["reply"] == expected_reply
    assert data["conversation_id"] == conv_id
    call_args, call_kwargs = mock_langgraph_runnable.invoke.call_args
    invoked_config = call_args[1]
    assert invoked_config == {}