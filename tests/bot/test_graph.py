import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Mock the LangGraph runnable used by the main app
mock_langgraph_runnable = MagicMock()

# Patch get_runnable BEFORE importing the app that calls it
with patch('main.get_runnable', return_value=mock_langgraph_runnable):
    from main import app as main_app

@pytest.fixture(scope="module")
def client():
    # You might need to set dummy env vars if settings loading relies on them
    # import os
    # os.environ['OPENAI_API_KEY'] = 'dummy'
    # ... other required env vars
    return TestClient(main_app)

@pytest.fixture(autouse=True)
def reset_mocks():
    mock_langgraph_runnable.reset_mock()
    mock_langgraph_runnable.invoke.side_effect = None


def test_health_check(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_chat_endpoint_success(client):
    user_query = "What is the status of ORD123?"
    expected_reply = "The status for order ORD123 is: Shipped."
    # Mock the final state returned by the LangGraph invoke call
    mock_final_state = {
        "messages": [
            HumanMessage(content=user_query),
            AIMessage(content=expected_reply) # The final reply from the bot
        ]
        # Include other state keys if needed for assertion, but messages is key
    }
    mock_langgraph_runnable.invoke.return_value = mock_final_state

    payload = {"query": user_query}
    response = client.post("/chat", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["reply"] == expected_reply
    assert "conversation_id" in data # Check conversation_id is returned

    # Verify LangGraph invoke was called correctly
    mock_langgraph_runnable.invoke.assert_called_once()
    call_args = mock_langgraph_runnable.invoke.call_args
    invoked_state = call_args[0][0] # First positional argument
    invoked_config = call_args[1] # Second positional argument (config)

    assert isinstance(invoked_state, dict)
    assert len(invoked_state['messages']) == 1
    assert isinstance(invoked_state['messages'][0], HumanMessage)
    assert invoked_state['messages'][0].content == user_query
    assert invoked_config == {} # Expecting stateless invocation config


def test_chat_endpoint_invoke_error(client):
    user_query = "Cause an error"
    # Mock the invoke call raising an exception
    mock_langgraph_runnable.invoke.side_effect = Exception("Graph execution failed")

    payload = {"query": user_query}
    response = client.post("/chat", json=payload)

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal Server Error: Failed to process the request."}
    mock_langgraph_runnable.invoke.assert_called_once()

def test_chat_endpoint_invoke_returns_no_aimessage(client):
    user_query = "No AI message"
    # Simulate graph finishing but last message isn't AI (unexpected state)
    mock_final_state = {
        "messages": [
            HumanMessage(content=user_query),
            ToolMessage(content="Tool ran", tool_call_id="t1") # Ends on ToolMessage
        ]
    }
    mock_langgraph_runnable.invoke.return_value = mock_final_state

    payload = {"query": user_query}
    response = client.post("/chat", json=payload)

    assert response.status_code == 200 # API handles it gracefully
    assert "encountered an issue generating a response" in response.json()["reply"]

@patch('main.langgraph_runnable', None) # Simulate graph failing to load at startup
def test_chat_endpoint_graph_not_loaded(client):
     # We need a new client instance that loads the app *after* the patch
     with patch('main.get_runnable', return_value=None):
          from main import app as patched_app
          temp_client = TestClient(patched_app)

     payload = {"query": "test"}
     response = temp_client.post("/chat", json=payload)
     assert response.status_code == 503
     assert response.json() == {"detail": "Service Unavailable: Bot engine not initialized."}