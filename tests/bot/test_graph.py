import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from unittest.mock import patch
from main import app

client = TestClient(app)

@pytest.fixture
def mock_langgraph():
    with patch("main.langgraph_runnable") as mock:
        yield mock

def test_chat_endpoint_success(mock_langgraph):
    user_query = "What is the status of ORD123?"
    expected_reply = "The status for order ORD123 is: Shipped."

    mock_langgraph.invoke.return_value = {
        "messages": [
            HumanMessage(content=user_query),
            AIMessage(content=expected_reply)
        ]
    }

    response = client.post("/chat", json={"query": user_query})
    assert response.status_code == 200
    assert response.json()["reply"] == expected_reply


def test_chat_endpoint_invoke_error(mock_langgraph):
    user_query = "Cause an error"
    mock_langgraph.invoke.side_effect = Exception("Graph execution failed")

    response = client.post("/chat", json={"query": user_query})
    assert response.status_code == 500


def test_chat_endpoint_invoke_returns_no_aimessage(mock_langgraph):
    user_query = "No AI message"
    mock_langgraph.invoke.return_value = {
        "messages": [
            HumanMessage(content=user_query),
            ToolMessage(content="Tool ran", tool_call_id="t1")
        ]
    }

    response = client.post("/chat", json={"query": user_query})
    assert response.status_code == 200
    assert "issue" in response.json()["reply"]
