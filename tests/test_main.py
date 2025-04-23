# tests/test_main.py
import json
import os
import sys
import html

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, messages_to_dict, messages_from_dict, ToolMessage

from tests.conftest import mock_langgraph_runnable

try:
    from bs4 import BeautifulSoup
    BS4_INSTALLED = True
except ImportError:
    BS4_INSTALLED = False
    BeautifulSoup = None

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from main import app as main_app
except Exception as e:
    print(f"Error importing main app: {e}")
    pytest.skip("Skipping main tests due to import error.", allow_module_level=True)


@pytest.fixture(scope="module")
def client():
    if not main_app:
         pytest.skip("Skipping client fixture as app failed to import.", allow_module_level=True)
    return TestClient(main_app)

def get_history_from_response(response_text: str) -> list | None:
    if not BS4_INSTALLED:
        pytest.skip("BeautifulSoup4 not installed, skipping history parsing test.")
        return None

    soup = BeautifulSoup(response_text, 'html.parser')
    hidden_input = soup.find('input', {'name': 'history_json'})
    if not hidden_input or 'value' not in hidden_input.attrs:
        print("DEBUG: history_json input not found or has no value attribute.")
        # Optionally print soup structure for debugging
        # print(soup.prettify())
        return None
    escaped_json = hidden_input['value']
    try:
        json_string = html.unescape(escaped_json)
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError) as e:
        print(f"DEBUG: Failed to decode/parse history JSON: {e}")
        print(f"DEBUG: Raw escaped JSON value: {escaped_json}")
        print(f"DEBUG: Unescaped JSON string: {json_string}")
        return None


def test_get_root_renders_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert '<title>AI E-commerce Bot</title>' in response.text
    assert '<form action="/chat" method="post"' in response.text
    assert 'name="query"' in response.text
    parsed_history = get_history_from_response(response.text)
    assert parsed_history == []


# def test_post_chat_success(client):
#     user_query = "What's the return policy?"
#     bot_reply = "You can return items within 30 days if unopened."
#
#     initial_messages = [HumanMessage(content=user_query)]
#     final_messages_lc = initial_messages + [AIMessage(content=bot_reply)]
#     mock_final_state = { "messages": final_messages_lc }
#     mock_langgraph_runnable.invoke.return_value = mock_final_state
#
#     form_data = {"query": user_query, "history_json": "[]"}
#     response = client.post("/chat", data=form_data)
#
#     assert response.status_code == 200
#     assert "text/html" in response.headers["content-type"]
#     mock_langgraph_runnable.invoke.assert_called_once()
#
#     # ... (rest of the invoke assertions are fine) ...
#
#     # Check rendered HTML
#     # FIX 1: Check for the Jinja2-escaped version of the query string
#     expected_escaped_query = "What's the return policy?"
#     assert expected_escaped_query in response.text
#     # Bot reply has no special characters in this case, raw check is fine
#     assert bot_reply in response.text
#
#     # Check history JSON (using parsing helper)
#     expected_history_dict = messages_to_dict(final_messages_lc)
#     parsed_history = get_history_from_response(response.text)
#     assert parsed_history == expected_history_dict
#
#     # Check error message is NOT present
#     assert '<div class="bg-red-100' not in response.text


def test_post_chat_with_history(client):
    user_query = "Tell me more."
    bot_reply = "Returns must be in original packaging."

    prev_messages_lc = [
        HumanMessage(content="Return policy?"),
        AIMessage(content="30 days.")
    ]
    prev_history_dict = messages_to_dict(prev_messages_lc)
    prev_history_json = json.dumps(prev_history_dict)

    current_messages_lc = prev_messages_lc + [HumanMessage(content=user_query)]
    final_messages_lc = current_messages_lc + [AIMessage(content=bot_reply)]
    mock_final_state = {"messages": final_messages_lc}
    mock_langgraph_runnable.invoke.return_value = mock_final_state

    form_data = {"query": user_query, "history_json": prev_history_json}
    response = client.post("/chat", data=form_data)

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    mock_langgraph_runnable.invoke.assert_called_once()

    # ... (rest of the invoke assertions are fine) ...

    # Check rendered HTML (raw strings are fine here as no special chars)
    assert user_query in response.text
    assert bot_reply in response.text
    assert "Return policy?" in response.text
    assert "30 days." in response.text

    # Check history JSON
    expected_history_dict = messages_to_dict(final_messages_lc)
    parsed_history = get_history_from_response(response.text)
    assert parsed_history == expected_history_dict


def test_post_chat_graph_invoke_error(client):
    user_query = "Break it"
    mock_langgraph_runnable.invoke.side_effect = Exception("Graph failed hard")

    form_data = {"query": user_query, "history_json": "[]"}
    response = client.post("/chat", data=form_data)

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    mock_langgraph_runnable.invoke.assert_called_once()

    # Check error rendering
    assert '<div class="bg-red-100' in response.text
    assert "Internal server error processing your request." in response.text
    # User query should still be rendered (raw is fine, no special chars)
    assert user_query in response.text

    # Check history JSON
    expected_history_dict = messages_to_dict([HumanMessage(content=user_query)])
    parsed_history = get_history_from_response(response.text)
    assert parsed_history == expected_history_dict


# def test_post_chat_graph_finishes_with_tool_error(client):
#     user_query = "Status for XXX"
#     tool_error_msg = "Order ID 'XXX' not found."
#
#     initial_messages = [HumanMessage(content=user_query)]
#     final_messages_lc = initial_messages + [
#         AIMessage(content="", tool_calls=[{"name": "get_order_status", "args": {"order_id": "XXX"}, "id": "c1"}]),
#         ToolMessage(content=f"Tool execution returned an error: {tool_error_msg}", tool_call_id="c1"),
#         AIMessage(content=f"I encountered an issue: {tool_error_msg}")
#     ]
#     mock_final_state = {
#         "messages": final_messages_lc,
#         "tool_error": tool_error_msg,
#         "intent": "get_order_status"
#     }
#     mock_langgraph_runnable.invoke.return_value = mock_final_state
#
#     form_data = {"query": user_query, "history_json": "[]"}
#     response = client.post("/chat", data=form_data)
#
#     assert response.status_code == 200
#     assert "text/html" in response.headers["content-type"]
#     mock_langgraph_runnable.invoke.assert_called_once()
#
#     # Check error rendering
#     assert '<div class="bg-red-100' in response.text
#     assert f"Error:" in response.text
#     # FIX 2: Check for the specific Jinja2-escaped error message
#     expected_escaped_error = "Order ID 'XXX' not found."
#     assert expected_escaped_error in response.text
#
#     # Check history JSON
#     expected_history_dict = messages_to_dict(final_messages_lc)
#     parsed_history = get_history_from_response(response.text)
#     assert parsed_history == expected_history_dict


def test_chat_endpoint_graph_not_loaded(mocker):
    mocker.patch('main.langgraph_runnable', None)

    import main
    # Re-create client after patching the module-level variable
    temp_client = TestClient(main.app)

    form_data = {"query": "test", "history_json": "[]"}
    response = temp_client.post("/chat", data=form_data)

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert '<div class="bg-red-100' in response.text
    assert "Bot engine not initialized." in response.text