import html
import json
import os
import sys

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage, AIMessage, messages_to_dict

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
        return None
    escaped_json = hidden_input['value']
    try:
        json_string = html.unescape(escaped_json)
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError) as e:
        print(f"DEBUG: Failed to decode/parse history JSON: {e}")
        print(f"DEBUG: Raw escaped JSON value: {escaped_json}")
        if 'json_string' in locals():
            print(f"DEBUG: Unescaped JSON string: {json_string}")
        return None


def test_get_root_renders_html(client):
    if not BS4_INSTALLED:
        pytest.skip("BeautifulSoup4 not installed, skipping HTML parsing.")

    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

    soup = BeautifulSoup(response.text, 'html.parser')

    title_tag = soup.find('title')
    assert title_tag is not None
    assert title_tag.string == 'AI E-commerce Bot'

    form_tag = soup.find('form')
    assert form_tag is not None, "Form tag not found in HTML"
    assert form_tag.get('action') == '/chat', "Form action attribute is incorrect"
    assert form_tag.get('method', '').lower() == 'post', "Form method attribute is incorrect or missing"

    assert soup.find('input', {'name': 'query'}) is not None, "Query input not found"
    assert soup.find('input', {'name': 'history_json', 'type': 'hidden'}) is not None, "Hidden history input not found"
    parsed_history = get_history_from_response(response.text)
    assert parsed_history == []


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

    call_object = mock_langgraph_runnable.invoke.call_args
    invoked_state = call_object.args[0]
    assert len(invoked_state['messages']) == 3
    assert invoked_state['messages'][0].content == "Return policy?"
    assert invoked_state['messages'][1].content == "30 days."
    assert invoked_state['messages'][2].content == user_query

    assert user_query in response.text
    assert bot_reply in response.text
    assert "Return policy?" in response.text
    assert "30 days." in response.text

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

    assert '<div class="bg-red-100' in response.text
    assert "Internal server error processing your request." in response.text
    assert user_query in response.text

    expected_history_dict = messages_to_dict([HumanMessage(content=user_query)])
    parsed_history = get_history_from_response(response.text)
    assert parsed_history == expected_history_dict


def test_chat_endpoint_graph_not_loaded(mocker):
    mocker.patch('main.langgraph_runnable', None)

    import main
    temp_client = TestClient(main.app)

    form_data = {"query": "test", "history_json": "[]"}
    response = temp_client.post("/chat", data=form_data)

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert '<div class="bg-red-100' in response.text
    assert "Bot engine not initialized." in response.text
