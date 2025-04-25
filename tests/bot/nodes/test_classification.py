# tests/bot/nodes/test_classification.py
import os
import sys
import pytest
from unittest.mock import MagicMock
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# Ensure the app root is in the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Import the specific node function and state
from bot.nodes.classification import classify_intent
from bot.state import GraphState

# Import only needed mocks from conftest
from tests.conftest import (
    # No need to import mock_llm_with_tools_instance, request as fixture
    MockGetOrderStatus_Func,
    MockGetTrackingInfo_Func,
    MockGetOrderDetails_Func,
    MockInitiateReturn_Func,
    MockKBLookup_Func
)

# Optional Helper (or use fixtures from conftest exclusively)
def create_full_initial_state(user_query: str | None = None, messages: list[BaseMessage] | None = None) -> GraphState:
    # (Definition as before)
    initial_messages = messages if messages is not None else []
    if user_query is not None and not any(isinstance(m, HumanMessage) and m.content == user_query for m in initial_messages):
         initial_messages.append(HumanMessage(content=user_query))
    return GraphState(
        messages=initial_messages, intent=None, order_id=None, item_sku_to_return=None, return_reason=None,
        needs_clarification=False, clarification_question=None, available_return_items=None,
        rag_context=None, api_response=None, tool_error=None, next_node=None, final_llm_response=None
    )


# --- Test Cases ---

def test_classify_intent_greeting(mock_llm_with_tools_instance): # Request fixture
    initial_state = create_full_initial_state("hello")
    result_state = classify_intent(initial_state)
    assert result_state['intent'] == "greeting"
    assert result_state['next_node'] == "generate_final_response"
    mock_llm_with_tools_instance.invoke.assert_not_called()

def test_classify_intent_goodbye(mock_llm_with_tools_instance): # Request fixture
    initial_state = create_full_initial_state("thanks bye")
    result_state = classify_intent(initial_state)
    assert result_state['intent'] == "goodbye"
    assert result_state['next_node'] == "generate_final_response"
    mock_llm_with_tools_instance.invoke.assert_not_called()

# Request mock fixture and set return_value INSIDE test
def test_classify_intent_tool_call_status(mock_llm_with_tools_instance):
    initial_state = create_full_initial_state("Status for ORD123?")
    mock_ai_response = AIMessage(
        content="",
        tool_calls=[{"name": MockGetOrderStatus_Func.name, "args": {"order_id": "ORD123"}, "id": "call_123"}]
    )
    # Set return value HERE
    mock_llm_with_tools_instance.invoke.return_value = mock_ai_response

    result_state = classify_intent(initial_state)

    # Assertion should pass
    mock_llm_with_tools_instance.invoke.assert_called_once_with(initial_state['messages'])
    assert result_state['intent'] == "get_order_status"
    assert result_state['order_id'] == "ORD123"
    assert result_state['next_node'] == "execute_tool_call"
    assert result_state.get('needs_clarification') is False
    assert len(initial_state['messages']) == 2
    assert initial_state['messages'][1] is mock_ai_response

def test_classify_intent_tool_call_tracking(mock_llm_with_tools_instance): # Request fixture
    initial_state = create_full_initial_state("Tracking for ORD123?")
    mock_ai_response = AIMessage(
        content="",
        tool_calls=[{"name": MockGetTrackingInfo_Func.name, "args": {"order_id": "ORD123"}, "id": "call_track"}]
    )
    mock_llm_with_tools_instance.invoke.return_value = mock_ai_response # Set HERE

    result_state = classify_intent(initial_state)

    mock_llm_with_tools_instance.invoke.assert_called_once_with(initial_state['messages'])
    assert result_state['intent'] == "get_tracking_info"
    assert result_state['order_id'] == "ORD123"
    assert result_state['next_node'] == "execute_tool_call"

def test_classify_intent_tool_call_kb(mock_llm_with_tools_instance): # Request fixture
    initial_state = create_full_initial_state("How do returns work?")
    mock_ai_response = AIMessage(
        content="",
        tool_calls=[{"name": MockKBLookup_Func.name, "args": {"query": "How do returns work?"}, "id": "call_kb"}]
    )
    mock_llm_with_tools_instance.invoke.return_value = mock_ai_response # Set HERE

    result_state = classify_intent(initial_state)

    mock_llm_with_tools_instance.invoke.assert_called_once_with(initial_state['messages'])
    assert result_state['intent'] == "knowledge_base_query"
    assert result_state['next_node'] == "execute_rag_lookup"

def test_classify_intent_tool_call_get_details_for_return(mock_llm_with_tools_instance): # Request fixture
    initial_state = create_full_initial_state("I want to return item from ORD789")
    mock_ai_response = AIMessage(
        content="",
        tool_calls=[{"name": MockGetOrderDetails_Func.name, "args": {"order_id": "ORD789"}, "id": "call_details"}]
    )
    mock_llm_with_tools_instance.invoke.return_value = mock_ai_response # Set HERE

    result_state = classify_intent(initial_state)

    mock_llm_with_tools_instance.invoke.assert_called_once_with(initial_state['messages'])
    assert result_state['intent'] == "initiate_return"
    assert result_state['order_id'] == "ORD789"
    assert result_state['next_node'] == "handle_return_step"

def test_classify_intent_tool_call_initiate_return_direct(mock_llm_with_tools_instance): # Request fixture
    initial_state = create_full_initial_state("Return SKU1 from ORD789 because broken")
    mock_ai_response = AIMessage(
        content="",
        tool_calls=[{"name": MockInitiateReturn_Func.name, "args": {"order_id": "ORD789", "sku": "SKU1", "reason": "broken"}, "id": "call_submit"}]
    )
    mock_llm_with_tools_instance.invoke.return_value = mock_ai_response # Set HERE

    result_state = classify_intent(initial_state)

    mock_llm_with_tools_instance.invoke.assert_called_once_with(initial_state['messages'])
    assert result_state['intent'] == "initiate_return"
    assert result_state['order_id'] == "ORD789"
    assert result_state['item_sku_to_return'] == "SKU1"
    assert result_state['return_reason'] == "broken"
    assert result_state['next_node'] == "handle_return_step"

def test_classify_intent_tool_call_unsupported_tool(mock_llm_with_tools_instance): # Request fixture
    initial_state = create_full_initial_state("Do my taxes")
    mock_ai_response = AIMessage(
        content="",
        tool_calls=[{"name": "do_taxes", "args": {}, "id": "call_tax"}]
    )
    mock_llm_with_tools_instance.invoke.return_value = mock_ai_response # Set HERE

    result_state = classify_intent(initial_state)

    mock_llm_with_tools_instance.invoke.assert_called_once_with(initial_state['messages'])
    # Assertion should pass now
    assert result_state['intent'] == "unsupported"
    assert "I understood you want to use 'do_taxes'" in result_state['api_response']['message']
    assert result_state['next_node'] == "generate_final_response"

# Request mock fixture and set return_value INSIDE test
def test_classify_intent_llm_provides_direct_content(mock_llm_with_tools_instance):
    initial_state = create_full_initial_state("Tell me about shipping")
    mock_ai_response = AIMessage(content="Standard shipping is 3-5 business days.")
    # Set return value HERE
    mock_llm_with_tools_instance.invoke.return_value = mock_ai_response

    result_state = classify_intent(initial_state)

    # Assertion should pass now
    mock_llm_with_tools_instance.invoke.assert_called_once_with(initial_state['messages'])
    assert result_state['intent'] == "knowledge_base_query"
    assert result_state['next_node'] == "generate_final_response"
    assert result_state.get('final_llm_response') == "Standard shipping is 3-5 business days."
    assert len(initial_state['messages']) == 2

def test_classify_intent_llm_no_tool_no_content(mock_llm_with_tools_instance): # Request fixture
    initial_state = create_full_initial_state("asdfghjkl")
    mock_ai_response = AIMessage(content="") # No tool calls, no content
    mock_llm_with_tools_instance.invoke.return_value = mock_ai_response # Set HERE

    result_state = classify_intent(initial_state)

    mock_llm_with_tools_instance.invoke.assert_called_once_with(initial_state['messages'])
    # Assertion should pass now
    assert result_state['intent'] == "unsupported"
    assert result_state['next_node'] == "generate_final_response"
    assert "I had trouble understanding" in result_state['api_response']['message']

def test_classify_intent_ongoing_return_flow(mock_llm_with_tools_instance): # Request fixture
    # Simulate state where bot asked for SKU
    messages = [
        HumanMessage(content="Return ORD789"),
        AIMessage(content="Okay, which SKU?"),
        HumanMessage(content="SKU123") # User's reply
    ]
    initial_state = create_full_initial_state(messages=messages)
    initial_state['intent'] = 'initiate_return'
    initial_state['needs_clarification'] = True
    initial_state['order_id'] = 'ORD789'
    initial_state['available_return_items'] = [{"sku": "SKU123", "name": "Test Item"}]

    result_state = classify_intent(initial_state)

    # Should bypass LLM call
    mock_llm_with_tools_instance.invoke.assert_not_called()
    assert result_state['next_node'] == "handle_return_step"
    # --- Corrected Assertion ---
    # Check that the intent key is NOT in the returned dictionary for this path
    assert 'intent' not in result_state
    # --- End Correction ---


def test_classify_intent_no_messages():
    initial_state = create_full_initial_state(messages=[])
    result_state = classify_intent(initial_state)
    assert result_state['intent'] == 'clarification_needed'
    assert result_state['next_node'] == 'generate_final_response'