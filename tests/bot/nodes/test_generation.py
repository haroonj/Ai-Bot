# tests/bot/nodes/test_generation.py
import os
import sys
import pytest
from unittest.mock import MagicMock
from langchain_core.messages import HumanMessage, AIMessage

# Ensure the app root is in the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Import the specific node function and state
from bot.nodes.generation import generate_final_response
from bot.state import GraphState

# Import mocks from conftest
from tests.conftest import mock_llm_instance

# REMOVE the problematic import:
# from .test_classification import create_full_initial_state # <--- REMOVE THIS LINE

# --- Test Cases ---

def test_generate_response_clarification(initial_state): # Use fixture
    question = "Which SKU do you want to return?"
    # Modify the state from the fixture
    initial_state.update({"needs_clarification": True, "clarification_question": question})

    update_dict = generate_final_response(initial_state)

    final_message = initial_state["messages"][-1]
    assert isinstance(final_message, AIMessage)
    assert final_message.content == question
    # ... rest of assertions

def test_generate_response_tool_error(initial_state): # Use fixture
    error_msg = "Order ID 'NF' not found."
    initial_state['tool_error'] = error_msg # Modify fixture state

    update_dict = generate_final_response(initial_state)

    final_message = initial_state["messages"][-1]
    assert isinstance(final_message, AIMessage)
    assert f"Sorry, I couldn't find the requested information. Details: {error_msg}" in final_message.content
    # ... rest of assertions

def test_generate_response_api_status_success(initial_state): # Use fixture
    initial_state['intent'] = 'get_order_status'
    initial_state['api_response'] = {"order_id": "ORD123", "status": "Shipped"}

    update_dict = generate_final_response(initial_state)

    final_message = initial_state["messages"][-1]
    assert "status for order ORD123 is: Shipped" in final_message.content
    # ... rest of assertions

def test_generate_response_api_tracking_success(initial_state): # Use fixture
    initial_state['intent'] = 'get_tracking_info'
    initial_state['api_response'] = {"order_id": "ORD456", "tracking_number": "TRACK1", "carrier": "UPS", "status": "In Transit"}

    update_dict = generate_final_response(initial_state)

    final_message = initial_state["messages"][-1]
    assert "Tracking for order ORD456" in final_message.content
    # ... rest of assertions

def test_generate_response_api_tracking_unavailable(initial_state): # Use fixture
    initial_state['intent'] = 'get_tracking_info'
    initial_state['api_response'] = {"order_id": "ORD789", "status": "Processing"} # No tracking number

    update_dict = generate_final_response(initial_state)

    final_message = initial_state["messages"][-1]
    assert "Tracking information for order ORD789" in final_message.content
    # ... rest of assertions


def test_generate_response_api_return_success(initial_state): # Use fixture
    initial_state['intent'] = 'initiate_return' # Intent might still be this after submit node
    initial_state['api_response'] = {"return_id": "RET123", "message": "Return approved."}

    update_dict = generate_final_response(initial_state)

    final_message = initial_state["messages"][-1]
    assert "Success! Return approved. Your return ID is RET123." in final_message.content
    # ... rest of assertions

def test_generate_response_direct_llm_response(initial_state): # Use fixture
    direct_content = "Please visit our website for more info."
    initial_state['final_llm_response'] = direct_content

    update_dict = generate_final_response(initial_state)

    final_message = initial_state["messages"][-1]
    assert final_message.content == direct_content
    # ... rest of assertions


def test_generate_response_rag_success(state_with_human_message): # Use factory fixture
    query = "Return policy?"
    initial_state = state_with_human_message(query) # Create state with query
    context = "Items can be returned within 30 days if unused."
    initial_state.update({"intent": "knowledge_base_query", "rag_context": context})

    mock_llm_response = AIMessage(content="Based on our knowledge base, you can return unused items within 30 days.")
    mock_llm_instance.invoke.return_value = mock_llm_response

    update_dict = generate_final_response(initial_state)

    mock_llm_instance.invoke.assert_called_once()
    prompt_arg = mock_llm_instance.invoke.call_args[0][0]
    assert context in prompt_arg
    assert query in prompt_arg

    final_message = initial_state["messages"][-1]
    assert final_message.content == mock_llm_response.content
    # ... rest of assertions


def test_generate_response_rag_no_context(state_with_human_message): # Use factory fixture
    initial_state = state_with_human_message("query") # Create state with query
    initial_state.update({"intent": "knowledge_base_query", "rag_context": None})

    update_dict = generate_final_response(initial_state)

    final_message = initial_state["messages"][-1]
    assert "couldn't find specific information about that topic" in final_message.content
    # ... rest of assertions


def test_generate_response_rag_llm_call_error(state_with_human_message, mocker): # Use factory fixture
    logger_error = mocker.patch('bot.nodes.generation.logger.error')
    query = "Return policy?"
    initial_state = state_with_human_message(query) # Create state with query
    context = "Items can be returned within 30 days if unused."
    initial_state.update({"intent": "knowledge_base_query", "rag_context": context})

    mock_llm_instance.invoke.side_effect = Exception("LLM RAG synthesis failed")

    update_dict = generate_final_response(initial_state)

    mock_llm_instance.invoke.assert_called_once()
    logger_error.assert_called_once()

    final_message = initial_state["messages"][-1]
    assert "had trouble formulating a final answer" in final_message.content
    # ... rest of assertions

def test_generate_response_greeting(initial_state): # Use fixture
    initial_state['intent'] = 'greeting'
    update_dict = generate_final_response(initial_state)
    final_message = initial_state["messages"][-1]
    assert "Hello! How can I assist you" in final_message.content

def test_generate_response_goodbye(initial_state): # Use fixture
    initial_state['intent'] = 'goodbye'
    update_dict = generate_final_response(initial_state)
    final_message = initial_state["messages"][-1]
    assert "Goodbye! Feel free to reach out" in final_message.content

def test_generate_response_unsupported(state_with_human_message): # Use factory fixture
    initial_state = state_with_human_message("gibberish") # Create state with query
    initial_state['intent'] = "unsupported"
    update_dict = generate_final_response(initial_state)
    final_message = initial_state["messages"][-1]
    assert "couldn't process that specific request" in final_message.content

def test_generate_response_empty_fallback(initial_state): # Use fixture
    initial_state['intent'] = "some_weird_intent"
    # No error, no api_response, no rag, no clarification...

    update_dict = generate_final_response(initial_state)

    final_message = initial_state["messages"][-1]
    # Check the final fallback message
    assert "I'm sorry, I couldn't process that specific request." in final_message.content or \
           "I'm sorry, I encountered an unexpected issue." in final_message.content