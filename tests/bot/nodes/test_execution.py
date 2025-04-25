# tests/bot/nodes/test_execution.py
import os
import sys
import pytest
from unittest.mock import MagicMock
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Ensure the app root is in the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Import the specific node functions and state
from bot.nodes.execution import execute_tool_call, execute_rag_lookup
from bot.state import GraphState

# Import mocks from conftest
from tests.conftest import (
    # No need to import mock_httpx_client_instance directly if tools are mocked
    MockGetOrderStatus_Func,
    MockKBLookup_Func,
    MockGetRetriever,
    mock_retriever_instance
)

# --- Tests for execute_tool_call ---

def test_execute_tool_call_status_success(initial_state): # Use fixture
    tool_call_ai_msg = AIMessage(content="", tool_calls=[
        {"name": MockGetOrderStatus_Func.name, "args": {"order_id": "ORD123"}, "id": "call_1"}])
    # Set the temporary state field
    initial_state['latest_ai_response'] = tool_call_ai_msg
    # Add a human message for context if needed, but not strictly required for the node
    initial_state['messages'].append(HumanMessage(content="Status ORD123?"))

    mock_api_response = {"order_id": "ORD123", "status": "Delivered"}
    MockGetOrderStatus_Func.invoke.return_value = mock_api_response

    result_state = execute_tool_call(initial_state)

    MockGetOrderStatus_Func.invoke.assert_called_once_with({"order_id": "ORD123"})
    assert result_state['api_response'] == mock_api_response
    assert result_state.get('tool_error') is None
    assert result_state.get('rag_context') is None
    assert result_state.get('next_node') is None
    assert result_state.get('latest_ai_response') is None # Check it was cleared

    # Check ToolMessage was added to the actual message history
    # History now contains HumanMsg + ToolMsg
    assert len(initial_state['messages']) == 2
    assert isinstance(initial_state['messages'][1], ToolMessage)
    assert initial_state['messages'][1].tool_call_id == "call_1"
    assert "Successfully called API tool get_order_status" in initial_state['messages'][1].content


def test_execute_tool_call_status_api_error(initial_state): # Use fixture
    tool_call_ai_msg = AIMessage(content="",
                                  tool_calls=[{"name": MockGetOrderStatus_Func.name, "args": {"order_id": "NF"}, "id": "call_2"}])
    # Set the temporary state field
    initial_state['latest_ai_response'] = tool_call_ai_msg
    initial_state['messages'].append(HumanMessage(content="Status NF?"))

    mock_error_response = {"error": "Order ID 'NF' not found."}
    MockGetOrderStatus_Func.invoke.return_value = mock_error_response

    result_state = execute_tool_call(initial_state)

    MockGetOrderStatus_Func.invoke.assert_called_once_with({"order_id": "NF"})
    assert result_state['api_response'] == mock_error_response
    assert result_state['tool_error'] == "Order ID 'NF' not found."
    assert result_state.get('rag_context') is None
    assert result_state.get('latest_ai_response') is None # Cleared

    # Check ToolMessage added
    assert len(initial_state['messages']) == 2
    assert isinstance(initial_state['messages'][1], ToolMessage)
    assert "Tool execution returned an error" in initial_state['messages'][1].content


def test_execute_tool_call_tool_not_found(initial_state): # Use fixture
    tool_call_ai_msg = AIMessage(content="",
                                  tool_calls=[{"name": "non_existent_tool", "args": {}, "id": "call_bad"}])
    initial_state['latest_ai_response'] = tool_call_ai_msg
    initial_state['messages'].append(HumanMessage(content="Do something invalid"))

    result_state = execute_tool_call(initial_state)

    assert result_state.get('api_response') is None
    assert "Tool 'non_existent_tool' not found" in result_state['tool_error']
    assert result_state.get('rag_context') is None
    assert result_state.get('latest_ai_response') is None # Cleared

    # Check ToolMessage added
    assert len(initial_state['messages']) == 2
    assert isinstance(initial_state['messages'][1], ToolMessage)
    assert "Error: Tool 'non_existent_tool' is not available." in initial_state['messages'][1].content


def test_execute_tool_call_exception_in_tool(initial_state): # Use fixture
    tool_call_ai_msg = AIMessage(content="",
                                  tool_calls=[{"name": MockGetOrderStatus_Func.name, "args": {"order_id": "ERR"}, "id": "call_err"}])
    initial_state['latest_ai_response'] = tool_call_ai_msg
    initial_state['messages'].append(HumanMessage(content="Status ERR?"))

    MockGetOrderStatus_Func.invoke.side_effect = Exception("Database connection failed")

    result_state = execute_tool_call(initial_state)

    MockGetOrderStatus_Func.invoke.assert_called_once_with({"order_id": "ERR"})
    assert result_state.get('api_response') is None
    assert "Failed to execute tool 'get_order_status'" in result_state['tool_error']
    assert "Database connection failed" in result_state['tool_error']
    assert result_state.get('rag_context') is None
    assert result_state.get('latest_ai_response') is None # Cleared

    # Check ToolMessage added
    assert len(initial_state['messages']) == 2
    assert isinstance(initial_state['messages'][1], ToolMessage)
    assert "An unexpected error occurred" in initial_state['messages'][1].content


def test_execute_tool_call_no_tool_call_in_state(initial_state): # Use fixture
    # Simulate state where classify failed to set latest_ai_response
    initial_state['messages'].append(HumanMessage(content="Request status"))
    initial_state['latest_ai_response'] = None # Explicitly None or missing

    result_state = execute_tool_call(initial_state)

    assert "no valid tool call found" in result_state['tool_error']
    assert result_state.get('api_response') is None
    assert result_state.get('latest_ai_response') is None # Still should be None

    # No ToolMessage should be added
    assert len(initial_state['messages']) == 1


# --- Tests for execute_rag_lookup ---

def test_execute_rag_lookup_success_from_human_msg(state_with_human_message): # Request fixture
    initial_state = state_with_human_message("Returns policy?") # Use fixture
    mock_kb_context = "Return within 30 days."
    MockKBLookup_Func.invoke.return_value = mock_kb_context
    MockGetRetriever.return_value = mock_retriever_instance

    result_state = execute_rag_lookup(initial_state)

    MockKBLookup_Func.invoke.assert_called_once_with({"query": "Returns policy?"})
    assert result_state['rag_context'] == mock_kb_context
    assert result_state.get('tool_error') is None
    assert result_state.get('api_response') is None
    assert result_state['next_node'] == "generate_final_response"
    assert result_state.get('latest_ai_response') is None # Check cleared


def test_execute_rag_lookup_success_explicit_query(initial_state): # Request fixture
    explicit_query = "Shipping times"
    mock_kb_context = "3-5 days standard."
    MockKBLookup_Func.invoke.return_value = mock_kb_context
    MockGetRetriever.return_value = mock_retriever_instance

    result_state = execute_rag_lookup(initial_state, explicit_query=explicit_query)

    MockKBLookup_Func.invoke.assert_called_once_with({"query": explicit_query})
    assert result_state['rag_context'] == mock_kb_context
    assert result_state.get('tool_error') is None
    assert result_state.get('latest_ai_response') is None


def test_execute_rag_lookup_no_results(state_with_human_message): # Request fixture
    initial_state = state_with_human_message("Info on elephants?") # Use fixture
    MockKBLookup_Func.invoke.return_value = ""
    MockGetRetriever.return_value = mock_retriever_instance

    result_state = execute_rag_lookup(initial_state)

    MockKBLookup_Func.invoke.assert_called_once_with({"query": "Info on elephants?"})
    assert result_state.get('rag_context') is None
    assert result_state.get('tool_error') is None
    assert result_state['next_node'] == "generate_final_response"
    assert result_state.get('latest_ai_response') is None


def test_execute_rag_lookup_tool_returns_error(state_with_human_message): # Request fixture
    initial_state = state_with_human_message("Search for stuff") # Use fixture
    MockKBLookup_Func.invoke.return_value = {"error": "Vector DB is offline"}
    MockGetRetriever.return_value = mock_retriever_instance

    result_state = execute_rag_lookup(initial_state)

    MockKBLookup_Func.invoke.assert_called_once_with({"query": "Search for stuff"})
    assert result_state.get('rag_context') is None
    assert result_state['tool_error'] == "Vector DB is offline"
    assert result_state['next_node'] == "generate_final_response"
    assert result_state.get('latest_ai_response') is None


def test_execute_rag_lookup_tool_raises_exception(state_with_human_message): # Request fixture
    initial_state = state_with_human_message("Help!") # Use fixture
    MockKBLookup_Func.invoke.side_effect = Exception("Connection lost during RAG")
    MockGetRetriever.return_value = mock_retriever_instance

    result_state = execute_rag_lookup(initial_state)

    MockKBLookup_Func.invoke.assert_called_once_with({"query": "Help!"})
    assert result_state.get('rag_context') is None
    assert "Failed to execute RAG lookup: Connection lost during RAG" in result_state['tool_error']
    assert result_state['next_node'] == "generate_final_response"
    assert result_state.get('latest_ai_response') is None


def test_execute_rag_lookup_no_human_message(initial_state): # Request fixture
    initial_state['messages'] = [AIMessage(content="Hello")]
    MockGetRetriever.return_value = mock_retriever_instance

    result_state = execute_rag_lookup(initial_state)

    MockKBLookup_Func.invoke.assert_not_called()
    assert result_state.get('rag_context') is None
    assert "Could not find user query" in result_state['tool_error']
    assert result_state.get('next_node') is None # Routing depends on graph edge from this error state
    assert result_state.get('latest_ai_response') is None


def test_execute_tool_call_redirects_rag_tool(initial_state): # Use fixture
    tool_call_ai_msg = AIMessage(content="", tool_calls=[
        {"name": MockKBLookup_Func.name, "args": {"query": "Policy query"}, "id": "call_kb_redirect"}])
    initial_state['latest_ai_response'] = tool_call_ai_msg
    initial_state['messages'].append(HumanMessage(content="Policy query"))


    mock_kb_context = "Policy details here."
    MockKBLookup_Func.invoke.return_value = mock_kb_context
    MockGetRetriever.return_value = mock_retriever_instance

    # Call execute_tool_call, expecting it to redirect internally
    result_state = execute_tool_call(initial_state)

    # Check KBLookup tool was invoked (by the internal redirection)
    MockKBLookup_Func.invoke.assert_called_once_with({"query": "Policy query"})

    # Check state reflects RAG success
    assert result_state['rag_context'] == mock_kb_context
    assert result_state.get('tool_error') is None
    assert result_state.get('api_response') is None
    assert result_state['next_node'] == "generate_final_response" # RAG routes to generate
    assert result_state.get('latest_ai_response') is None # Check cleared

    # ToolMessage should NOT have been added by execute_tool_call in this redirection case
    assert len(initial_state['messages']) == 1 # Only Human msg remains in original state dict
    assert not any(isinstance(m, ToolMessage) for m in initial_state['messages'])