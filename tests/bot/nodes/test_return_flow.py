# tests/bot/nodes/test_return_flow.py
import os
import sys
import pytest
from unittest.mock import MagicMock
from langchain_core.messages import HumanMessage, AIMessage

# Ensure the app root is in the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Import the specific node functions and state
from bot.nodes.return_flow import handle_return_step, submit_return_request
from bot.state import GraphState

# Import mocks from conftest
from tests.conftest import (
    MockGetOrderDetails_Func,
    MockInitiateReturn_Func
)

# REMOVE the problematic import:
# from .test_classification import create_full_initial_state # <--- REMOVE THIS LINE

# --- Test Data ---
mock_items_valid = [{"sku": "SKU1", "name": "Item One"}, {"sku": "SKU2", "name": "Item Two"}]
mock_details_valid = {"order_id": "ORD789", "items": mock_items_valid, "delivered": True, "status": "Delivered"}
mock_details_not_delivered = {"order_id": "ORD123", "items": mock_items_valid, "delivered": False, "status": "Shipped"}
mock_details_no_items = {"order_id": "ORD000", "items": [], "delivered": True, "status": "Delivered"}
mock_details_error = {"error": "Order ID 'NF' not found."}
mock_return_success = {"return_id": "RET123", "status": "Return Initiated", "message": "Return initiated successfully."}
mock_return_error = {"error": "Item already returned."}


# --- Tests for handle_return_step ---

def test_handle_return_step1_fetch_details_success(state_with_human_message): # Use fixture
    initial_state = state_with_human_message("Return ORD789") # Use fixture
    initial_state['order_id'] = 'ORD789'
    # No available_items yet

    MockGetOrderDetails_Func.invoke.return_value = mock_details_valid

    result_state = handle_return_step(initial_state)

    MockGetOrderDetails_Func.invoke.assert_called_once_with({"order_id": "ORD789"})
    assert result_state['available_return_items'] == mock_items_valid
    assert result_state['needs_clarification'] is True
    # ... rest of assertions

def test_handle_return_step1_fetch_details_not_delivered(state_with_human_message): # Use fixture
    initial_state = state_with_human_message("Return ORD123") # Use fixture
    initial_state['order_id'] = 'ORD123'

    MockGetOrderDetails_Func.invoke.return_value = mock_details_not_delivered

    result_state = handle_return_step(initial_state)

    MockGetOrderDetails_Func.invoke.assert_called_once_with({"order_id": "ORD123"})
    assert "not marked as delivered yet" in result_state['tool_error']
    # ... rest of assertions

def test_handle_return_step1_fetch_details_no_items(state_with_human_message): # Use fixture
    initial_state = state_with_human_message("Return ORD000") # Use fixture
    initial_state['order_id'] = 'ORD000'

    MockGetOrderDetails_Func.invoke.return_value = mock_details_no_items

    result_state = handle_return_step(initial_state)

    MockGetOrderDetails_Func.invoke.assert_called_once_with({"order_id": "ORD000"})
    assert "No returnable items found" in result_state['tool_error']
    # ... rest of assertions

def test_handle_return_step1_fetch_details_api_error(state_with_human_message): # Use fixture
    initial_state = state_with_human_message("Return NF") # Use fixture
    initial_state['order_id'] = 'NF'

    MockGetOrderDetails_Func.invoke.return_value = mock_details_error

    result_state = handle_return_step(initial_state)

    MockGetOrderDetails_Func.invoke.assert_called_once_with({"order_id": "NF"})
    assert result_state['tool_error'] == mock_details_error["error"]
    # ... rest of assertions


def test_handle_return_step2_process_sku_valid(initial_state): # Use base fixture
    # Manually construct messages for multi-turn
    messages = [
        HumanMessage(content="Return ORD789"),
        AIMessage(content="Which SKU?"),
        HumanMessage(content="SKU1") # User replies with valid SKU
    ]
    initial_state['messages'] = messages
    initial_state['order_id'] = 'ORD789'
    initial_state['available_return_items'] = mock_items_valid
    initial_state['needs_clarification'] = True # Bot asked for SKU

    result_state = handle_return_step(initial_state)

    assert result_state['item_sku_to_return'] == "SKU1"
    assert result_state['needs_clarification'] is True # Now needs reason
    # ... rest of assertions

def test_handle_return_step2_process_sku_invalid(initial_state): # Use base fixture
    messages = [
        HumanMessage(content="Return ORD789"),
        AIMessage(content="Which SKU?"),
        HumanMessage(content="INVALID_SKU") # User replies with invalid SKU
    ]
    initial_state['messages'] = messages
    initial_state['order_id'] = 'ORD789'
    initial_state['available_return_items'] = mock_items_valid
    initial_state['needs_clarification'] = True # Bot asked for SKU

    result_state = handle_return_step(initial_state)

    assert result_state.get('item_sku_to_return') is None # Should reset/stay None
    assert result_state['needs_clarification'] is True # Still needs SKU
    # ... rest of assertions


def test_handle_return_step3_process_reason_provided(initial_state): # Use base fixture
    messages = [
        HumanMessage(content="Return ORD789"), AIMessage(content="Which SKU?"),
        HumanMessage(content="SKU1"), AIMessage(content="Why are you returning?"),
        HumanMessage(content="It arrived broken") # User provides reason
    ]
    initial_state['messages'] = messages
    initial_state['order_id'] = 'ORD789'
    initial_state['available_return_items'] = mock_items_valid
    initial_state['item_sku_to_return'] = "SKU1"
    initial_state['needs_clarification'] = True # Bot asked for reason

    result_state = handle_return_step(initial_state)

    assert result_state['return_reason'] == "It arrived broken"
    assert result_state['needs_clarification'] is False # All info gathered
    # ... rest of assertions

def test_handle_return_step3_process_reason_skipped(initial_state): # Use base fixture
    messages = [
        HumanMessage(content="Return ORD789"), AIMessage(content="Which SKU?"),
        HumanMessage(content="SKU1"), AIMessage(content="Why are you returning?"),
        HumanMessage(content="skip") # User skips reason
    ]
    initial_state['messages'] = messages
    initial_state['order_id'] = 'ORD789'; initial_state['available_return_items'] = mock_items_valid
    initial_state['item_sku_to_return'] = "SKU1"; initial_state['needs_clarification'] = True

    result_state = handle_return_step(initial_state)

    assert result_state.get('return_reason') is None
    assert result_state['needs_clarification'] is False
    # ... rest of assertions


def test_handle_return_step_unexpected_state(state_with_human_message): # Use fixture
    initial_state = state_with_human_message("What about returns?") # Use fixture
    initial_state['intent'] = 'initiate_return' # But no order_id

    result_state = handle_return_step(initial_state)

    assert "Something went wrong" in result_state['tool_error']
    # ... rest of assertions


# --- Tests for submit_return_request ---

def test_submit_return_request_success(initial_state): # Use fixture
    initial_state['order_id'] = 'ORD789'
    initial_state['item_sku_to_return'] = 'SKU1'
    initial_state['return_reason'] = 'Defective'

    MockInitiateReturn_Func.invoke.return_value = mock_return_success

    result_state = submit_return_request(initial_state)

    MockInitiateReturn_Func.invoke.assert_called_once_with(
        {"order_id": "ORD789", "sku": "SKU1", "reason": "Defective"}
    )
    assert result_state['api_response'] == mock_return_success
    assert result_state.get('tool_error') is None
    # Check state is cleared on success
    assert result_state.get('item_sku_to_return') is None
    # ... rest of assertions

def test_submit_return_request_api_error(initial_state): # Use fixture
    initial_state['order_id'] = 'ORD789'
    initial_state['item_sku_to_return'] = 'SKU1'
    initial_state['return_reason'] = None

    MockInitiateReturn_Func.invoke.return_value = mock_return_error

    result_state = submit_return_request(initial_state)

    MockInitiateReturn_Func.invoke.assert_called_once_with(
        {"order_id": "ORD789", "sku": "SKU1", "reason": None}
    )
    assert result_state['api_response'] == mock_return_error
    assert result_state['tool_error'] == mock_return_error["error"]
    # ... rest of assertions

def test_submit_return_request_missing_info(initial_state): # Use fixture
    initial_state['order_id'] = 'ORD789'
    # Missing item_sku_to_return

    result_state = submit_return_request(initial_state)

    MockInitiateReturn_Func.invoke.assert_not_called()
    assert "Missing Order ID or Item SKU" in result_state['tool_error']
    # ... rest of assertions