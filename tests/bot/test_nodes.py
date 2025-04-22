# tests/bot/test_nodes.py
import pytest
from unittest.mock import MagicMock, patch, ANY
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
import sys
import os

# Import only the mocks DIRECTLY needed for setup/assertions in this file
# We only really need the LLM instance mock here, the tools are mocked globally
from tests.conftest import mock_llm_instance
# Alias mock_llm_instance for clarity if preferred where llm_with_tools is used
mock_llm_with_tools_instance = mock_llm_instance

# Apply specific patches needed for THIS module if conftest isn't enough
@pytest.fixture(scope='module', autouse=True)
def patch_nodes_local_dependencies(module_mocker):
    """Patch dependencies specifically used or defined within bot.nodes."""
    # Import tool function mocks from conftest *within the fixture* for patching available_tools
    from tests.conftest import (
        MockGetOrderStatus_Func, MockGetTrackingInfo_Func, MockGetOrderDetails_Func,
        MockInitiateReturn_Func, MockKBLookup_Func
    )
    # Patch the list directly where it's referenced in bot.nodes
    module_mocker.patch('bot.nodes.available_tools', [
        MockGetOrderStatus_Func, MockGetTrackingInfo_Func, MockGetOrderDetails_Func,
        MockInitiateReturn_Func, MockKBLookup_Func
    ])
    # Ensure these point to the conftest mocks
    module_mocker.patch('bot.nodes.llm', mock_llm_instance)
    module_mocker.patch('bot.nodes.llm_with_tools', mock_llm_with_tools_instance)
    yield
    # No explicit stopall needed for module_mocker with yield

@pytest.fixture
def nodes_module():
    import importlib
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    if project_root not in sys.path:
         sys.path.insert(0, project_root)
    import bot.nodes
    importlib.reload(bot.nodes)
    return bot.nodes

@pytest.fixture
def state_module():
    import bot.state
    return bot.state

def create_full_initial_state(user_query: str, state_module) -> 'GraphState':
     """Creates a GraphState with all keys initialized."""
     return state_module.GraphState(
         messages=[HumanMessage(content=user_query)],
         intent=None, order_id=None, item_sku_to_return=None, return_reason=None,
         needs_clarification=False, clarification_question=None, available_return_items=None,
         rag_context=None, api_response=None, tool_error=None, next_node=None,
     )

# --- classify_intent Tests ---

def test_classify_intent_tool_call_status(nodes_module, state_module):
    # Test logic remains the same, relying on conftest mocks
    initial_state = create_full_initial_state("Status for ORD123?", state_module)
    mock_ai_response = AIMessage(
        content="",
        tool_calls=[{"name": "get_order_status", "args": {"order_id": "ORD123"}, "id": "call_123"}]
    )
    mock_llm_with_tools_instance.invoke.return_value = mock_ai_response
    result_state = nodes_module.classify_intent(initial_state)
    mock_llm_with_tools_instance.invoke.assert_called_once_with(initial_state['messages'])
    assert result_state['intent'] == "get_order_status"
    assert result_state['order_id'] == "ORD123"
    assert result_state['next_node'] == "execute_tool"
    assert len(initial_state['messages']) == 2
    assert initial_state['messages'][1] is mock_ai_response

def test_classify_intent_tool_call_kb(nodes_module, state_module):
    # Test logic remains the same
    initial_state = create_full_initial_state("How do returns work?", state_module)
    mock_ai_response = AIMessage(
        content="",
        tool_calls=[{"name": "knowledge_base_lookup", "args": {"query": "How do returns work?"}, "id": "call_kb"}]
    )
    mock_llm_with_tools_instance.invoke.return_value = mock_ai_response
    result_state = nodes_module.classify_intent(initial_state)
    mock_llm_with_tools_instance.invoke.assert_called_once_with(initial_state['messages'])
    assert result_state['intent'] == "knowledge_base_query"
    assert result_state['next_node'] == "execute_tool"

# ... rest of test_nodes.py tests ...
# IMPORTANT: In execute_tool tests, when asserting calls like
# MockGetOrderStatus.invoke.assert_called_once_with(...)
# you MUST use the mock function name from conftest, e.g. MockGetOrderStatus_Func
def test_execute_tool_via_tool_call_status_success(nodes_module, state_module):
    # Need the mock func from conftest to check calls
    from tests.conftest import MockGetOrderStatus_Func

    tool_call_message = AIMessage(content="", tool_calls=[{"name": "get_order_status", "args": {"order_id": "ORD123"}, "id": "call_1"}])
    initial_state = create_full_initial_state("Status ORD123?", state_module)
    initial_state['messages'].append(tool_call_message)
    initial_state['intent'] = "get_order_status"
    initial_state['order_id'] = "ORD123"

    mock_api_response = {"order_id": "ORD123", "status": "Delivered"}
    MockGetOrderStatus_Func.invoke.return_value = mock_api_response # Configure the func mock

    result_state = nodes_module.execute_tool(initial_state)

    MockGetOrderStatus_Func.invoke.assert_called_once_with({"order_id": "ORD123"}) # Assert on func mock
    assert result_state['api_response'] == mock_api_response
    # ... rest of assertions


def test_execute_tool_via_tool_call_kb_success(nodes_module, state_module):
    # Need the mock func from conftest to check calls
    from tests.conftest import MockKBLookup_Func

    tool_call_message = AIMessage(content="", tool_calls=[{"name": "knowledge_base_lookup", "args": {"query": "Returns policy?"}, "id": "call_3"}])
    initial_state = create_full_initial_state("Returns policy?", state_module)
    initial_state['messages'].append(tool_call_message)
    initial_state['intent'] = "knowledge_base_query"

    mock_kb_context = "Return within 30 days."
    MockKBLookup_Func.invoke.return_value = mock_kb_context # Configure the func mock

    result_state = nodes_module.execute_tool(initial_state)

    MockKBLookup_Func.invoke.assert_called_once_with({"query": "Returns policy?"}) # Assert on func mock
    assert result_state['rag_context'] == mock_kb_context
    # ... rest of assertions


def test_execute_tool_explicit_kb_lookup(nodes_module, state_module):
    # Need the mock func from conftest to check calls
    from tests.conftest import MockKBLookup_Func

    initial_state = create_full_initial_state("Info on shipping?", state_module)
    initial_state['intent'] = "knowledge_base_query"
    mock_kb_context = "Shipping takes 3-5 days."
    MockKBLookup_Func.invoke.return_value = mock_kb_context

    result_state = nodes_module.execute_tool(initial_state)

    MockKBLookup_Func.invoke.assert_called_once_with({"query": "Info on shipping?"}) # Assert on func mock
    # ... rest of assertions

# --- Apply similar changes to other execute_tool tests ---
# Replace assertions on MockGetOrderStatus.invoke with MockGetOrderStatus_Func.invoke etc.

# (Keep the rest of the tests from the previous version, ensuring state is complete
# and assertions use the correct mock function names imported from conftest if needed)

# Example: test_execute_tool_internal_call_get_details_success needs MockGetOrderDetails_Func
def test_execute_tool_internal_call_get_details_success(nodes_module, state_module):
    from tests.conftest import MockGetOrderDetails_Func # Import correct mock name
    initial_state = create_full_initial_state("Return from ORD789", state_module)
    initial_state['intent'] = "initiate_return"
    initial_state['order_id'] = "ORD789"
    initial_state['next_node'] = "handle_return_step_1"

    mock_details = {"order_id": "ORD789", "items": [{"sku": "S1"}], "delivered": True}
    MockGetOrderDetails_Func.invoke.return_value = mock_details # Use func mock

    result_state = nodes_module.execute_tool(initial_state)

    MockGetOrderDetails_Func.invoke.assert_called_once_with({"order_id": "ORD789"}) # Assert func mock
    assert result_state['api_response'] == mock_details
    assert result_state.get('tool_error') is None

# Apply this pattern to all execute_tool tests checking tool calls.
# Handle_multi_turn tests don't call tools directly, so no changes needed there.
# Generate_response tests call llm.invoke, which uses mock_llm_instance, so no changes needed there.

# --- (Paste remaining tests from previous version, ensuring correct mock usage for asserts) ---

def test_classify_intent_no_tool_call_defaults_to_kb(nodes_module, state_module):
    initial_state = create_full_initial_state("Tell me about shipping.", state_module)
    mock_ai_response = AIMessage(content="Okay, let me check.")
    mock_llm_with_tools_instance.invoke.return_value = mock_ai_response
    result_state = nodes_module.classify_intent(initial_state)
    mock_llm_with_tools_instance.invoke.assert_called_once_with(initial_state['messages'])
    assert result_state['intent'] == "knowledge_base_query"
    assert result_state['next_node'] == "execute_tool"
    assert result_state['needs_clarification'] is False

def test_classify_intent_multi_turn_return_sku_provided_match(nodes_module, state_module):
    initial_state = state_module.GraphState(
        messages=[HumanMessage(content="start"), AIMessage(content="Which SKU?"), HumanMessage(content="ITEM004")],
        intent="initiate_return", needs_clarification=True,
        available_return_items=[{"sku": "ITEM004", "name":"A"}, {"sku": "ITEM005", "name":"B"}],
        order_id="ORD_TEST", rag_context=None, api_response=None, tool_error=None, next_node=None, item_sku_to_return=None, return_reason=None
    )
    mock_ai_response = AIMessage(content="")
    mock_llm_with_tools_instance.invoke.return_value = mock_ai_response
    result_state = nodes_module.classify_intent(initial_state)
    mock_llm_with_tools_instance.invoke.assert_called_once_with(initial_state['messages'])
    assert result_state['intent'] == "return_item_selection"
    assert result_state['item_sku_to_return'] == "ITEM004"
    assert result_state['needs_clarification'] is False
    assert result_state['next_node'] == "handle_return_step_2"

def test_classify_intent_multi_turn_return_sku_provided_no_match(nodes_module, state_module):
    initial_state = state_module.GraphState(
        messages=[HumanMessage(content="start"), AIMessage(content="Which SKU?"), HumanMessage(content="WRONG_SKU")],
        intent="initiate_return", needs_clarification=True,
        available_return_items=[{"sku": "ITEM004", "name":"A"}],
        order_id="ORD_TEST", rag_context=None, api_response=None, tool_error=None, next_node=None, item_sku_to_return=None, return_reason=None
    )
    mock_ai_response = AIMessage(content="")
    mock_llm_with_tools_instance.invoke.return_value = mock_ai_response
    result_state = nodes_module.classify_intent(initial_state)
    mock_llm_with_tools_instance.invoke.assert_called_once_with(initial_state['messages'])
    assert result_state.get('intent') == "initiate_return" # Stays in current intent
    assert result_state.get('item_sku_to_return') is None
    assert result_state['needs_clarification'] is True
    assert "doesn't seem to match" in result_state['clarification_question']
    assert result_state['next_node'] == "generate_response"

def test_classify_intent_multi_turn_return_reason_provided(nodes_module, state_module):
    initial_state = state_module.GraphState(
        messages=[HumanMessage(content="SKU ok"), AIMessage(content="Reason?"), HumanMessage(content="It broke")],
        intent="return_item_selection", needs_clarification=True,
        item_sku_to_return="ITEM004", order_id="ORD_TEST", rag_context=None, api_response=None, tool_error=None, next_node=None, available_return_items=None, return_reason=None
    )
    mock_ai_response = AIMessage(content="")
    mock_llm_with_tools_instance.invoke.return_value = mock_ai_response
    result_state = nodes_module.classify_intent(initial_state)
    mock_llm_with_tools_instance.invoke.assert_called_once_with(initial_state['messages'])
    assert result_state['intent'] == "return_reason_provided"
    assert result_state['return_reason'] == "It broke"
    assert result_state['needs_clarification'] is False
    assert result_state['next_node'] == "handle_return_step_3"

def test_classify_intent_llm_exception(nodes_module, state_module, mocker):
    logger_error = mocker.patch('logging.Logger.error')
    initial_state = create_full_initial_state("Break please", state_module)
    mock_llm_with_tools_instance.invoke.side_effect = Exception("LLM provider error")
    result_state = nodes_module.classify_intent(initial_state)
    mock_llm_with_tools_instance.invoke.assert_called_once_with(initial_state['messages'])
    assert result_state['intent'] == "unsupported"
    assert "trouble understanding" in result_state['api_response']['message']
    assert result_state['next_node'] == "generate_response"
    logger_error.assert_called_once()

# --- execute_tool Tests ---

# test_execute_tool_via_tool_call_status_success corrected above

def test_execute_tool_via_tool_call_status_api_error(nodes_module, state_module):
    from tests.conftest import MockGetOrderStatus_Func # Import correct mock name
    tool_call_message = AIMessage(content="", tool_calls=[{"name": "get_order_status", "args": {"order_id": "NF"}, "id": "call_2"}])
    initial_state = create_full_initial_state("Status NF?", state_module)
    initial_state['messages'].append(tool_call_message)
    initial_state['intent'] = "get_order_status"
    initial_state['order_id'] = "NF"

    mock_error_response = {"error": "Order ID 'NF' not found."}
    MockGetOrderStatus_Func.invoke.return_value = mock_error_response

    result_state = nodes_module.execute_tool(initial_state)

    MockGetOrderStatus_Func.invoke.assert_called_once_with({"order_id": "NF"})
    assert result_state['api_response'] == mock_error_response
    assert result_state['tool_error'] == "Order ID 'NF' not found."
    assert len(initial_state['messages']) == 3
    assert isinstance(initial_state['messages'][2], ToolMessage)

# test_execute_tool_via_tool_call_kb_success corrected above

# test_execute_tool_explicit_kb_lookup corrected above

def test_execute_tool_explicit_kb_lookup_no_query(nodes_module, state_module):
    from tests.conftest import MockKBLookup_Func # Import correct mock name
    initial_state = state_module.GraphState( messages=[], intent="knowledge_base_query")
    result_state = nodes_module.execute_tool(initial_state)
    MockKBLookup_Func.invoke.assert_not_called() # Assert correct mock
    assert result_state.get('rag_context') is None
    assert "Could not find user query" in result_state['tool_error']

# test_execute_tool_internal_call_get_details_success corrected above

def test_execute_tool_internal_call_get_details_error(nodes_module, state_module):
    from tests.conftest import MockGetOrderDetails_Func # Import correct mock name
    initial_state = create_full_initial_state("Return from NF", state_module)
    initial_state['intent']="initiate_return"
    initial_state['order_id']="NF"
    initial_state['next_node']="handle_return_step_1"

    mock_error = {"error": "Order ID 'NF' not found."}
    MockGetOrderDetails_Func.invoke.return_value = mock_error # Use func mock
    result_state = nodes_module.execute_tool(initial_state)
    MockGetOrderDetails_Func.invoke.assert_called_once_with({"order_id": "NF"}) # Assert func mock
    assert result_state['api_response'] == mock_error
    assert result_state['tool_error'] == "Order ID 'NF' not found."

def test_execute_tool_internal_call_submit_return_success(nodes_module, state_module):
    from tests.conftest import MockInitiateReturn_Func # Import correct mock name
    initial_state = create_full_initial_state("Reason is X", state_module)
    initial_state.update({
        "intent": "return_reason_provided", "order_id": "ORD789",
        "item_sku_to_return": "SKU1", "return_reason": "Reason is X",
        "next_node": "execute_tool"
    })
    mock_return_resp = {"return_id": "RET123", "message": "Success"}
    MockInitiateReturn_Func.invoke.return_value = mock_return_resp # Use func mock

    result_state = nodes_module.execute_tool(initial_state)

    MockInitiateReturn_Func.invoke.assert_called_once_with({"order_id": "ORD789", "sku": "SKU1", "reason": "Reason is X"}) # Assert func mock
    assert result_state['api_response'] == mock_return_resp
    assert result_state.get('tool_error') is None
    assert result_state.get('item_sku_to_return') is None
    assert result_state.get('return_reason') is None
    assert result_state.get('available_return_items') is None

def test_execute_tool_internal_call_submit_return_error(nodes_module, state_module):
    from tests.conftest import MockInitiateReturn_Func # Import correct mock name
    initial_state = create_full_initial_state("Reason is R", state_module)
    initial_state.update({
        "intent":"return_reason_provided", "order_id":"ORD",
        "item_sku_to_return":"S", "return_reason":"R",
        "next_node":"execute_tool"
    })
    mock_error = {"error": "Return failed"}
    MockInitiateReturn_Func.invoke.return_value = mock_error # Use func mock
    result_state = nodes_module.execute_tool(initial_state)

    MockInitiateReturn_Func.invoke.assert_called_once_with({"order_id": "ORD", "sku": "S", "reason": "R"}) # Assert func mock
    assert result_state['api_response'] == mock_error
    assert result_state['tool_error'] == "Return failed"
    assert 'item_sku_to_return' not in result_state or result_state.get('item_sku_to_return') is not None

# --- handle_multi_turn_return Tests ---
# (These tests should be okay as they don't directly check tool mocks)
def test_handle_multi_turn_step1_details_ok(nodes_module, state_module):
    items = [{"sku": "S1", "name": "N1"}, {"sku": "S2", "name": "N2"}]
    initial_state = create_full_initial_state("Return ORD789", state_module)
    initial_state.update({
        "intent": "initiate_return", "order_id": "ORD789",
        "api_response": {"order_id": "ORD789", "items": items, "delivered": True},
        "tool_error": None, "next_node": "handle_return_step_1",
    })
    result_state = nodes_module.handle_multi_turn_return(initial_state)
    assert result_state['needs_clarification'] is True
    assert result_state['available_return_items'] == items
    assert "Which item would you like to return?" in result_state['clarification_question']
    assert result_state['next_node'] == "generate_response"

def test_handle_multi_turn_step1_details_error(nodes_module, state_module):
     initial_state = create_full_initial_state("Return NF", state_module)
     initial_state.update({
        "intent":"initiate_return", "order_id":"NF",
        "api_response":{"error": "Order not found"},
        "tool_error":"Order not found", "next_node":"handle_return_step_1"
    })
     result_state = nodes_module.handle_multi_turn_return(initial_state)
     assert result_state['needs_clarification'] is False
     assert result_state['tool_error'] == 'Order not found' # FIX: Pass error through
     assert result_state['next_node'] == "generate_response"

def test_handle_multi_turn_step1_details_no_items(nodes_module, state_module):
    initial_state = create_full_initial_state("Return ORD123", state_module)
    initial_state.update({
        "intent":"initiate_return", "order_id":"ORD123",
        "api_response":{"order_id": "ORD123", "items": [], "delivered": True},
        "tool_error":None, "next_node":"handle_return_step_1"
    })
    result_state = nodes_module.handle_multi_turn_return(initial_state)
    assert result_state['needs_clarification'] is False
    assert "couldn't find any returnable items" in result_state['tool_error']
    assert result_state['next_node'] == "generate_response"

def test_handle_multi_turn_step2_sku_selected(nodes_module, state_module):
    initial_state = create_full_initial_state("ITEM004", state_module)
    initial_state.update({
        "intent":"return_item_selection", "item_sku_to_return":"SKU1",
        "tool_error":None, "next_node":"handle_return_step_2"
    })
    result_state = nodes_module.handle_multi_turn_return(initial_state)
    assert result_state['needs_clarification'] is True
    assert "why you're returning it?" in result_state['clarification_question']
    assert result_state['next_node'] == "generate_response"

def test_handle_multi_turn_step3_reason_provided(nodes_module, state_module):
    initial_state = create_full_initial_state("It broke", state_module)
    initial_state.update({
        "intent":"return_reason_provided", "return_reason":"Broken",
        "tool_error":None, "next_node":"handle_return_step_3"
    })
    result_state = nodes_module.handle_multi_turn_return(initial_state)
    assert result_state['needs_clarification'] is False
    assert result_state.get('clarification_question') is None
    assert result_state['next_node'] == "execute_tool"

# --- generate_response Tests ---
# (These tests use mock_llm_instance, which should be correctly mocked by conftest)
def test_generate_response_clarification(nodes_module, state_module):
    question = "Which SKU do you want to return?"
    initial_state = create_full_initial_state("", state_module)
    initial_state.update({"needs_clarification": True, "clarification_question": question})
    update_dict = nodes_module.generate_response(initial_state)
    assert update_dict == {}
    final_message = initial_state["messages"][-1]
    assert isinstance(final_message, AIMessage)
    assert final_message.content == question

def test_generate_response_tool_error(nodes_module, state_module):
    error_msg = "Order ID 'NF' not found."
    initial_state = create_full_initial_state("", state_module)
    initial_state['tool_error'] = error_msg
    update_dict = nodes_module.generate_response(initial_state)
    assert update_dict == {}
    final_message = initial_state["messages"][-1]
    assert isinstance(final_message, AIMessage)
    assert f"I encountered an issue: {error_msg}" == final_message.content

def test_generate_response_api_return_failure_with_message(nodes_module, state_module):
     initial_state = create_full_initial_state("", state_module)
     initial_state.update({ "intent":"return_reason_provided", "api_response":{"message": "Return system offline.", "status": "Failed"} })
     update_dict = nodes_module.generate_response(initial_state)
     assert update_dict == {}
     final_message = initial_state["messages"][-1]
     assert isinstance(final_message, AIMessage)
     assert final_message.content == 'There was an issue: Return system offline.'

def test_generate_response_api_return_failure_with_error(nodes_module, state_module):
     initial_state = create_full_initial_state("", state_module)
     initial_state.update({ "intent":"return_reason_provided", "api_response":{"error": "Item already returned."} })
     update_dict = nodes_module.generate_response(initial_state)
     assert update_dict == {}
     final_message = initial_state["messages"][-1]
     assert isinstance(final_message, AIMessage)
     assert final_message.content == 'There was an issue: Item already returned.'

def test_generate_response_rag_success(nodes_module, state_module):
    query = "Return policy?"
    context = "Items can be returned within 30 days if unused."
    initial_state = create_full_initial_state(query, state_module)
    initial_state.update({
        "intent":"knowledge_base_query", "rag_context":context
    })
    mock_llm_response = AIMessage(content="You can return unused items within 30 days.")
    mock_llm_instance.invoke.return_value = mock_llm_response
    update_dict = nodes_module.generate_response(initial_state)
    assert update_dict == {}
    mock_llm_instance.invoke.assert_called_once()
    prompt_arg = mock_llm_instance.invoke.call_args[0][0]
    assert context in prompt_arg
    assert query in prompt_arg
    final_message = initial_state["messages"][-1]
    assert isinstance(final_message, AIMessage)
    assert final_message.content == mock_llm_response.content

def test_generate_response_rag_llm_failure(nodes_module, state_module, mocker):
    logger_error = mocker.patch('logging.Logger.error')
    query = "Return policy?"
    context = "Items can be returned within 30 days if unused."
    initial_state = create_full_initial_state(query, state_module)
    initial_state.update({ "intent":"knowledge_base_query", "rag_context":context})
    mock_llm_instance.invoke.side_effect = Exception("LLM RAG synthesis failed")
    update_dict = nodes_module.generate_response(initial_state)
    assert update_dict == {}
    mock_llm_instance.invoke.assert_called_once()
    logger_error.assert_called_once()
    final_message = initial_state["messages"][-1]
    assert isinstance(final_message, AIMessage)
    assert "had trouble formulating a final answer" in final_message.content

def test_generate_response_rag_no_context(nodes_module, state_module):
     initial_state = create_full_initial_state("query", state_module)
     initial_state.update({"intent":"knowledge_base_query", "rag_context":None})
     update_dict = nodes_module.generate_response(initial_state)
     assert update_dict == {}
     final_message = initial_state["messages"][-1]
     assert isinstance(final_message, AIMessage)
     assert "couldn't find specific information" in final_message.content
     mock_llm_instance.invoke.assert_not_called()

def test_generate_response_unsupported(nodes_module, state_module):
     initial_state = create_full_initial_state("gibberish", state_module)
     initial_state['intent'] = "unsupported"
     update_dict = nodes_module.generate_response(initial_state)
     assert update_dict == {}
     final_message = initial_state["messages"][-1]
     assert isinstance(final_message, AIMessage)
     assert final_message.content == "I'm sorry, I can't assist with that specific request right now. I can help with order status, tracking, returns, and answer general questions from our FAQ."