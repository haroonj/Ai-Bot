# tests/bot/test_nodes.py
import pytest
from unittest.mock import MagicMock, patch, ANY
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
import sys

# --- Mock External Dependencies EARLY ---
mock_llm_instance = MagicMock(name="MockLLMInstance", spec=True)
mock_llm_with_tools_instance = MagicMock(name="MockLLMWithToolsInstance", spec=True)

# Mock tool functions - essential for execute_tool tests
MockGetOrderStatus = MagicMock(name="get_order_status", spec=True)
MockGetTrackingInfo = MagicMock(name="get_tracking_info", spec=True)
MockGetOrderDetails = MagicMock(name="get_order_details", spec=True)
MockInitiateReturn = MagicMock(name="initiate_return_request", spec=True)
MockKBLookup = MagicMock(name="knowledge_base_lookup", spec=True)

# Apply patches using a fixture targeting bot.nodes where objects are imported/used
@pytest.fixture(scope='module', autouse=True)
def patch_nodes_dependencies(module_mocker):
    """Apply patches for the duration of the nodes test module."""
    # Prepare dict for patch.multiple targeting bot.nodes
    node_patches = {
        'llm': mock_llm_instance,
        'llm_with_tools': mock_llm_with_tools_instance,
        'get_order_status': MockGetOrderStatus,
        'get_tracking_info': MockGetTrackingInfo,
        'get_order_details': MockGetOrderDetails,
        'initiate_return_request': MockInitiateReturn,
        'knowledge_base_lookup': MockKBLookup,
        'available_tools': [ # Patch the list with mock tools
            MockGetOrderStatus, MockGetTrackingInfo, MockGetOrderDetails,
            MockInitiateReturn, MockKBLookup
        ]
    }
    # Patch directly into the bot.nodes module namespace
    module_mocker.patch.multiple('bot.nodes', **node_patches)
    yield
    module_mocker.stopall()


# Import modules under test *after* patches applied by fixture
@pytest.fixture
def nodes_module():
    import importlib
    import bot.nodes
    # Reload to ensure patched objects are used
    importlib.reload(bot.nodes)
    return bot.nodes

@pytest.fixture
def state_module():
    import bot.state
    return bot.state

# Helper function using the state_module fixture
def create_full_initial_state(user_query: str, state_module) -> 'GraphState':
     """Creates a GraphState with all keys initialized."""
     # Ensure all keys expected by ANY node are present
     return state_module.GraphState(
         messages=[HumanMessage(content=user_query)],
         intent=None, order_id=None, item_sku_to_return=None, return_reason=None,
         needs_clarification=False, clarification_question=None, available_return_items=None,
         rag_context=None, api_response=None, tool_error=None, next_node=None,
     )

# Fixture to reset mocks before each test function
@pytest.fixture(autouse=True)
def reset_node_mocks():
    """Reset mocks between each test function."""
    mock_llm_instance.reset_mock()
    mock_llm_with_tools_instance.reset_mock()
    MockGetOrderStatus.reset_mock()
    MockGetTrackingInfo.reset_mock()
    MockGetOrderDetails.reset_mock()
    MockInitiateReturn.reset_mock()
    MockKBLookup.reset_mock()
    # Reset side effects
    mock_llm_instance.invoke.side_effect = None
    mock_llm_with_tools_instance.invoke.side_effect = None
    MockGetOrderStatus.invoke.side_effect = None
    MockGetTrackingInfo.invoke.side_effect = None
    MockGetOrderDetails.invoke.side_effect = None
    MockInitiateReturn.invoke.side_effect = None
    MockKBLookup.invoke.side_effect = None

# --- classify_intent Tests ---

def test_classify_intent_tool_call_status(nodes_module, state_module):
    initial_state = create_full_initial_state("Status for ORD123?", state_module)
    # Fix: Use string name for the tool to pass Pydantic validation
    mock_ai_response = AIMessage(
        content="",
        tool_calls=[{
            "name": "get_order_status", # STRING name
            "args": {"order_id": "ORD123"},
            "id": "call_123"
        }]
    )
    mock_llm_with_tools_instance.invoke.return_value = mock_ai_response

    result_state = nodes_module.classify_intent(initial_state)

    mock_llm_with_tools_instance.invoke.assert_called_once_with(initial_state['messages'])
    assert result_state['intent'] == "get_order_status"
    assert result_state['order_id'] == "ORD123"
    assert result_state['next_node'] == "execute_tool"
    assert len(initial_state['messages']) == 2 # Check message appended
    assert initial_state['messages'][1] is mock_ai_response

def test_classify_intent_tool_call_kb(nodes_module, state_module):
    initial_state = create_full_initial_state("How do returns work?", state_module)
     # Fix: Use string name for the tool
    mock_ai_response = AIMessage(
        content="",
        tool_calls=[{
            "name": "knowledge_base_lookup", # STRING name
            "args": {"query": "How do returns work?"},
            "id": "call_kb"
        }]
    )
    mock_llm_with_tools_instance.invoke.return_value = mock_ai_response

    result_state = nodes_module.classify_intent(initial_state)

    mock_llm_with_tools_instance.invoke.assert_called_once_with(initial_state['messages'])
    assert result_state['intent'] == "knowledge_base_query"
    assert result_state['next_node'] == "execute_tool"

def test_classify_intent_no_tool_call_defaults_to_kb(nodes_module, state_module):
    initial_state = create_full_initial_state("Tell me about shipping.", state_module)
    mock_ai_response = AIMessage(content="Okay, let me check.") # NO tool calls
    mock_llm_with_tools_instance.invoke.return_value = mock_ai_response

    result_state = nodes_module.classify_intent(initial_state)

    mock_llm_with_tools_instance.invoke.assert_called_once_with(initial_state['messages'])
    # Mocking works, so it should correctly default to KB query
    assert result_state['intent'] == "knowledge_base_query"
    assert result_state['next_node'] == "execute_tool"
    assert result_state['needs_clarification'] is False

def test_classify_intent_multi_turn_return_sku_provided_match(nodes_module, state_module):
    # State represents being asked for SKU
    initial_state = state_module.GraphState(
        messages=[HumanMessage(content="start"), AIMessage(content="Which SKU?"), HumanMessage(content="ITEM004")],
        intent="initiate_return",
        needs_clarification=True,
        available_return_items=[{"sku": "ITEM004", "name":"A"}, {"sku": "ITEM005", "name":"B"}],
        order_id="ORD_TEST", rag_context=None, api_response=None, tool_error=None, next_node=None, item_sku_to_return=None, return_reason=None
    )
    mock_ai_response = AIMessage(content="") # No tool call expected when parsing response
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
        intent="initiate_return",
        needs_clarification=True,
        available_return_items=[{"sku": "ITEM004", "name":"A"}],
        order_id="ORD_TEST", rag_context=None, api_response=None, tool_error=None, next_node=None, item_sku_to_return=None, return_reason=None
    )
    mock_ai_response = AIMessage(content="")
    mock_llm_with_tools_instance.invoke.return_value = mock_ai_response

    result_state = nodes_module.classify_intent(initial_state)

    mock_llm_with_tools_instance.invoke.assert_called_once_with(initial_state['messages'])
    # Stays in 'initiate_return' but asks clarification
    assert result_state.get('intent') == "initiate_return" # Does not proceed
    assert result_state.get('item_sku_to_return') is None
    assert result_state['needs_clarification'] is True
    assert "doesn't seem to match" in result_state['clarification_question']
    assert result_state['next_node'] == "generate_response"

def test_classify_intent_multi_turn_return_reason_provided(nodes_module, state_module):
    initial_state = state_module.GraphState(
        messages=[HumanMessage(content="SKU ok"), AIMessage(content="Reason?"), HumanMessage(content="It broke")],
        intent="return_item_selection", # Previous intent
        needs_clarification=True,      # Was expecting reason
        item_sku_to_return="ITEM004",
        order_id="ORD_TEST", rag_context=None, api_response=None, tool_error=None, next_node=None, available_return_items=None, return_reason=None
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

def test_execute_tool_via_tool_call_status_success(nodes_module, state_module):
    # Fix: Use string name for tool call
    tool_call_message = AIMessage(content="", tool_calls=[{"name": "get_order_status", "args": {"order_id": "ORD123"}, "id": "call_1"}])
    # Fix: Ensure all keys exist
    initial_state = state_module.GraphState(
        messages=[ HumanMessage(content="Status ORD123?"), tool_call_message ],
        intent="get_order_status", order_id="ORD123",
        # Add other keys with default values
        item_sku_to_return=None, return_reason=None, needs_clarification=False, clarification_question=None, available_return_items=None, rag_context=None, api_response=None, tool_error=None, next_node=None
    )
    mock_api_response = {"order_id": "ORD123", "status": "Delivered"}
    MockGetOrderStatus.invoke.return_value = mock_api_response # Mock the *tool function's* invoke

    result_state = nodes_module.execute_tool(initial_state)

    # Assert the *tool function's mock* was called
    MockGetOrderStatus.invoke.assert_called_once_with({"order_id": "ORD123"})
    assert result_state['api_response'] == mock_api_response
    assert result_state.get('rag_context') is None
    assert result_state.get('tool_error') is None
    assert len(initial_state['messages']) == 3 # Original Human, AI (tool call), ToolMessage
    assert isinstance(initial_state['messages'][2], ToolMessage)
    assert initial_state['messages'][2].tool_call_id == "call_1"
    assert "Successfully called API tool" in initial_state['messages'][2].content

def test_execute_tool_via_tool_call_status_api_error(nodes_module, state_module):
    tool_call_message = AIMessage(content="", tool_calls=[{"name": "get_order_status", "args": {"order_id": "NF"}, "id": "call_2"}])
    initial_state = state_module.GraphState(
        messages=[ HumanMessage(content="Status NF?"), tool_call_message ],
        intent="get_order_status", order_id="NF",
        item_sku_to_return=None, return_reason=None, needs_clarification=False, clarification_question=None, available_return_items=None, rag_context=None, api_response=None, tool_error=None, next_node=None
    )
    mock_error_response = {"error": "Order ID 'NF' not found."}
    MockGetOrderStatus.invoke.return_value = mock_error_response

    result_state = nodes_module.execute_tool(initial_state)

    MockGetOrderStatus.invoke.assert_called_once_with({"order_id": "NF"})
    assert result_state['api_response'] == mock_error_response
    assert result_state.get('rag_context') is None
    assert result_state['tool_error'] == "Order ID 'NF' not found."
    assert "Tool execution returned an error" in initial_state['messages'][2].content

def test_execute_tool_via_tool_call_kb_success(nodes_module, state_module):
    tool_call_message = AIMessage(content="", tool_calls=[{"name": "knowledge_base_lookup", "args": {"query": "Returns policy?"}, "id": "call_3"}])
    initial_state = state_module.GraphState(
        messages=[ HumanMessage(content="Returns policy?"), tool_call_message ],
        intent="knowledge_base_query",
        order_id=None, item_sku_to_return=None, return_reason=None, needs_clarification=False, clarification_question=None, available_return_items=None, rag_context=None, api_response=None, tool_error=None, next_node=None
    )
    mock_kb_context = "Return within 30 days."
    MockKBLookup.invoke.return_value = mock_kb_context

    result_state = nodes_module.execute_tool(initial_state)

    MockKBLookup.invoke.assert_called_once_with({"query": "Returns policy?"})
    assert result_state.get('api_response') is None
    assert result_state['rag_context'] == mock_kb_context
    assert result_state.get('tool_error') is None
    assert "Successfully looked up knowledge base" in initial_state['messages'][2].content

def test_execute_tool_explicit_kb_lookup(nodes_module, state_module):
    initial_state = create_full_initial_state("Info on shipping?", state_module)
    initial_state['intent'] = "knowledge_base_query" # Set intent for this path
    mock_kb_context = "Shipping takes 3-5 days."
    MockKBLookup.invoke.return_value = mock_kb_context

    result_state = nodes_module.execute_tool(initial_state)

    MockKBLookup.invoke.assert_called_once_with({"query": "Info on shipping?"})
    assert result_state['rag_context'] == mock_kb_context
    assert result_state.get('tool_error') is None
    assert len(initial_state['messages']) == 1 # No tool call, so no ToolMessage added

def test_execute_tool_explicit_kb_lookup_no_query(nodes_module, state_module):
    initial_state = state_module.GraphState( messages=[], intent="knowledge_base_query") # Missing keys added by helper
    result_state = nodes_module.execute_tool(initial_state)
    MockKBLookup.invoke.assert_not_called()
    assert result_state.get('rag_context') is None
    assert "Could not find user query" in result_state['tool_error']

def test_execute_tool_internal_call_get_details_success(nodes_module, state_module):
    initial_state = create_full_initial_state("Return from ORD789", state_module)
    initial_state['intent'] = "initiate_return"
    initial_state['order_id'] = "ORD789"
    initial_state['next_node'] = "handle_return_step_1"

    mock_details = {"order_id": "ORD789", "items": [{"sku": "S1"}], "delivered": True}
    MockGetOrderDetails.invoke.return_value = mock_details

    result_state = nodes_module.execute_tool(initial_state)

    MockGetOrderDetails.invoke.assert_called_once_with({"order_id": "ORD789"})
    assert result_state['api_response'] == mock_details
    assert result_state.get('tool_error') is None

def test_execute_tool_internal_call_get_details_error(nodes_module, state_module):
    initial_state = create_full_initial_state("Return from NF", state_module) # Use helper
    initial_state['intent']="initiate_return"
    initial_state['order_id']="NF"
    initial_state['next_node']="handle_return_step_1"

    mock_error = {"error": "Order ID 'NF' not found."}
    MockGetOrderDetails.invoke.return_value = mock_error
    result_state = nodes_module.execute_tool(initial_state)
    MockGetOrderDetails.invoke.assert_called_once_with({"order_id": "NF"})
    assert result_state['api_response'] == mock_error
    assert result_state['tool_error'] == "Order ID 'NF' not found."

def test_execute_tool_internal_call_submit_return_success(nodes_module, state_module):
    initial_state = create_full_initial_state("Reason is X", state_module) # Use helper
    initial_state.update({
        "intent": "return_reason_provided",
        "order_id": "ORD789",
        "item_sku_to_return": "SKU1",
        "return_reason": "Reason is X",
        "next_node": "execute_tool" # Correct marker for this path
    })
    mock_return_resp = {"return_id": "RET123", "message": "Success"}
    MockInitiateReturn.invoke.return_value = mock_return_resp

    result_state = nodes_module.execute_tool(initial_state)

    MockInitiateReturn.invoke.assert_called_once_with({"order_id": "ORD789", "sku": "SKU1", "reason": "Reason is X"})
    assert result_state['api_response'] == mock_return_resp
    assert result_state.get('tool_error') is None
    assert result_state.get('item_sku_to_return') is None # State cleared
    assert result_state.get('return_reason') is None
    assert result_state.get('available_return_items') is None

def test_execute_tool_internal_call_submit_return_error(nodes_module, state_module):
    initial_state = create_full_initial_state("Reason is R", state_module) # Use helper
    initial_state.update({
        "intent":"return_reason_provided",
        "order_id":"ORD",
        "item_sku_to_return":"S",
        "return_reason":"R",
        "next_node":"execute_tool" # Correct marker
    })
    mock_error = {"error": "Return failed"}
    MockInitiateReturn.invoke.return_value = mock_error
    result_state = nodes_module.execute_tool(initial_state)

    MockInitiateReturn.invoke.assert_called_once_with({"order_id": "ORD", "sku": "S", "reason": "R"})
    assert result_state['api_response'] == mock_error
    assert result_state['tool_error'] == "Return failed"
    # Ensure state NOT cleared on error
    # We check the result dict, not the input dict (which isn't modified)
    assert result_state.get('item_sku_to_return') is None # This node doesn't preserve these on error


# --- handle_multi_turn_return Tests ---

def test_handle_multi_turn_step1_details_ok(nodes_module, state_module):
    items = [{"sku": "S1", "name": "N1"}, {"sku": "S2", "name": "N2"}]
    initial_state = create_full_initial_state("Return ORD789", state_module)
    initial_state.update({
        "intent": "initiate_return",
        "order_id": "ORD789",
        "api_response": {"order_id": "ORD789", "items": items, "delivered": True},
        "tool_error": None,
        "next_node": "handle_return_step_1", # Simulate classify setting this
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
        "tool_error":"Order not found",
        "next_node":"handle_return_step_1"
    })
     result_state = nodes_module.handle_multi_turn_return(initial_state)
     assert result_state['needs_clarification'] is False
     assert 'tool_error' not in result_state # Error should remain in state, not added again by this node
     assert result_state['next_node'] == "generate_response" # Route to generate error message

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
    assert result_state['next_node'] == "execute_tool" # Route to execute submission


# --- generate_response Tests ---

def test_generate_response_clarification(nodes_module, state_module):
    question = "Which SKU do you want to return?"
    initial_state = create_full_initial_state("", state_module) # Use helper
    initial_state.update({"needs_clarification": True, "clarification_question": question})
    result_state = nodes_module.generate_response(initial_state)
    assert len(result_state['messages']) == 1
    assert isinstance(result_state['messages'][0], AIMessage)
    assert result_state['messages'][0].content == question

def test_generate_response_tool_error(nodes_module, state_module):
    error_msg = "Order ID 'NF' not found."
    initial_state = create_full_initial_state("", state_module)
    initial_state['tool_error'] = error_msg
    result_state = nodes_module.generate_response(initial_state)
    assert f"I encountered an issue: {error_msg}" == result_state['messages'][0].content

# ... (rest of generate_response tests, ensuring state is created with helper
#      and LLM calls are mocked with mock_llm_instance) ...

def test_generate_response_api_return_failure_with_message(nodes_module, state_module):
     initial_state = create_full_initial_state("", state_module)
     initial_state.update({ "intent":"return_reason_provided", "api_response":{"message": "Return system offline.", "status": "Failed"} })
     result_state = nodes_module.generate_response(initial_state)
     assert result_state['messages'][0].content == 'There was an issue: Return system offline.'

def test_generate_response_api_return_failure_with_error(nodes_module, state_module):
     initial_state = create_full_initial_state("", state_module)
     initial_state.update({ "intent":"return_reason_provided", "api_response":{"error": "Item already returned."} })
     result_state = nodes_module.generate_response(initial_state)
     assert result_state['messages'][0].content == 'There was an issue: Item already returned.'


def test_generate_response_rag_success(nodes_module, state_module):
    query = "Return policy?"
    context = "Items can be returned within 30 days if unused."
    initial_state = create_full_initial_state(query, state_module)
    initial_state.update({
        "intent":"knowledge_base_query", "rag_context":context
    })
    mock_llm_response = AIMessage(content="You can return unused items within 30 days.")
    mock_llm_instance.invoke.return_value = mock_llm_response

    result_state = nodes_module.generate_response(initial_state)

    mock_llm_instance.invoke.assert_called_once()
    prompt_arg = mock_llm_instance.invoke.call_args[0][0]
    assert context in prompt_arg
    assert query in prompt_arg
    assert result_state['messages'][0].content == mock_llm_response.content

def test_generate_response_rag_llm_failure(nodes_module, state_module, mocker):
    logger_error = mocker.patch('logging.Logger.error')
    query = "Return policy?"
    context = "Items can be returned within 30 days if unused."
    initial_state = create_full_initial_state(query, state_module)
    initial_state.update({ "intent":"knowledge_base_query", "rag_context":context})
    mock_llm_instance.invoke.side_effect = Exception("LLM RAG synthesis failed")

    result_state = nodes_module.generate_response(initial_state)

    mock_llm_instance.invoke.assert_called_once()
    logger_error.assert_called_once()
    assert "had trouble formulating a final answer" in result_state['messages'][0].content


def test_generate_response_rag_no_context(nodes_module, state_module):
     initial_state = create_full_initial_state("query", state_module)
     initial_state.update({"intent":"knowledge_base_query", "rag_context":None})
     result_state = nodes_module.generate_response(initial_state)
     assert "couldn't find specific information" in result_state['messages'][0].content
     mock_llm_instance.invoke.assert_not_called()


def test_generate_response_unsupported(nodes_module, state_module):
     initial_state = create_full_initial_state("gibberish", state_module) # Use helper
     initial_state['intent'] = "unsupported"
     result_state = nodes_module.generate_response(initial_state)
     # Fix AttributeError by checking last_ai_message existence
     assert result_state['messages'][0].content == "I'm sorry, I can't assist with that specific request right now. I can help with order status, tracking, returns, and answer general questions from our FAQ."