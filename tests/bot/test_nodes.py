import os
import sys

import pytest
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from tests.conftest import mock_llm_instance

mock_llm_with_tools_instance = mock_llm_instance


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
    return state_module.GraphState(
        messages=[HumanMessage(content=user_query)],
        intent=None, order_id=None, item_sku_to_return=None, return_reason=None,
        needs_clarification=False, clarification_question=None, available_return_items=None,
        rag_context=None, api_response=None, tool_error=None, next_node=None,
    )


def test_classify_intent_tool_call_status(nodes_module, state_module):
    initial_state = create_full_initial_state("Status for 123?", state_module)
    mock_ai_response = AIMessage(
        content="",
        tool_calls=[{"name": "get_order_status", "args": {"order_id": "123"}, "id": "call_123"}]
    )
    mock_llm_with_tools_instance.invoke.return_value = mock_ai_response
    result_state = nodes_module.classify_intent(initial_state)
    mock_llm_with_tools_instance.invoke.assert_called_once_with(initial_state['messages'])
    assert result_state['intent'] == "get_order_status"  # Should now pass
    assert result_state['order_id'] == "123"
    assert result_state['next_node'] == "execute_tool"
    assert len(initial_state['messages']) == 2
    assert initial_state['messages'][1] is mock_ai_response


def test_classify_intent_tool_call_kb(nodes_module, state_module):
    initial_state = create_full_initial_state("How do returns work?", state_module)
    mock_ai_response = AIMessage(
        content="",
        tool_calls=[{"name": "knowledge_base_lookup", "args": {"query": "How do returns work?"}, "id": "call_kb"}]
    )
    mock_llm_with_tools_instance.invoke.return_value = mock_ai_response
    result_state = nodes_module.classify_intent(initial_state)
    mock_llm_with_tools_instance.invoke.assert_called_once_with(initial_state['messages'])
    assert result_state['intent'] == "knowledge_base_query"  # Should now pass
    assert result_state['next_node'] == "execute_tool"


def test_classify_intent_multi_turn_return_sku_provided_no_match(nodes_module, state_module):
    initial_state = state_module.GraphState(
        messages=[HumanMessage(content="start"), AIMessage(content="Which SKU?"), HumanMessage(content="WRONG_SKU")],
        intent="initiate_return", needs_clarification=True,
        available_return_items=[{"sku": "ITEM004", "name": "A"}],
        order_id="ORD_TEST", rag_context=None, api_response=None, tool_error=None, next_node=None,
        item_sku_to_return=None, return_reason=None
    )
    mock_ai_response = AIMessage(content="")
    mock_llm_with_tools_instance.invoke.return_value = mock_ai_response
    result_state = nodes_module.classify_intent(initial_state)
    mock_llm_with_tools_instance.invoke.assert_called_once_with(initial_state['messages'])
    assert 'intent' not in result_state or result_state['intent'] == initial_state['intent']
    assert result_state.get('item_sku_to_return') is None  # Correct
    assert result_state['needs_clarification'] is True  # Correct
    assert "doesn't seem to match" in result_state['clarification_question']  # Correct
    assert result_state['next_node'] == "generate_response"  # Correct


def test_execute_tool_via_tool_call_status_success(nodes_module, state_module):
    from tests.conftest import MockGetOrderStatus_Func
    tool_call_message = AIMessage(content="", tool_calls=[
        {"name": "get_order_status", "args": {"order_id": "123"}, "id": "call_1"}])
    initial_state = create_full_initial_state("Status 123?", state_module)
    initial_state['messages'].append(tool_call_message)
    initial_state['intent'] = "get_order_status"
    initial_state['order_id'] = "123"

    mock_api_response = {"order_id": "123", "status": "Delivered"}
    MockGetOrderStatus_Func.invoke.return_value = mock_api_response

    result_state = nodes_module.execute_tool(initial_state)

    MockGetOrderStatus_Func.invoke.assert_called_once_with({"order_id": "123"})
    assert result_state['api_response'] == mock_api_response
    assert result_state.get('tool_error') is None
    assert len(initial_state['messages']) == 3
    assert isinstance(initial_state['messages'][2], ToolMessage)


def test_execute_tool_via_tool_call_status_api_error(nodes_module, state_module):
    from tests.conftest import MockGetOrderStatus_Func
    tool_call_message = AIMessage(content="",
                                  tool_calls=[{"name": "get_order_status", "args": {"order_id": "NF"}, "id": "call_2"}])
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


def test_execute_tool_via_tool_call_kb_success(nodes_module, state_module):
    from tests.conftest import MockKBLookup_Func
    tool_call_message = AIMessage(content="", tool_calls=[
        {"name": "knowledge_base_lookup", "args": {"query": "Returns policy?"}, "id": "call_3"}])
    initial_state = create_full_initial_state("Returns policy?", state_module)
    initial_state['messages'].append(tool_call_message)
    initial_state['intent'] = "knowledge_base_query"

    mock_kb_context = "Return within 30 days."
    MockKBLookup_Func.invoke.return_value = mock_kb_context

    result_state = nodes_module.execute_tool(initial_state)

    MockKBLookup_Func.invoke.assert_called_once_with({"query": "Returns policy?"})
    assert result_state['rag_context'] == mock_kb_context
    assert result_state.get('tool_error') is None
    assert len(initial_state['messages']) == 3
    assert isinstance(initial_state['messages'][2], ToolMessage)


def test_execute_tool_explicit_kb_lookup(nodes_module, state_module):
    from tests.conftest import MockKBLookup_Func
    initial_state = create_full_initial_state("Info on shipping?", state_module)
    initial_state['intent'] = "knowledge_base_query"
    mock_kb_context = "Shipping takes 3-5 days."
    MockKBLookup_Func.invoke.return_value = mock_kb_context
    result_state = nodes_module.execute_tool(initial_state)
    MockKBLookup_Func.invoke.assert_called_once_with({"query": "Info on shipping?"})
    assert result_state['rag_context'] == mock_kb_context
    assert result_state.get('tool_error') is None
    assert len(initial_state['messages']) == 1


def test_execute_tool_explicit_kb_lookup_no_query(nodes_module, state_module):
    from tests.conftest import MockKBLookup_Func
    initial_state = state_module.GraphState(messages=[], intent="knowledge_base_query")
    result_state = nodes_module.execute_tool(initial_state)
    MockKBLookup_Func.invoke.assert_not_called()
    assert result_state.get('rag_context') is None
    assert "Could not find user query" in result_state['tool_error']


def test_execute_tool_internal_call_get_details_success(nodes_module, state_module):
    from tests.conftest import MockGetOrderDetails_Func
    initial_state = create_full_initial_state("Return from 789", state_module)
    initial_state['intent'] = "initiate_return"
    initial_state['order_id'] = "789"
    initial_state['next_node'] = "handle_return_step_1"
    mock_details = {"order_id": "789", "items": [{"sku": "S1"}], "delivered": True}
    MockGetOrderDetails_Func.invoke.return_value = mock_details
    result_state = nodes_module.execute_tool(initial_state)
    MockGetOrderDetails_Func.invoke.assert_called_once_with({"order_id": "789"})
    assert result_state['api_response'] == mock_details
    assert result_state.get('tool_error') is None


def test_execute_tool_internal_call_get_details_error(nodes_module, state_module):
    from tests.conftest import MockGetOrderDetails_Func
    initial_state = create_full_initial_state("Return from NF", state_module)
    initial_state['intent'] = "initiate_return"
    initial_state['order_id'] = "NF"
    initial_state['next_node'] = "handle_return_step_1"
    mock_error = {"error": "Order ID 'NF' not found."}
    MockGetOrderDetails_Func.invoke.return_value = mock_error
    result_state = nodes_module.execute_tool(initial_state)
    MockGetOrderDetails_Func.invoke.assert_called_once_with({"order_id": "NF"})
    assert result_state['api_response'] == mock_error
    assert result_state['tool_error'] == "Order ID 'NF' not found."


def test_execute_tool_internal_call_submit_return_success(nodes_module, state_module):
    from tests.conftest import MockInitiateReturn_Func
    initial_state = create_full_initial_state("Reason is X", state_module)
    initial_state.update({
        "intent": "return_reason_provided", "order_id": "789",
        "item_sku_to_return": "SKU1", "return_reason": "Reason is X",
        "next_node": "execute_tool"
    })
    mock_return_resp = {"return_id": "RET123", "message": "Success"}
    MockInitiateReturn_Func.invoke.return_value = mock_return_resp
    result_state = nodes_module.execute_tool(initial_state)
    MockInitiateReturn_Func.invoke.assert_called_once_with(
        {"order_id": "789", "sku": "SKU1", "reason": "Reason is X"})
    assert result_state['api_response'] == mock_return_resp
    assert result_state.get('tool_error') is None
    assert result_state.get('item_sku_to_return') is None
    assert result_state.get('return_reason') is None
    assert result_state.get('available_return_items') is None


def test_execute_tool_internal_call_submit_return_error(nodes_module, state_module):
    from tests.conftest import MockInitiateReturn_Func
    initial_state = create_full_initial_state("Reason is R", state_module)
    initial_state.update({
        "intent": "return_reason_provided", "order_id": "ORD",
        "item_sku_to_return": "S", "return_reason": "R",
        "next_node": "execute_tool"
    })
    mock_error = {"error": "Return failed"}
    MockInitiateReturn_Func.invoke.return_value = mock_error
    result_state = nodes_module.execute_tool(initial_state)
    MockInitiateReturn_Func.invoke.assert_called_once_with({"order_id": "ORD", "sku": "S", "reason": "R"})
    assert result_state['api_response'] == mock_error
    assert result_state['tool_error'] == "Return failed"
    assert 'item_sku_to_return' not in result_state or result_state.get('item_sku_to_return') is not None


def test_handle_multi_turn_step1_details_ok(nodes_module, state_module):
    items = [{"sku": "S1", "name": "N1"}, {"sku": "S2", "name": "N2"}]
    initial_state = create_full_initial_state("Return 789", state_module)
    initial_state.update({
        "intent": "initiate_return", "order_id": "789",
        "api_response": {"order_id": "789", "items": items, "delivered": True},
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
        "intent": "initiate_return", "order_id": "NF",
        "api_response": {"error": "Order not found"},
        "tool_error": "Order not found", "next_node": "handle_return_step_1"
    })
    result_state = nodes_module.handle_multi_turn_return(initial_state)
    assert result_state['needs_clarification'] is False
    assert result_state['tool_error'] == 'Order not found'
    assert result_state['next_node'] == "generate_response"


def test_handle_multi_turn_step1_details_no_items(nodes_module, state_module):
    initial_state = create_full_initial_state("Return 123", state_module)
    initial_state.update({
        "intent": "initiate_return", "order_id": "123",
        "api_response": {"order_id": "123", "items": [], "delivered": True},
        "tool_error": None, "next_node": "handle_return_step_1"
    })
    result_state = nodes_module.handle_multi_turn_return(initial_state)
    assert result_state['needs_clarification'] is False
    assert "couldn't find any returnable items" in result_state['tool_error']
    assert result_state['next_node'] == "generate_response"


def test_handle_multi_turn_step2_sku_selected(nodes_module, state_module):
    initial_state = create_full_initial_state("ITEM004", state_module)
    initial_state.update({
        "intent": "return_item_selection", "item_sku_to_return": "SKU1",
        "tool_error": None, "next_node": "handle_return_step_2"
    })
    result_state = nodes_module.handle_multi_turn_return(initial_state)
    assert result_state['needs_clarification'] is True
    assert "why you're returning it?" in result_state['clarification_question']
    assert result_state['next_node'] == "generate_response"


def test_handle_multi_turn_step3_reason_provided(nodes_module, state_module):
    initial_state = create_full_initial_state("It broke", state_module)
    initial_state.update({
        "intent": "return_reason_provided", "return_reason": "Broken",
        "tool_error": None, "next_node": "handle_return_step_3"
    })
    result_state = nodes_module.handle_multi_turn_return(initial_state)
    assert result_state['needs_clarification'] is False
    assert result_state.get('clarification_question') is None
    assert result_state['next_node'] == "execute_tool"


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
    initial_state.update(
        {"intent": "return_reason_provided", "api_response": {"message": "Return system offline.", "status": "Failed"}})
    update_dict = nodes_module.generate_response(initial_state)
    assert update_dict == {}
    final_message = initial_state["messages"][-1]
    assert isinstance(final_message, AIMessage)
    assert final_message.content == 'There was an issue: Return system offline.'


def test_generate_response_api_return_failure_with_error(nodes_module, state_module):
    initial_state = create_full_initial_state("", state_module)
    initial_state.update({"intent": "return_reason_provided", "api_response": {"error": "Item already returned."}})
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
        "intent": "knowledge_base_query", "rag_context": context
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
    initial_state.update({"intent": "knowledge_base_query", "rag_context": context})
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
    initial_state.update({"intent": "knowledge_base_query", "rag_context": None})
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
