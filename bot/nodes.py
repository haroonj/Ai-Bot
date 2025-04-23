import logging
from typing import Dict, Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from .llm import llm
from .state import GraphState
from .tools import (
    llm_with_tools,
    get_order_status,
    get_tracking_info,
    get_order_details,
    initiate_return_request,
    knowledge_base_lookup,
    available_tools
)

logger = logging.getLogger(__name__)


def classify_intent(state: GraphState) -> Dict[str, Any]:
    messages = state["messages"]
    if not messages:
        logger.warning("classify_intent called with empty messages state.")
        return {"intent": "unsupported", "api_response": {"message": "I need a message to understand your request."},
                "next_node": "generate_response"}

    last_message = messages[-1]
    logger.info(f"Classifying intent for message: '{last_message.content}'")

    try:
        ai_response = llm_with_tools.invoke(messages)
        state["messages"].append(ai_response)

        tool_calls = ai_response.tool_calls
        logger.debug(f"LLM tool calls: {tool_calls}")

        if tool_calls:
            first_call = tool_calls[0]
            tool_name = first_call['name']
            tool_args = first_call['args']
            tool_id = first_call['id']

            extracted_order_id = tool_args.get('order_id')
            extracted_sku = tool_args.get('sku')
            extracted_reason = tool_args.get('reason')

            updates: Dict[str, Any] = {
                "order_id": extracted_order_id or state.get("order_id"),
                "item_sku_to_return": extracted_sku or state.get("item_sku_to_return"),
                "return_reason": extracted_reason or state.get("return_reason"),
                "needs_clarification": False,
                "clarification_question": None,
                "tool_error": None,
                "next_node": "execute_tool"
            }

            if tool_name == get_order_status.name:
                updates["intent"] = "get_order_status"
            elif tool_name == get_tracking_info.name:
                updates["intent"] = "get_tracking_info"
            elif tool_name == knowledge_base_lookup.name:
                updates["intent"] = "knowledge_base_query"
            elif tool_name == initiate_return_request.name:
                updates["intent"] = "initiate_return"
            elif tool_name == get_order_details.name:
                updates["intent"] = "initiate_return"
                updates["next_node"] = "handle_return_step_1"
            else:
                logger.warning(f"LLM called unknown tool: {tool_name}")
                updates["intent"] = "unsupported"
                updates["api_response"] = {
                    "message": f"I encountered an issue understanding how to use the tool: {tool_name}."}
                updates["next_node"] = "generate_response"

            logger.info(f"Intent classified as '{updates.get('intent')}' with tool call '{tool_name}'.")
            return updates

        else:
            logger.info("No tool call detected by LLM. Assuming KB query or multi-turn context.")
            current_intent = state.get("intent")
            needs_clarification = state.get("needs_clarification")

            if current_intent == "initiate_return" and needs_clarification:
                potential_sku = last_message.content.strip().upper()
                available_items = state.get("available_return_items", [])
                matched_item = next((item for item in available_items if item["sku"] == potential_sku), None)

                if matched_item:
                    logger.info(f"Identified SKU '{potential_sku}' for return.")
                    return {
                        "item_sku_to_return": potential_sku,
                        "needs_clarification": False,
                        "clarification_question": None,
                        "intent": "return_item_selection",
                        "next_node": "handle_return_step_2"
                    }
                else:
                    logger.warning(f"User provided input '{potential_sku}', but it doesn't match available SKUs.")
                    item_skus_str = ', '.join(
                        [item['sku'] for item in available_items]) if available_items else "any available items"
                    return {
                        "clarification_question": f"Sorry, '{potential_sku}' doesn't seem to match the items in your order. Please provide one of the following SKUs: {item_skus_str}",
                        "needs_clarification": True,
                        "next_node": "generate_response"
                    }

            elif current_intent == "return_item_selection" and needs_clarification:
                logger.info("Received potential return reason.")
                return {
                    "return_reason": last_message.content,
                    "needs_clarification": False,
                    "clarification_question": None,
                    "intent": "return_reason_provided",
                    "next_node": "handle_return_step_3"
                }

            logger.info("No active multi-turn context. Treating as knowledge base query.")
            return {
                "intent": "knowledge_base_query",
                "needs_clarification": False,
                "clarification_question": None,
                "next_node": "execute_tool",
                "tool_error": None,
            }

    except Exception as e:
        logger.error(f"Error during intent classification: {e}", exc_info=True)
        return {
            "intent": "unsupported",
            "api_response": {"message": "I had trouble understanding your request."},
            "next_node": "generate_response"
        }


def execute_tool(state: GraphState) -> Dict[str, Any]:
    messages = state["messages"]
    last_ai_message = messages[-1] if messages and isinstance(messages[-1], AIMessage) else None
    tool_calls = last_ai_message.tool_calls if last_ai_message and hasattr(last_ai_message, 'tool_calls') else []
    intent = state.get("intent")
    logger.info(f"Executing tool for intent: {intent}")

    api_response = None
    rag_context = None
    tool_error = None
    tool_message_content = "Tool execution did not proceed as expected."

    if tool_calls:
        call = tool_calls[0]
        tool_name = call['name']
        tool_args = call['args']
        tool_id = call['id']
        logger.info(f"Executing tool '{tool_name}' via LLM call with args: {tool_args}")

        try:
            tool_func = next((t for t in available_tools if t.name == tool_name), None)
            if tool_func:
                result = tool_func.invoke(tool_args)
                logger.info(f"Tool '{tool_name}' executed successfully.")

                if tool_name == knowledge_base_lookup.name:
                    if isinstance(result, dict) and "error" in result:
                        tool_error = result["error"]
                        rag_context = None
                        tool_message_content = f"Tool execution returned an error: {tool_error}"
                    elif isinstance(result, str) and result:
                        rag_context = result
                        tool_message_content = f"Successfully looked up knowledge base. Context length: {len(rag_context)}"
                    else:
                        rag_context = None
                        tool_message_content = "Knowledge base lookup returned no specific results."
                else:
                    api_response = result if isinstance(result, dict) else {"result": str(result)}
                    if "error" in api_response:
                        tool_error = api_response["error"]
                        tool_message_content = f"Tool execution returned an error: {tool_error}"
                    else:
                        tool_message_content = f"Successfully called API tool {tool_name}."
            else:
                tool_error = f"Tool '{tool_name}' not found in available tools."
                logger.error(tool_error)
                tool_message_content = f"Error: Tool '{tool_name}' is not available."

        except Exception as e:
            tool_error = f"Failed to execute tool '{tool_name}': {e}"
            logger.error(tool_error, exc_info=True)
            tool_message_content = f"An unexpected error occurred while executing tool {tool_name}."

        state["messages"].append(ToolMessage(content=tool_message_content, tool_call_id=tool_id))

    elif intent == "knowledge_base_query":
        logger.info("Executing RAG tool explicitly for knowledge_base_query intent.")
        last_user_message = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        if last_user_message:
            try:
                rag_context_result = knowledge_base_lookup.invoke({"query": last_user_message})
                if isinstance(rag_context_result, dict) and "error" in rag_context_result:
                    tool_error = rag_context_result["error"]
                    rag_context = None
                elif isinstance(rag_context_result, str) and rag_context_result:
                    rag_context = rag_context_result
                    logger.info(f"Explicit RAG execution successful.")
                else:
                    rag_context = None
                    logger.warning("Explicit RAG lookup returned no results.")

            except Exception as e:
                tool_error = f"Failed to execute explicit RAG lookup: {e}"
                logger.error(tool_error, exc_info=True)
        else:
            tool_error = "Could not find user query for explicit RAG lookup."
            logger.warning(tool_error)

    elif intent == "initiate_return" and state.get("next_node") == "handle_return_step_1":
        order_id = state.get("order_id")
        if order_id:
            logger.info(f"Executing get_order_details for return step 1, order: {order_id}")
            api_response = get_order_details.invoke({"order_id": order_id})
            if "error" in api_response:
                tool_error = api_response["error"]
            elif not api_response.get("delivered") or not api_response.get("items"):
                if not api_response.get("delivered"):
                    tool_error = f"Order {order_id} is not marked as delivered yet, cannot initiate return."
                else:
                    tool_error = f"No returnable items found for order {order_id} (items might be ineligible)."
                logger.warning(f"Return check failed for {order_id}: {tool_error}")
        else:
            tool_error = "Order ID missing for initiating return."
            logger.warning(tool_error)

    elif intent == "return_reason_provided" and state.get("next_node") == "execute_tool":
        order_id = state.get("order_id")
        sku = state.get("item_sku_to_return")
        reason = state.get("return_reason")
        if order_id and sku:
            logger.info(f"Executing initiate_return_request for return step 3. Order: {order_id}, SKU: {sku}")
            api_response = initiate_return_request.invoke({"order_id": order_id, "sku": sku, "reason": reason})
            if "error" in api_response:
                tool_error = api_response["error"]
        else:
            tool_error = "Missing Order ID or Item SKU for submitting return."
            logger.warning(tool_error)

    updates = {
        "api_response": api_response,
        "rag_context": rag_context,
        "tool_error": tool_error,
        "next_node": None
    }

    if intent == "return_reason_provided" and not tool_error and api_response and "return_id" in api_response:
        logger.info(f"Return successful (ID: {api_response.get('return_id')}), clearing return state.")
        updates.update({
            "item_sku_to_return": None,
            "return_reason": None,
            "available_return_items": None,
        })

    return updates


def handle_multi_turn_return(state: GraphState) -> Dict[str, Any]:
    intent = state.get("intent")
    tool_error = state.get("tool_error")
    api_response = state.get("api_response")
    order_id = state.get("order_id")
    expected_step_marker = state.get("next_node")

    logger.info(
        f"Handling multi-turn return. Intent: {intent}, Error: {tool_error}, Expected Step Marker: {expected_step_marker}")

    if tool_error and expected_step_marker == "handle_return_step_1":
        logger.warning(f"Error occurred getting order details: {tool_error}")
        return {"tool_error": tool_error, "needs_clarification": False, "clarification_question": None,
                "next_node": "generate_response"}

    if intent == "initiate_return" and not tool_error and api_response and api_response.get("items"):
        items = api_response["items"]
        logger.info(f"Order {order_id} details fetched. Items available for return: {items}")
        item_list_str = "\n".join([f"- {item['name']} (SKU: {item['sku']})" for item in items])
        question = f"Okay, I found order {order_id}. Which item would you like to return? Please provide the SKU:\n{item_list_str}"
        return {
            "available_return_items": items,
            "needs_clarification": True,
            "clarification_question": question,
            "next_node": "generate_response"
        }
    elif intent == "initiate_return" and expected_step_marker == "handle_return_step_1" and not tool_error:
        error_msg = (api_response.get(
            "error") if api_response else None) or f"Sorry, I couldn't find any returnable items for order {order_id}, or the order is not eligible for return yet."
        logger.warning(f"No returnable items identified or error in details for {order_id}: {error_msg}")
        return {"tool_error": error_msg, "needs_clarification": False, "next_node": "generate_response"}


    elif intent == "return_item_selection" and expected_step_marker == "handle_return_step_2":
        sku = state.get("item_sku_to_return")
        logger.info(f"Item SKU {sku} selected by user. Asking for reason.")
        question = f"Got it, you want to return item {sku}. Could you briefly tell me why you're returning it? (Optional, press Enter or say 'skip' to skip)"
        return {
            "needs_clarification": True,
            "clarification_question": question,
            "next_node": "generate_response"
        }

    elif intent == "return_reason_provided" and expected_step_marker == "handle_return_step_3":
        logger.info("Reason provided (or skipped). Ready to submit return request.")
        return {"needs_clarification": False, "clarification_question": None, "next_node": "execute_tool"}

    logger.warning(
        f"Unexpected state in handle_multi_turn_return. Intent: {intent}, Marker: {expected_step_marker}. Routing to generate_response.")
    return {"tool_error": tool_error, "needs_clarification": False, "clarification_question": None,
            "next_node": "generate_response"}


def generate_response(state: GraphState) -> Dict[str, Any]:
    messages = state["messages"]
    intent = state.get("intent")
    api_response = state.get("api_response")
    rag_context = state.get("rag_context")
    tool_error = state.get("tool_error")
    needs_clarification = state.get("needs_clarification")
    clarification_question = state.get("clarification_question")

    logger.info(f"Generating response. Intent: {intent}, Error: {tool_error}, Clarification: {needs_clarification}")

    response_text = ""

    if needs_clarification and clarification_question:
        response_text = clarification_question
        logger.info(f"Generated clarification response: {response_text}")

    elif tool_error:
        response_text = f"I encountered an issue: {tool_error}"
        logger.warning(f"Generated error response: {response_text}")

    elif intent == "knowledge_base_query":
        if rag_context:
            logger.info("Generating response from RAG context.")
            last_user_message_content = "your question"
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    last_user_message_content = msg.content
                    break

            prompt = f"""Based *only* on the following information from our knowledge base:
            --- Knowledge Base Context ---
            {rag_context}
            --- End Context ---
            Answer the user's question: "{last_user_message_content}"
            Provide a concise and helpful answer based *strictly* on the provided context.
            If the context doesn't contain the answer, state that you couldn't find the specific detail in the knowledge base. Do not make up information.
            """
            try:
                ai_message = llm.invoke(prompt)
                response_text = ai_message.content
            except Exception as e:
                logger.error(f"LLM invocation failed during RAG response generation: {e}", exc_info=True)
                response_text = "I found some information in our knowledge base, but had trouble formulating a final answer. Could you perhaps rephrase your question?"
        else:
            response_text = "I looked in our knowledge base, but couldn't find specific information about that."
            logger.warning("Generating response, but RAG context was empty or lookup failed.")

    elif api_response:
        logger.info(f"Generating response from API result: {api_response}")
        if intent == "get_order_status":
            response_text = f"The status for order {api_response.get('order_id', 'N/A')} is: {api_response.get('status', 'Unknown')}."
        elif intent == "get_tracking_info":
            if api_response.get('tracking_number'):
                response_text = (f"Tracking for order {api_response.get('order_id', 'N/A')}: "
                                 f"Number {api_response.get('tracking_number')}, "
                                 f"Carrier: {api_response.get('carrier', 'N/A')}, "
                                 f"Status: {api_response.get('status', 'N/A')}.")
            else:
                status_msg = api_response.get('status', 'Unavailable')
                response_text = (
                    f"Tracking information is not yet available for order {api_response.get('order_id', 'N/A')}. "
                    f"Current status: {status_msg}.")
        elif intent == "return_reason_provided" and api_response.get("return_id"):
            response_text = f"Success! {api_response.get('message', 'Return initiated.')} Your return ID is {api_response.get('return_id')}."
        else:
            error_msg = api_response.get('error') or api_response.get('message')
            if error_msg:
                response_text = f"There was an issue: {error_msg}"
            else:
                response_text = f"I received the following details: {str(api_response)}"

    elif intent == "greeting":
        response_text = "Hello! How can I help you with your order or answer your questions?"
    elif intent == "goodbye":
        response_text = "Goodbye! Let me know if you need anything else."

    else:
        logger.info("No specific response path matched, generating fallback response.")
        response_text = "I'm sorry, I can't assist with that specific request right now. I can help with order status, tracking, returns, and answer general questions from our FAQ."
        last_ai_message = messages[-1] if messages and isinstance(messages[-1], AIMessage) else None
        if last_ai_message and last_ai_message.content and not (
                hasattr(last_ai_message, 'tool_calls') and last_ai_message.tool_calls):
            response_text = last_ai_message.content

    logger.info(f"Final generated response: {response_text}")

    state["messages"].append(AIMessage(content=response_text))
    return {}
