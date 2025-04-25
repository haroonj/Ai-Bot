# bot/nodes/classification.py
import logging
from typing import Dict, Any

from langchain_core.messages import AIMessage

from ..state import GraphState
from ..tools import (
    llm_with_tools,
    get_order_status,
    get_tracking_info,
    get_order_details,
    initiate_return_request,
    knowledge_base_lookup,
)

logger = logging.getLogger(__name__)

def classify_intent(state: GraphState) -> Dict[str, Any]:
    """
    Classifies the user's intent based on the latest message and decides the next step.
    It may identify a direct tool call, a knowledge base query, or a step in an ongoing flow.
    """
    messages = state["messages"]
    if not messages:
        logger.warning("Classify intent called with no messages.")
        return {"intent": "clarification_needed", "next_node": "generate_final_response"}

    last_message = messages[-1]
    current_intent = state.get("intent")
    needs_clarification = state.get("needs_clarification", False)

    # --- Check for ongoing multi-turn flows first ---
    if needs_clarification and current_intent == "initiate_return":
        logger.debug("Ongoing return flow detected, routing to handle_return_step.")
        return {"next_node": "handle_return_step"}

    # --- Simple Greetings/Goodbyes ---
    if last_message.content.strip().lower() in {"hi", "hello", "hey"}:
        logger.debug("Intent classified as greeting.")
        return {
            "intent": "greeting",
            "next_node": "generate_final_response"
        }
    if last_message.content.strip().lower() in {"bye", "goodbye", "thanks bye"}:
        logger.debug("Intent classified as goodbye.")
        return {
            "intent": "goodbye",
            "next_node": "generate_final_response"
        }

    # --- Use LLM with tools to determine intent/next action ---
    logger.debug("Invoking LLM with tools for intent classification.")
    ai_response: AIMessage = llm_with_tools.invoke(messages)
    state["messages"].append(ai_response)

    tool_calls = ai_response.tool_calls
    updates: Dict[str, Any] = {
        "needs_clarification": False,
        "clarification_question": None,
        "tool_error": None,
        # --- Add a temporary field to pass the AI response to the next node if needed ---
        # This allows execute_tool_call to find the tool calls without polluting history
        "latest_ai_response": ai_response
        # --- End temporary field ---
    }

    if tool_calls:
        first_call = tool_calls[0]
        tool_name = first_call['name']
        tool_args = first_call['args']
        logger.info(f"LLM suggested tool call: {tool_name} with args: {tool_args}")

        extracted_order_id = tool_args.get('order_id')
        if extracted_order_id:
            updates["order_id"] = extracted_order_id

        if tool_name == get_order_status.name:
            updates["intent"] = "get_order_status"
            updates["next_node"] = "execute_tool_call"
        elif tool_name == get_tracking_info.name:
            updates["intent"] = "get_tracking_info"
            updates["next_node"] = "execute_tool_call"
        elif tool_name == knowledge_base_lookup.name:
            updates["intent"] = "knowledge_base_query"
            updates["next_node"] = "execute_rag_lookup"
        elif tool_name == get_order_details.name:
            updates["intent"] = "initiate_return"
            updates["next_node"] = "handle_return_step"
        elif tool_name == initiate_return_request.name:
            updates["intent"] = "initiate_return"
            updates["item_sku_to_return"] = tool_args.get('sku') or state.get("item_sku_to_return")
            updates["return_reason"] = tool_args.get('reason') or state.get("return_reason")
            updates["next_node"] = "handle_return_step"
        else:
            logger.warning(f"LLM suggested unsupported tool: {tool_name}")
            updates["intent"] = "unsupported"
            updates["api_response"] = {"message": f"I understood you want to use '{tool_name}', but I can't handle that specific action."}
            updates["next_node"] = "generate_final_response"
            updates.pop("latest_ai_response", None) # Remove temporary field if not routing to execution

        return updates

    # --- No tool call suggested ---
    updates.pop("latest_ai_response", None) # Remove temporary field
    if ai_response.content.strip():
        logger.debug("LLM provided direct response content.")
        updates.update({
            "intent": "knowledge_base_query", # Or just 'general_query'?
            "next_node": "generate_final_response",
            "final_llm_response": ai_response.content
        })
        return updates

    # --- Fallback ---
    logger.warning("LLM provided no tool calls or content. Classifying as unsupported.")
    updates.pop("latest_ai_response", None) # Remove temporary field
    updates.update({
        "intent": "unsupported",
        "api_response": {"message": "I had trouble understanding that request."},
        "next_node": "generate_final_response"
    })
    return updates