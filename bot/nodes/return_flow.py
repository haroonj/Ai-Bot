# bot/nodes/return_flow.py
import logging
from typing import Dict, Any, List

from ..state import GraphState
from ..tools import get_order_details, initiate_return_request

logger = logging.getLogger(__name__)

def handle_return_step(state: GraphState) -> Dict[str, Any]:
    """
    Manages the multi-turn return process. It determines the current step based on
    the state and performs the necessary action: fetch details, ask for SKU,
    ask for reason, or submit the return.
    """
    intent = state.get("intent")
    order_id = state.get("order_id")
    available_items = state.get("available_return_items")
    selected_sku = state.get("item_sku_to_return")
    return_reason = state.get("return_reason")
    last_message = state["messages"][-1] if state["messages"] else None
    needs_clarification = state.get("needs_clarification", False)

    logger.info(f"Handling return step. Order: {order_id}, Items Fetched: {bool(available_items)}, "
                f"SKU Selected: {selected_sku}, Reason Given: {bool(return_reason)}, Needs Clar: {needs_clarification}")

    # --- Step 0/1: Fetch order details if not already done ---
    if not available_items and order_id and not needs_clarification: # Check needs_clarification to avoid re-fetching if just asked SKU
        logger.info(f"Fetching order details for return, Order ID: {order_id}")
        details_response = get_order_details.invoke({"order_id": order_id})

        if "error" in details_response:
            logger.warning(f"Error fetching details for return: {details_response['error']}")
            return {"tool_error": details_response["error"], "next_node": "generate_final_response"}

        if not details_response.get("delivered") or not details_response.get("items"):
            error_msg = (f"Order {order_id} is not marked as delivered yet."
                         if not details_response.get("delivered")
                         else f"No returnable items found for order {order_id}.")
            logger.warning(f"Order {order_id} not eligible for return: {error_msg}")
            return {"tool_error": error_msg, "next_node": "generate_final_response"}

        # Details fetched successfully, ask for SKU
        items: List[Dict[str, Any]] = details_response["items"]
        item_list_str = "\n".join([f"- {item.get('name', 'N/A')} (SKU: {item['sku']})" for item in items])
        question = (f"Okay, I found order {order_id} which is delivered and eligible for returns.\n"
                    f"Which item would you like to return? Please provide the SKU:\n{item_list_str}")
        logger.info(f"Asking user to select SKU for return from order {order_id}.")
        return {
            "available_return_items": items,
            "needs_clarification": True,
            "clarification_question": question,
            "next_node": "generate_final_response" # Generate the question
        }

    # --- Step 2: Process user's SKU selection ---
    elif available_items and not selected_sku and needs_clarification and hasattr(last_message, 'content'):
        user_input_sku = last_message.content.strip()
        logger.info(f"Processing user SKU input: '{user_input_sku}'")

        # Validate SKU against available items
        matched_item = next((item for item in available_items if item["sku"] == user_input_sku), None)

        if matched_item:
            logger.info(f"Valid SKU '{user_input_sku}' selected. Asking for reason.")
            question = (f"Got it, you want to return item '{matched_item.get('name', user_input_sku)}' (SKU: {user_input_sku}).\n"
                        f"Could you briefly tell me why you're returning it? (Optional, you can say 'skip' or just press Enter)")
            return {
                "item_sku_to_return": user_input_sku,
                "needs_clarification": True, # Still need reason
                "clarification_question": question,
                "next_node": "generate_final_response" # Generate the reason question
            }
        else:
            logger.warning(f"Invalid SKU '{user_input_sku}' provided by user.")
            item_list_str = "\n".join([f"- {item.get('name', 'N/A')} (SKU: {item['sku']})" for item in available_items])
            question = (f"Sorry, '{user_input_sku}' doesn't seem to match the SKUs in your order.\n"
                        f"Please provide one of the following SKUs:\n{item_list_str}")
            return {
                "item_sku_to_return": None, # Reset SKU
                "needs_clarification": True, # Still need SKU
                "clarification_question": question,
                "next_node": "generate_final_response" # Re-ask SKU
            }

    # --- Step 3: Process user's return reason (or skip) ---
    elif selected_sku and not return_reason and needs_clarification and hasattr(last_message, 'content'):
        user_input_reason = last_message.content.strip()
        final_reason = None
        if user_input_reason.lower() not in ["skip", ""]:
            final_reason = user_input_reason
        logger.info(f"Return reason provided (or skipped): '{final_reason}'. Preparing to submit.")

        # Now we have Order ID, SKU, and Reason (optional). Ready to submit.
        return {
            "return_reason": final_reason,
            "needs_clarification": False, # All info gathered
            "clarification_question": None,
            "next_node": "submit_return_request" # Go to submit node
        }

    # --- Step 4: Submit the return request ---
    # This logic is now moved to its own node 'submit_return_request' triggered by the graph edge

    # --- Handle unexpected states or direct calls ---
    elif selected_sku and return_reason and intent == "initiate_return": # Case where LLM provided all info initially
        logger.info("All return info seems present from initial classification. Proceeding to submit.")
        return {"needs_clarification": False, "next_node": "submit_return_request"}

    # --- Fallback / Error State ---
    logger.warning(f"Unexpected state in handle_return_step. Routing to generate response.")
    # Maybe preserve existing error if any?
    existing_error = state.get("tool_error")
    return {
        "needs_clarification": False,
        "clarification_question": None,
        "tool_error": existing_error or "Something went wrong during the return process. Please try again.",
        "next_node": "generate_final_response"
        }


def submit_return_request(state: GraphState) -> Dict[str, Any]:
    """
    Calls the initiate_return_request tool with the collected information.
    """
    order_id = state.get("order_id")
    sku = state.get("item_sku_to_return")
    reason = state.get("return_reason")
    api_response = None
    tool_error = None

    if order_id and sku:
        logger.info(f"Submitting return request. Order: {order_id}, SKU: {sku}, Reason: {reason}")
        api_response = initiate_return_request.invoke({"order_id": order_id, "sku": sku, "reason": reason})
        if "error" in api_response:
            tool_error = api_response["error"]
            logger.warning(f"Error submitting return request: {tool_error}")
        else:
            logger.info(f"Return request submitted successfully. Response: {api_response}")
            # Clear return-specific state on success
            return {
                "api_response": api_response,
                "tool_error": None,
                "item_sku_to_return": None,
                "return_reason": None,
                "available_return_items": None,
                "needs_clarification": False,
                "clarification_question": None,
                "next_node": "generate_final_response"
            }
    else:
        tool_error = "Missing Order ID or Item SKU when trying to submit return."
        logger.error(tool_error)

    # If error occurred or prerequisite missing
    return {
        "api_response": api_response, # Keep API response even if error for context
        "tool_error": tool_error,
        "next_node": "generate_final_response"
    }