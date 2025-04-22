# bot/nodes.py

import logging
from typing import Dict, Any, Optional, List
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from .state import GraphState, Intent
from .llm import llm, llm_with_tools
from .tools import (
    get_order_status,
    get_tracking_info,
    get_order_details,
    initiate_return_request,
    knowledge_base_lookup,
    available_tools # Used for identifying tool calls
)

logger = logging.getLogger(__name__)

# --- Node Functions ---

def classify_intent(state: GraphState) -> Dict[str, Any]:
    """
    Classifies the user's intent based on the latest message and conversation history.
    Uses LLM with tool calling to determine the primary action or if clarification is needed.
    Also extracts relevant entities like order_id.
    """
    messages = state["messages"]
    last_message = messages[-1]
    logger.info(f"Classifying intent for message: '{last_message.content}'")

    # Prepare messages for the LLM (can include history for context)
    # For simplicity, just using the last message might be enough for tool calling
    # If more context needed: llm_input_messages = messages

    try:
        # Invoke the LLM with tools bound
        ai_response = llm_with_tools.invoke(messages)
        state["messages"].append(ai_response) # Add AI's thought process/tool calls to history

        tool_calls = ai_response.tool_calls
        logger.debug(f"LLM tool calls: {tool_calls}")

        if tool_calls:
            # --- Tool Call Detected ---
            # For simplicity, handle the first tool call if multiple exist
            first_call = tool_calls[0]
            tool_name = first_call['name']
            tool_args = first_call['args']
            tool_id = first_call['id'] # Keep track for ToolMessage later

            extracted_order_id = tool_args.get('order_id')
            extracted_sku = tool_args.get('sku')
            extracted_reason = tool_args.get('reason')

            updates: Dict[str, Any] = {
                "order_id": extracted_order_id or state.get("order_id"), # Persist if already known
                "item_sku_to_return": extracted_sku or state.get("item_sku_to_return"),
                "return_reason": extracted_reason or state.get("return_reason"),
                "needs_clarification": False, # Reset clarification flags
                "clarification_question": None,
                 "tool_error": None, # Reset error on new intent
                 "next_node": "execute_tool" # Signal to run the tool
            }

            if tool_name == get_order_status.name:
                updates["intent"] = "get_order_status"
            elif tool_name == get_tracking_info.name:
                updates["intent"] = "get_tracking_info"
            # Direct KB tool call
            elif tool_name == knowledge_base_lookup.name:
                 updates["intent"] = "knowledge_base_query"
            # Return initiation might be triggered by keywords, not a direct tool call yet,
            # unless the user provided all info at once.
            elif tool_name == initiate_return_request.name:
                 updates["intent"] = "initiate_return" # Direct attempt if all info present
            elif tool_name == get_order_details.name:
                 # This tool is usually called internally during the return flow
                 # If called directly, treat it like starting a return
                 updates["intent"] = "initiate_return"
                 updates["next_node"] = "handle_return_step_1" # Go to return logic

            else:
                logger.warning(f"LLM called unknown tool: {tool_name}")
                updates["intent"] = "unsupported"
                updates["api_response"] = {"message": f"I encountered an issue understanding how to use the tool: {tool_name}."}
                updates["next_node"] = "generate_response" # Go straight to response generation

            logger.info(f"Intent classified as '{updates.get('intent')}' with tool call '{tool_name}'.")
            return updates

        else:
            # --- No Tool Call - Interpret Intent with LLM ---
            # Could be KB query, start of return, greeting, etc.
            # Use a separate LLM call or prompt engineering for this classification
            # For simplicity, let's assume KB query if no tool is called and it's not a return start
            logger.info("No tool call detected by LLM. Assuming KB query or multi-turn context.")

            # Check if this message relates to an ongoing multi-turn process (like returns)
            current_intent = state.get("intent")
            needs_clarification = state.get("needs_clarification")

            if current_intent == "initiate_return" and needs_clarification:
                # User might be providing the item SKU
                # Simple keyword check for demonstration
                # A more robust approach would use LLM to parse the item from the message
                # Or present choices and parse selection.
                # For now, assume the message *is* the SKU.
                potential_sku = last_message.content.strip().upper()
                available_items = state.get("available_return_items", [])
                matched_item = next((item for item in available_items if item["sku"] == potential_sku), None)

                if matched_item:
                     logger.info(f"Identified SKU '{potential_sku}' for return.")
                     return {
                        "item_sku_to_return": potential_sku,
                        "needs_clarification": False,
                        "clarification_question": None,
                        "intent": "return_item_selection", # Explicitly set intent
                        "next_node": "handle_return_step_2" # Move to next return step
                     }
                else:
                     logger.warning(f"User provided input '{potential_sku}', but it doesn't match available SKUs.")
                     # Ask again or fallback
                     return {
                         "clarification_question": f"Sorry, '{potential_sku}' doesn't seem to match the items in your order. Please provide one of the following SKUs: {', '.join([item['sku'] for item in available_items])}",
                         "needs_clarification": True, # Remain in clarification state
                         "next_node": "generate_response" # Ask clarification question
                     }

            elif current_intent == "return_item_selection" and needs_clarification:
                 # User might be providing the reason
                 logger.info("Received potential return reason.")
                 return {
                     "return_reason": last_message.content,
                     "needs_clarification": False,
                     "clarification_question": None,
                     "intent": "return_reason_provided", # Explicitly set intent
                     "next_node": "handle_return_step_3" # Move to final return step
                 }

            # Default to KB lookup if no other context applies
            logger.info("No active multi-turn context. Treating as knowledge base query.")
            return {
                "intent": "knowledge_base_query",
                "needs_clarification": False,
                "clarification_question": None,
                "next_node": "execute_tool", # Trigger RAG via execute_tool
                "tool_error": None,
            }

    except Exception as e:
        logger.error(f"Error during intent classification: {e}", exc_info=True)
        return {"intent": "unsupported", "api_response": {"message": "I had trouble understanding your request."}, "next_node": "generate_response"}


def execute_tool(state: GraphState) -> Dict[str, Any]:
    """Executes the appropriate tool based on the classified intent and tool calls."""
    messages = state["messages"]
    last_ai_message = messages[-1] if messages and isinstance(messages[-1], AIMessage) else None
    tool_calls = last_ai_message.tool_calls if last_ai_message else []
    intent = state.get("intent")
    logger.info(f"Executing tool for intent: {intent}")

    api_response = None
    rag_context = None
    tool_error = None
    tool_message_content = "Tool execution failed." # Default error message

    if tool_calls:
        # --- Execute based on LLM Tool Call ---
        call = tool_calls[0] # Handle first call
        tool_name = call['name']
        tool_args = call['args']
        tool_id = call['id']
        logger.info(f"Executing tool '{tool_name}' with args: {tool_args}")

        try:
            tool_func = next((t for t in available_tools if t.name == tool_name), None)
            if tool_func:
                # Execute the actual tool function
                result = tool_func.invoke(tool_args)
                logger.info(f"Tool '{tool_name}' executed successfully.")
                # Store based on tool type
                if tool_name == knowledge_base_lookup.name:
                    rag_context = result if isinstance(result, str) else str(result)
                    tool_message_content = f"Successfully looked up knowledge base. Context length: {len(rag_context)}" if rag_context else "Knowledge base lookup returned no results."
                else: # API tools
                    api_response = result if isinstance(result, dict) else {"result": str(result)}
                    # Check for errors returned *by the tool* (e.g., 404)
                    if "error" in api_response:
                        tool_error = api_response["error"]
                        tool_message_content = f"Tool execution returned an error: {tool_error}"
                    else:
                         tool_message_content = f"Successfully called API tool {tool_name}."

            else:
                tool_error = f"Tool '{tool_name}' not found in available tools."
                logger.error(tool_error)

        except Exception as e:
            tool_error = f"Failed to execute tool '{tool_name}': {e}"
            logger.error(tool_error, exc_info=True)
            tool_message_content = f"An unexpected error occurred while executing tool {tool_name}."

        # Append ToolMessage to history
        state["messages"].append(ToolMessage(content=tool_message_content, tool_call_id=tool_id))

    elif intent == "knowledge_base_query":
        # --- Execute RAG tool explicitly if intent is KB but no tool call occurred ---
        # This can happen if the classification logic defaults to KB
        logger.info("Executing RAG tool explicitly for knowledge_base_query intent.")
        last_user_message = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
        if last_user_message:
            try:
                rag_context = knowledge_base_lookup.invoke({"query": last_user_message})
                if isinstance(rag_context, dict) and "error" in rag_context: # Handle RAG error case
                    tool_error = rag_context["error"]
                    rag_context = None # Clear context on error
                elif not rag_context:
                    tool_error = "No relevant information found in the knowledge base."

                # Note: No ToolMessage needed here as it wasn't an LLM-driven tool call
                logger.info(f"Explicit RAG execution completed. Error: {tool_error}")
            except Exception as e:
                 tool_error = f"Failed to execute explicit RAG lookup: {e}"
                 logger.error(tool_error, exc_info=True)
        else:
             tool_error = "Could not find user query for explicit RAG lookup."
             logger.warning(tool_error)

    # --- Handle Return Flow Tools (Called Internally) ---
    # These might be triggered by `next_node` routing rather than direct LLM tool calls
    elif intent == "initiate_return" and state.get("next_node") == "handle_return_step_1":
        order_id = state.get("order_id")
        if order_id:
             logger.info(f"Executing get_order_details for return step 1, order: {order_id}")
             api_response = get_order_details.invoke({"order_id": order_id})
             if "error" in api_response:
                 tool_error = api_response["error"]
             elif not api_response.get("items"):
                  tool_error = f"No returnable items found for order {order_id} (may not be delivered or empty)."
        else:
            tool_error = "Order ID missing for initiating return."

    elif intent == "return_reason_provided" and state.get("next_node") == "handle_return_step_3":
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

    # Update state
    updates = {
        "api_response": api_response,
        "rag_context": rag_context,
        "tool_error": tool_error,
        "next_node": None # Clear marker unless explicitly set again later
    }
    # Clear multi-turn state if the final return step was successful
    if intent == "return_reason_provided" and not tool_error and api_response and "return_id" in api_response:
         updates.update({
              "item_sku_to_return": None,
              "return_reason": None,
              "available_return_items": None,
              # Keep order_id if needed for follow-up? Or clear? Let's clear for now.
              # "order_id": None
         })

    return updates


def handle_multi_turn_return(state: GraphState) -> Dict[str, Any]:
    """
    Manages the multi-step return process based on the current state.
    Generates clarification questions or prepares for the next step.
    This node acts as a router/state manager within the return flow.
    """
    intent = state.get("intent")
    tool_error = state.get("tool_error")
    api_response = state.get("api_response") # From get_order_details
    order_id = state.get("order_id")
    next_node_override = None # Allows overriding the default route to generate_response

    logger.info(f"Handling multi-turn return. Intent: {intent}, Error: {tool_error}")

    if tool_error:
        # If get_order_details or initiate_return failed, let generate_response handle the error message
        logger.warning(f"Error occurred in return flow: {tool_error}")
        return {"needs_clarification": False, "clarification_question": None} # Let generate_response show the error

    # --- Step 1: After trying to get order details ---
    if intent == "initiate_return" and state.get("next_node") == "handle_return_step_1":
        if api_response and api_response.get("items"):
            items = api_response["items"]
            logger.info(f"Order {order_id} details fetched. Items available for return: {items}")
            # Ask user which item to return
            item_list_str = "\n".join([f"- {item['name']} (SKU: {item['sku']})" for item in items])
            question = f"Okay, I found order {order_id}. Which item would you like to return? Please provide the SKU:\n{item_list_str}"
            return {
                "available_return_items": items,
                "needs_clarification": True,
                "clarification_question": question,
                "next_node": "generate_response" # Ask the question
            }
        else:
            # Error handled upstream or no items eligible
            error_msg = tool_error or api_response.get("error") or f"Sorry, I couldn't find any returnable items for order {order_id}. This might be because the order hasn't been delivered yet."
            return {"tool_error": error_msg, "needs_clarification": False} # Let generate_response show this

    # --- Step 2: After user provides SKU ---
    elif intent == "return_item_selection" and state.get("next_node") == "handle_return_step_2":
        sku = state.get("item_sku_to_return")
        logger.info(f"Item SKU {sku} selected by user.")
        # Ask for reason (optional step)
        question = f"Got it, you want to return item {sku}. Could you briefly tell me why you're returning it? (Optional)"
        return {
            "needs_clarification": True, # Need the reason (or confirmation to skip)
            "clarification_question": question,
            "next_node": "generate_response" # Ask the question
        }

    # --- Step 3: After user provides reason (or skips) ---
    elif intent == "return_reason_provided" and state.get("next_node") == "handle_return_step_3":
        # The actual API call is done in execute_tool for this intent
        # This node just confirms readiness for the final step.
        logger.info("Reason provided (or skipped). Ready to submit return request.")
        # No state change needed here, execute_tool will handle the API call next
        return {"needs_clarification": False, "clarification_question": None} # Proceed to execute_tool

    # --- Fallback/Error within multi-turn ---
    logger.warning(f"Unexpected state in handle_multi_turn_return. Intent: {intent}")
    return {"needs_clarification": False, "clarification_question": None} # Default exit


def generate_response(state: GraphState) -> Dict[str, Any]:
    """Generates the final response to the user based on the graph's state."""
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
            # Use LLM to synthesize answer from context and query
            prompt = f"""Based on the following information from our knowledge base:
            ---
            {rag_context}
            ---
            Answer the user's question: "{messages[-1].content if isinstance(messages[-1], HumanMessage) else 'the user query'}"
            Provide a concise and helpful answer. If the information isn't directly there, say you couldn't find the specific detail but provide related info if available.
            """
            ai_message = llm.invoke(prompt)
            response_text = ai_message.content
        else:
            response_text = "I looked in our knowledge base, but couldn't find specific information about that."
            logger.warning("Generating response, but RAG context was empty.")
    elif api_response:
        logger.info(f"Generating response from API result: {api_response}")
        # Format API responses nicely
        if intent == "get_order_status":
            response_text = f"The status for order {api_response.get('order_id', 'N/A')} is: {api_response.get('status', 'Unknown')}."
        elif intent == "get_tracking_info":
            if api_response.get('tracking_number'):
                 response_text = f"Tracking for order {api_response.get('order_id', 'N/A')}: Number {api_response.get('tracking_number')}, Carrier: {api_response.get('carrier', 'N/A')}, Status: {api_response.get('status', 'N/A')}."
            else:
                 response_text = f"Tracking information is not yet available for order {api_response.get('order_id', 'N/A')}. Status: {api_response.get('status', 'Unavailable')}."
        elif intent == "initiate_return" or intent == "return_reason_provided": # Final return step
            if api_response.get("return_id"):
                response_text = f"Success! {api_response.get('message', 'Return initiated.')} Your return ID is {api_response.get('return_id')}."
            else:
                # Should have been caught by tool_error, but fallback
                response_text = f"There was an issue processing your return. {api_response.get('message', 'Please try again later or contact support.')}"
        # Add more formatting for other intents/API responses if needed
        else:
            # Generic response for unformatted API results
            response_text = f"I received the following information: {str(api_response)}"
    elif intent == "greeting":
        response_text = "Hello! How can I help you with your order or answer your questions?"
    elif intent == "goodbye":
        response_text = "Goodbye! Let me know if you need anything else."
    else: # Fallback / Unsupported
        response_text = "I'm sorry, I can't assist with that request directly. I can help with order status, tracking, returns, and answer general questions from our FAQ."
        # Check if the last AI message had content (e.g. from failed classification)
        last_ai_message = messages[-1] if messages and isinstance(messages[-1], AIMessage) else None
        if last_ai_message and last_ai_message.content and not last_ai_message.tool_calls:
             response_text = last_ai_message.content # Use LLM's direct response if it exists and wasn't a tool call

    logger.info(f"Final generated response: {response_text}")
    # Append the final response to messages state
    # Important: Return only the update for the 'messages' key
    return {"messages": [AIMessage(content=response_text)]}