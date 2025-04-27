import logging
from typing import Dict, Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from .llm import llm # Keep this for generate_response
from .state import GraphState
from .tools import (
    llm_with_tools, # Keep this for classify_intent
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

    # --- MODIFICATION START ---
    # Get the last message, expecting it to be the user's query
    last_message = messages[-1]
    logger.info(f"Classifying intent for last message: '{last_message.content}'")

    # Ensure the last message is a HumanMessage before sending to LLM
    if not isinstance(last_message, HumanMessage):
        logger.error(f"classify_intent expected the last message to be HumanMessage, but got {type(last_message)}. Cannot proceed.")
        # Return an error state or handle appropriately
        return {
            "intent": "unsupported",
            "api_response": {"message": "Internal error: Unexpected message sequence."},
            "tool_error": "Unexpected message sequence.",
            "next_node": "generate_response"
        }

    # Invoke the LLM *only* with the latest human message for classification/tool routing
    # This avoids sending complex history with unresolved tool calls to this specific LLM call
    try:
        # Pass only the last user message in a list
        ai_response = llm_with_tools.invoke([last_message])

        # Crucially, append the AI response to the *full* state messages list for graph context
        state["messages"].append(ai_response)
    # --- MODIFICATION END ---

    # try: # Original try block starts here - moved invoke call above
        # ai_response = llm_with_tools.invoke(messages) # Original call - REMOVED
        # state["messages"].append(ai_response) # Already appended above

        tool_calls = ai_response.tool_calls
        logger.debug(f"LLM tool calls based on last message: {tool_calls}")

        if tool_calls:
            # ... (rest of the tool call handling logic remains the same) ...
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
                "next_node": "execute_tool" # Default for tool calls
            }
            # Determine intent and next_node based on tool_name
            if tool_name == get_order_status.name:
                updates["intent"] = "get_order_status"
            elif tool_name == get_tracking_info.name:
                updates["intent"] = "get_tracking_info"
            elif tool_name == knowledge_base_lookup.name:
                updates["intent"] = "knowledge_base_query"
            elif tool_name == initiate_return_request.name:
                # If the LLM directly calls initiate_return, maybe it should go to step 3?
                # Or handle it differently? For now, assume it needs details first if reason/sku missing.
                # Let's assume it goes to execute_tool which will call initiate_return_request directly.
                 updates["intent"] = "initiate_return" # Set intent
                 # Check if all args present, otherwise maybe it should go to details?
                 if not extracted_sku or not extracted_order_id:
                     logger.warning("LLM called initiate_return without order_id/sku, routing to get details first.")
                     updates["next_node"] = "handle_return_step_1" # Route to get details
                     updates["intent"] = "initiate_return" # Keep intent
                 else:
                     # If LLM provides everything, go straight to execute_tool to submit
                     logger.info("LLM provided all details for initiate_return.")
                     updates["next_node"] = "execute_tool"


            elif tool_name == get_order_details.name:
                updates["intent"] = "initiate_return" # Part of the return flow
                updates["next_node"] = "handle_return_step_1" # Signal to get details
            else:
                logger.warning(f"LLM called unknown tool: {tool_name}")
                updates["intent"] = "unsupported"
                updates["api_response"] = {
                    "message": f"I encountered an issue understanding how to use the tool: {tool_name}."}
                updates["next_node"] = "generate_response" # Go straight to response with error

            logger.info(f"Intent classified as '{updates.get('intent')}' with tool call '{tool_name}'. Routing to '{updates.get('next_node')}'.")
            return updates

        else: # No tool calls from LLM based on last message
            logger.info("No tool call detected by LLM based on last message. Checking multi-turn context or treating as KB query.")
            # ... (rest of the multi-turn / KB query logic remains the same) ...
            # This part relies on the full state history (intent, needs_clarification etc.)
            # which is correctly maintained by LangGraph.
            current_intent = state.get("intent")
            needs_clarification = state.get("needs_clarification")

            if current_intent == "initiate_return" and needs_clarification:
                # Try to match SKU from user message
                potential_sku = last_message.content.strip().upper()
                available_items = state.get("available_return_items", [])
                matched_item = next((item for item in available_items if item["sku"] == potential_sku), None)

                if matched_item:
                    logger.info(f"Identified SKU '{potential_sku}' for return from user message.")
                    return {
                        "item_sku_to_return": potential_sku,
                        "needs_clarification": False,
                        "clarification_question": None,
                        "intent": "return_item_selection", # Update intent
                        "next_node": "handle_return_step_2" # Route to ask for reason
                    }
                else:
                    # SKU doesn't match, ask again
                    logger.warning(f"User provided input '{potential_sku}', but it doesn't match available SKUs.")
                    item_skus_str = ', '.join(
                        [item['sku'] for item in available_items]) if available_items else "any items"
                    # Re-ask the clarification question stored in the state, adding feedback
                    clarification_feedback = f"Sorry, '{potential_sku}' doesn't seem to match the items. "
                    original_question = state.get("clarification_question", f"Please provide one of the following SKUs: {item_skus_str}")
                    # Avoid appending feedback repeatedly if original question already contains it
                    if "Please provide one of the following SKUs" in original_question:
                         new_question = clarification_feedback + original_question
                    else: # If original question was different, just prepend feedback
                         new_question = clarification_feedback + f"Please provide one of the following SKUs: {item_skus_str}"


                    return {
                        "clarification_question": new_question,
                        "needs_clarification": True, # Remain in clarification state
                        # "intent": "initiate_return", # Keep intent
                        "next_node": "generate_response" # Go back to generate response to ask again
                    }

            elif current_intent == "return_item_selection" and needs_clarification:
                # Assume this message is the return reason
                logger.info("Received potential return reason from user message.")
                return_reason_text = last_message.content.strip()
                # Handle cases where user explicitly says skip or provides empty input
                if return_reason_text.lower() in ["skip", "none", ""]:
                    logger.info("User skipped providing a return reason.")
                    return_reason_text = None

                return {
                    "return_reason": return_reason_text,
                    "needs_clarification": False,
                    "clarification_question": None,
                    "intent": "return_reason_provided", # Update intent
                    "next_node": "handle_return_step_3" # Route to submit
                }

            # Default: If no tool call and not in a multi-turn clarification, assume KB query
            logger.info("No active multi-turn context. Treating as knowledge base query.")
            return {
                "intent": "knowledge_base_query",
                "needs_clarification": False,
                "clarification_question": None,
                "next_node": "execute_tool", # Route to execute KB lookup
                "tool_error": None,
            }

    except Exception as e: # Catch errors during LLM invocation or subsequent logic
        logger.error(f"Error during intent classification or processing: {e}", exc_info=True)
        # Ensure state includes the message that caused the error if possible
        if 'ai_response' not in locals() and isinstance(last_message, HumanMessage):
             # If error happened before LLM response, add the human message back if needed?
             # The graph state should already contain it from main.py
             pass

        return {
            "intent": "unsupported",
            "api_response": {"message": f"I had trouble understanding your request due to an internal error: {type(e).__name__}"},
            "tool_error": f"Intent classification failed: {e}",
            "next_node": "generate_response"
        }


# --- execute_tool ---
# (No changes needed here - it correctly finds the tool calls in the *last* message added by classify_intent
# and adds the ToolMessage to the state)
def execute_tool(state: GraphState) -> Dict[str, Any]:
    messages = state["messages"]
    # Ensure messages list is not empty and last message is AIMessage before accessing tool_calls
    last_ai_message = messages[-1] if messages and isinstance(messages[-1], AIMessage) else None
    tool_calls = last_ai_message.tool_calls if last_ai_message and hasattr(last_ai_message, 'tool_calls') else []
    intent = state.get("intent")
    next_node_marker = state.get("next_node") # Check the marker set by classify_intent if needed

    logger.info(f"Executing tool. Current Intent: {intent}. Node marker: {next_node_marker}. Found tool calls in last AI msg: {bool(tool_calls)}")

    api_response = None
    rag_context = None
    tool_error = None
    # Default message if no specific tool message content is generated
    tool_message_content = "Tool execution status unknown."

    if tool_calls:
        # Always process the first tool call if present from the LLM
        call = tool_calls[0]
        tool_name = call['name']
        tool_args = call['args']
        tool_id = call['id']
        logger.info(f"Executing tool '{tool_name}' via LLM call with args: {tool_args} and ID: {tool_id}")

        try:
            # Find the corresponding tool function from the available list
            tool_func = next((t for t in available_tools if t.name == tool_name), None)
            if tool_func:
                # Invoke the tool function with the arguments provided by the LLM
                result = tool_func.invoke(tool_args)
                logger.info(f"Tool '{tool_name}' executed.") # Log success before checking result type

                # Process the result based on the tool called
                if tool_name == knowledge_base_lookup.name:
                    if isinstance(result, dict) and "error" in result:
                        # Handle errors returned by the KB tool itself
                        tool_error = result["error"]
                        rag_context = None
                        tool_message_content = f"Tool '{tool_name}' execution returned an error: {tool_error}"
                        logger.warning(tool_message_content)
                    elif isinstance(result, str):
                        # Success case for KB lookup
                        rag_context = result if result else None # Handle empty string result
                        if rag_context:
                             tool_message_content = f"Successfully looked up knowledge base. Context length: {len(rag_context)}"
                             logger.info(tool_message_content)
                        else:
                             tool_message_content = "Knowledge base lookup returned no specific results."
                             logger.warning(tool_message_content)
                    else:
                        # Unexpected result type from KB tool
                        tool_error = f"Tool '{tool_name}' returned unexpected result type: {type(result)}"
                        rag_context = None
                        tool_message_content = tool_error
                        logger.error(tool_message_content)

                else: # Handle results from API-like tools (get_status, get_tracking, get_details, initiate_return)
                    api_response = result if isinstance(result, dict) else {"result": str(result)}
                    if "error" in api_response:
                        # Tool returned an error dictionary
                        tool_error = api_response["error"]
                        tool_message_content = f"Tool '{tool_name}' execution returned an error: {tool_error}"
                        logger.warning(tool_message_content)
                    else:
                        # Tool execution successful
                        tool_message_content = f"Successfully called tool {tool_name}."
                        logger.info(tool_message_content)
            else:
                # Tool name provided by LLM doesn't match any available tool
                tool_error = f"Tool '{tool_name}' not found in available tools."
                logger.error(tool_error)
                tool_message_content = f"Error: Tool '{tool_name}' is not available."
                api_response = {"error": tool_error} # Set api_response error as well

        except Exception as e:
            # Catch any unexpected error during tool function invocation
            tool_error = f"Failed to execute tool '{tool_name}': {e}"
            logger.error(tool_error, exc_info=True)
            tool_message_content = f"An unexpected error occurred while executing tool {tool_name}."
            api_response = {"error": tool_message_content} # Set api_response error

        # IMPORTANT: Append the ToolMessage to the history AFTER execution attempt
        state["messages"].append(ToolMessage(content=str(tool_message_content), tool_call_id=tool_id))
        logger.debug(f"Appended ToolMessage for call ID {tool_id}")

    # --- Handle cases where execution is triggered by state/intent, not direct LLM tool_call ---
    # Example: Explicit KB lookup requested by classify_intent routing
    elif intent == "knowledge_base_query" and next_node_marker == "execute_tool":
        logger.info("Executing RAG tool explicitly based on 'knowledge_base_query' intent.")
        last_user_message = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        if last_user_message:
            try:
                # Invoke KB lookup tool directly
                # Note: We don't add a ToolMessage here as there was no preceding AIMessage with a tool_call_id
                rag_context_result = knowledge_base_lookup.invoke({"query": last_user_message})

                if isinstance(rag_context_result, dict) and "error" in rag_context_result:
                    tool_error = rag_context_result["error"]
                    rag_context = None
                    logger.warning(f"Explicit RAG lookup failed: {tool_error}")
                elif isinstance(rag_context_result, str) and rag_context_result:
                    rag_context = rag_context_result
                    logger.info(f"Explicit RAG execution successful, context length: {len(rag_context)}.")
                else: # Empty string or other non-error result
                    rag_context = None
                    logger.warning("Explicit RAG lookup returned no results.")

            except Exception as e:
                tool_error = f"Failed to execute explicit RAG lookup: {e}"
                logger.error(tool_error, exc_info=True)
        else:
            tool_error = "Could not find user query for explicit RAG lookup."
            logger.warning(tool_error)

    # Example: Getting order details as step 1 of return process
    elif intent == "initiate_return" and next_node_marker == "handle_return_step_1":
        order_id = state.get("order_id")
        if order_id:
            logger.info(f"Executing get_order_details internally for return step 1, order: {order_id}")
            # Note: No ToolMessage added here either
            api_response = get_order_details.invoke({"order_id": order_id})
            if "error" in api_response:
                tool_error = api_response["error"]
                logger.warning(f"get_order_details failed for return step 1: {tool_error}")
            # Further checks for delivered/items happen in handle_multi_turn_return
            elif not api_response.get("items"):
                 # Handle case where API returns success but no items (maybe already returned?)
                 tool_error = f"Order {order_id} details retrieved, but no items found."
                 logger.warning(tool_error)
                 # api_response still contains the details, handle_multi_turn can use it
        else:
            tool_error = "Order ID missing for initiating return step 1."
            logger.warning(tool_error)
            api_response = {"error": tool_error}

    # Example: Submitting the return request as step 3
    elif intent == "return_reason_provided" and next_node_marker == "execute_tool":
        order_id = state.get("order_id")
        sku = state.get("item_sku_to_return")
        reason = state.get("return_reason")
        if order_id and sku:
            logger.info(f"Executing initiate_return_request internally for return step 3. Order: {order_id}, SKU: {sku}, Reason: '{reason}'")
            # Note: No ToolMessage added
            api_response = initiate_return_request.invoke({"order_id": order_id, "sku": sku, "reason": reason})
            if "error" in api_response:
                tool_error = api_response["error"]
                logger.warning(f"initiate_return_request failed: {tool_error}")
        else:
            tool_error = "Missing Order ID or Item SKU for submitting return."
            logger.warning(tool_error)
            api_response = {"error": tool_error}

    else:
         logger.debug("No specific tool execution path matched in execute_tool node.")


    # --- Prepare updates for the graph state ---
    updates = {
        "api_response": api_response, # Store API results (or errors)
        "rag_context": rag_context,   # Store RAG results
        "tool_error": tool_error,     # Store explicit error messages
        "next_node": None # Clear marker for next explicit node call
    }

    # If the return was submitted successfully, clear the return-specific state
    if intent == "return_reason_provided" and not tool_error and api_response and "return_id" in api_response:
        logger.info(f"Return successful (ID: {api_response.get('return_id')}), clearing return state fields.")
        updates.update({
            "item_sku_to_return": None,
            "return_reason": None,
            "available_return_items": None,
            # Keep order_id? Maybe for follow-up questions? Decide based on desired flow.
            # "order_id": None,
            # Reset intent? Or let generate_response handle the final message?
            # "intent": None,
        })

    # Log the final updates being returned by the node
    logger.debug(f"execute_tool node returning updates: {updates}")
    return updates


# --- handle_multi_turn_return ---
# (No changes needed here - relies on state updated by previous nodes)
def handle_multi_turn_return(state: GraphState) -> Dict[str, Any]:
    intent = state.get("intent")
    tool_error = state.get("tool_error")
    api_response = state.get("api_response")
    order_id = state.get("order_id")
    expected_step_marker = state.get("next_node") # Check the marker passed from classify_intent

    logger.info(
        f"Handling multi-turn return. Intent: {intent}, API Response: {bool(api_response)}, Error: {tool_error}, Expected Step Marker: {expected_step_marker}"
    )

    # Case 1: Coming from execute_tool after trying to get details (handle_return_step_1)
    if expected_step_marker == "handle_return_step_1":
        if tool_error:
            logger.warning(f"Error occurred getting order details for return: {tool_error}")
            # Pass the error along to generate_response
            return {"tool_error": tool_error, "needs_clarification": False, "clarification_question": None, "next_node": "generate_response"}
        elif api_response and api_response.get("items") and api_response.get("delivered"):
             # Details fetched successfully, items exist, order delivered
            items = api_response["items"]
            logger.info(f"Order {order_id} details fetched. Items available for return: {items}")
            item_list_str = "\n".join([f"- {item['name']} (SKU: {item['sku']})" for item in items])
            question = f"Okay, I found order {order_id}. Which item would you like to return? Please provide the SKU:\n{item_list_str}"
            # Ask user for SKU
            return {
                "available_return_items": items,
                "needs_clarification": True,
                "clarification_question": question,
                "next_node": "generate_response" # Go to generate response to ask the question
                # Keep intent as initiate_return
            }
        else:
            # Details fetched but order not delivered, no items, or other issue
            error_msg = tool_error or (api_response.get("error") if api_response else None) or f"Sorry, I couldn't find any returnable items for order {order_id}, or the order is not eligible for return."
            logger.warning(f"Order {order_id} not eligible for return based on details: {error_msg}")
            return {"tool_error": error_msg, "needs_clarification": False, "next_node": "generate_response"}

    # Case 2: Coming from classify_intent after user provided SKU (handle_return_step_2)
    elif expected_step_marker == "handle_return_step_2" and intent == "return_item_selection":
        sku = state.get("item_sku_to_return")
        if sku:
            logger.info(f"Item SKU {sku} selected by user. Asking for reason.")
            question = f"Got it, you want to return item {sku}. Could you briefly tell me why you're returning it? (Optional, you can just press Enter or say 'skip')"
            # Ask for reason
            return {
                "needs_clarification": True,
                "clarification_question": question,
                "next_node": "generate_response" # Go to generate response to ask
                # Keep intent as return_item_selection
            }
        else: # Should not happen if classify_intent worked correctly
             logger.error("Reached handle_return_step_2 but item_sku_to_return is not set in state.")
             return {"tool_error": "Internal error: SKU selection missing.", "needs_clarification": False, "next_node": "generate_response"}

    # Case 3: Coming from classify_intent after user provided reason (handle_return_step_3)
    elif expected_step_marker == "handle_return_step_3" and intent == "return_reason_provided":
        logger.info("Reason provided (or skipped). Preparing to submit return request via execute_tool.")
        # Signal to go back to execute_tool to perform the actual submission
        return {
            "needs_clarification": False,
            "clarification_question": None,
            "next_node": "execute_tool" # Route back to execute_tool
            # Keep intent as return_reason_provided
            }

    # Fallback / Unexpected state
    else:
        logger.warning(
            f"Unexpected state in handle_multi_turn_return. Intent: {intent}, Marker: {expected_step_marker}. Routing to generate_response.")
        final_tool_error = tool_error or "There was an issue processing the return step."
        return {"tool_error": final_tool_error, "needs_clarification": False, "clarification_question": None, "next_node": "generate_response"}



# --- generate_response ---
# (No changes needed here - it generates response based on the final state)
def generate_response(state: GraphState) -> Dict[str, Any]:
    messages = state["messages"]
    intent = state.get("intent")
    api_response = state.get("api_response")
    rag_context = state.get("rag_context")
    tool_error = state.get("tool_error")
    needs_clarification = state.get("needs_clarification")
    clarification_question = state.get("clarification_question")

    logger.info(f"Generating response. Intent: {intent}, Error: {tool_error}, Clarification: {needs_clarification}, API Response: {bool(api_response)}, RAG Context: {bool(rag_context)}")

    response_text = "" # Default empty response

    # Priority 1: Ask clarification question if needed
    if needs_clarification and clarification_question:
        response_text = clarification_question
        logger.info(f"Generated clarification response: '{response_text[:100]}...'")

    # Priority 2: Report tool errors if no clarification needed
    elif tool_error:
        # Provide a user-friendly error message
        response_text = f"I encountered an issue: {tool_error}"
        # Avoid overly technical details unless necessary
        # Example: Don't show "API error 500" directly to user if possible
        # You might map common errors to friendlier messages here
        logger.warning(f"Generated error response: {response_text}")

    # Priority 3: Respond based on RAG context if available
    elif intent == "knowledge_base_query" and rag_context:
        logger.info("Generating response from RAG context.")
        # Find the last human message to provide context for the LLM prompt
        last_user_message_content = "your question" # Default fallback
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_user_message_content = msg.content
                break

        # Construct a prompt for the LLM to synthesize an answer from the context
        prompt = f"""Based *only* on the following information from our knowledge base:
        --- Knowledge Base Context ---
        {rag_context}
        --- End Context ---

        Answer the user's question: "{last_user_message_content}"

        Provide a concise and helpful answer based *strictly* on the provided context.
        If the context doesn't contain the answer, state that you couldn't find the specific detail in the knowledge base. Do not make up information.
        """
        try:
            # Invoke the base LLM (without tools)
            ai_message = llm.invoke(prompt)
            response_text = ai_message.content
            logger.info("Successfully generated response from RAG context.")
        except Exception as e:
            logger.error(f"LLM invocation failed during RAG response generation: {e}", exc_info=True)
            response_text = "I found some information in our knowledge base, but had trouble formulating a final answer. Could you perhaps rephrase your question?"

    # Priority 4: Respond based on successful API response
    elif api_response: # Check if api_response exists and doesn't contain an 'error' key handled above
        logger.info(f"Generating response from API result: {api_response}")
        if intent == "get_order_status":
            response_text = f"The status for order {api_response.get('order_id', 'N/A')} is: {api_response.get('status', 'Unknown')}."
        elif intent == "get_tracking_info":
            if api_response.get('tracking_number'):
                response_text = (f"Tracking for order {api_response.get('order_id', 'N/A')}: "
                                 f"Number: {api_response.get('tracking_number')}, "
                                 f"Carrier: {api_response.get('carrier', 'N/A')}, "
                                 f"Status: {api_response.get('status', 'N/A')}.")
            else:
                # Handle case where tracking is not available yet (specific status from tool)
                status_msg = api_response.get('status', 'Unavailable') # Use status if present
                response_text = (
                    f"Tracking information is not yet available for order {api_response.get('order_id', 'N/A')}. "
                    f"Current status: {status_msg}.")
        # Specific response for successful return submission
        elif intent == "return_reason_provided" and api_response.get("return_id"):
             response_text = f"Success! {api_response.get('message', 'Your return has been initiated.')} Your return ID is {api_response.get('return_id')}."
             logger.info(f"Generated success response for return ID {api_response.get('return_id')}")
        # Generic handler for other API responses (or fallback if specific intent logic missed)
        # Check if the api_response dictionary contains a meaningful message itself
        elif api_response.get('message'):
             response_text = api_response['message']
        else:
            # Fallback if no specific format is matched
            response_text = f"I received the following details: {str(api_response)}"
            logger.warning(f"Generated generic response for API result: {response_text}")

    # Priority 5: Handle simple intents like greetings/goodbyes
    elif intent == "greeting":
        response_text = "Hello! How can I help you with your order or answer your questions?"
    elif intent == "goodbye":
        response_text = "Goodbye! Let me know if you need anything else."

    # Priority 6: Fallback for unsupported intents or if RAG failed silently
    elif intent == "knowledge_base_query" and not rag_context and not tool_error:
        # RAG was attempted but returned no results and no error
         response_text = "I looked in our knowledge base, but couldn't find specific information about that topic."
         logger.warning("Generating response: RAG context was empty or lookup returned no results.")

    # Final Fallback: If none of the above conditions met
    else:
        logger.info("No specific response path matched, generating fallback response.")
        # Check if the last message in history is already an AI message (e.g., from LLM without tool calls)
        # This prevents overriding a potentially valid direct LLM response.
        last_message_in_state = messages[-1] if messages else None
        if isinstance(last_message_in_state, AIMessage) and not last_message_in_state.tool_calls:
             response_text = last_message_in_state.content
             logger.info("Using content from the last non-tool-calling AI message as fallback response.")
        else:
             # Generic fallback message
             response_text = "I'm sorry, I can't assist with that specific request right now. I can help with order status, tracking, returns, and answer general questions from our FAQ."
             logger.warning(f"Generated generic fallback response for intent '{intent}'.")


    logger.info(f"Final generated response text: '{response_text[:100]}...'")

    # Append the final AIMessage to the state
    # Ensure we don't add an empty message if response_text somehow ended up empty
    if response_text:
         state["messages"].append(AIMessage(content=response_text))
    else:
         logger.error("generate_response resulted in an empty response_text. Appending a default error message.")
         state["messages"].append(AIMessage(content="Sorry, I encountered an issue and couldn't generate a response."))

    # This node signifies the end of processing for this turn for most paths
    # Return empty dict as state modifications happen by appending to messages list
    return {}