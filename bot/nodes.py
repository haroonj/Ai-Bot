# bot/nodes.py

import logging
from typing import Dict, Any, Optional, List
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# Corrected: Import only llm from .llm
from .llm import llm
# Corrected: Import llm_with_tools and specific tools from .tools
from .tools import (
    llm_with_tools,
    get_order_status,
    get_tracking_info,
    get_order_details,
    initiate_return_request,
    knowledge_base_lookup,
    available_tools
)
from .state import GraphState, Intent


logger = logging.getLogger(__name__)

# --- Node Functions ---

def classify_intent(state: GraphState) -> Dict[str, Any]:
    """
    Classifies the user's intent based on the latest message and conversation history.
    Uses LLM with tool calling to determine the primary action or if clarification is needed.
    Also extracts relevant entities like order_id.
    """
    messages = state["messages"]
    # Ensure messages list is not empty before accessing the last element
    if not messages:
        logger.warning("classify_intent called with empty messages state.")
        # Decide how to handle this - perhaps return unsupported or ask for input
        return {"intent": "unsupported", "api_response": {"message": "I need a message to understand your request."}, "next_node": "generate_response"}

    last_message = messages[-1]
    logger.info(f"Classifying intent for message: '{last_message.content}'")

    try:
        # Invoke the LLM bound with tools
        ai_response = llm_with_tools.invoke(messages)
        # IMPORTANT: Update the state's messages list IMMEDIATELY after the call
        # LangGraph merges the dictionary returned by the node with the current state.
        # To modify lists like 'messages', we should include the updated list in the return dict.
        # However, appending here and relying on the state object being modified directly
        # *can* work depending on LangGraph's internal handling, but returning the
        # full updated list is safer. Let's stick to returning updates for now,
        # and assume LangGraph correctly passes the modified state object if we mutate it.
        # For robustness, consider returning {"messages": messages + [ai_response]}
        state["messages"].append(ai_response) # Append AI's thought process/tool calls

        tool_calls = ai_response.tool_calls
        logger.debug(f"LLM tool calls: {tool_calls}")

        if tool_calls:
            # --- Tool Call Detected ---
            first_call = tool_calls[0]
            tool_name = first_call['name']
            tool_args = first_call['args']
            tool_id = first_call['id'] # Keep track for ToolMessage later

            extracted_order_id = tool_args.get('order_id')
            extracted_sku = tool_args.get('sku')
            extracted_reason = tool_args.get('reason')

            # Prepare updates to be returned and merged into the state
            updates: Dict[str, Any] = {
                "order_id": extracted_order_id or state.get("order_id"),
                "item_sku_to_return": extracted_sku or state.get("item_sku_to_return"),
                "return_reason": extracted_reason or state.get("return_reason"),
                "needs_clarification": False,
                "clarification_question": None,
                "tool_error": None,
                "next_node": "execute_tool" # Default next step after tool call detected
            }

            # Determine intent based on the tool called
            if tool_name == get_order_status.name:
                updates["intent"] = "get_order_status"
            elif tool_name == get_tracking_info.name:
                updates["intent"] = "get_tracking_info"
            elif tool_name == knowledge_base_lookup.name:
                updates["intent"] = "knowledge_base_query"
            elif tool_name == initiate_return_request.name:
                # LLM might directly call this if user provides all info upfront
                updates["intent"] = "initiate_return"
            elif tool_name == get_order_details.name:
                # LLM might call this if user asks for details before return
                updates["intent"] = "initiate_return"
                updates["next_node"] = "handle_return_step_1" # Explicitly go to return flow
            else:
                # Handle unknown tool calls
                logger.warning(f"LLM called unknown tool: {tool_name}")
                updates["intent"] = "unsupported"
                # Use api_response to pass error message to generate_response node
                updates["api_response"] = {"message": f"I encountered an issue understanding how to use the tool: {tool_name}."}
                updates["next_node"] = "generate_response" # Go straight to response generation

            logger.info(f"Intent classified as '{updates.get('intent')}' with tool call '{tool_name}'.")
            # Return the dictionary of updates to be merged into the graph state
            return updates

        else:
            # --- No Tool Call - Interpret Intent or Handle Multi-Turn ---
            logger.info("No tool call detected by LLM. Assuming KB query or multi-turn context.")
            current_intent = state.get("intent")
            needs_clarification = state.get("needs_clarification")

            # Check if we are in the middle of the return process and expecting input
            if current_intent == "initiate_return" and needs_clarification:
                # Expecting SKU
                potential_sku = last_message.content.strip().upper()
                available_items = state.get("available_return_items", [])
                matched_item = next((item for item in available_items if item["sku"] == potential_sku), None)

                if matched_item:
                    logger.info(f"Identified SKU '{potential_sku}' for return.")
                    return {
                        "item_sku_to_return": potential_sku,
                        "needs_clarification": False,
                        "clarification_question": None,
                        "intent": "return_item_selection", # Update intent
                        "next_node": "handle_return_step_2" # Proceed in return flow
                    }
                else:
                    # SKU didn't match, ask again
                    logger.warning(f"User provided input '{potential_sku}', but it doesn't match available SKUs.")
                    item_skus_str = ', '.join([item['sku'] for item in available_items]) if available_items else "any available items"
                    return {
                        "clarification_question": f"Sorry, '{potential_sku}' doesn't seem to match the items in your order. Please provide one of the following SKUs: {item_skus_str}",
                        "needs_clarification": True, # Keep asking
                        "next_node": "generate_response" # Ask the clarification question
                    }

            elif current_intent == "return_item_selection" and needs_clarification:
                # Expecting Reason (or skip)
                logger.info("Received potential return reason.")
                return {
                    "return_reason": last_message.content, # Store the reason
                    "needs_clarification": False,
                    "clarification_question": None,
                    "intent": "return_reason_provided", # Update intent
                    "next_node": "handle_return_step_3" # Proceed in return flow
                }

            # Default to KB lookup if no other context applies
            logger.info("No active multi-turn context. Treating as knowledge base query.")
            return {
                "intent": "knowledge_base_query",
                "needs_clarification": False,
                "clarification_question": None,
                "next_node": "execute_tool", # Trigger RAG via execute_tool
                "tool_error": None, # Clear any previous tool error
            }

    except Exception as e:
        logger.error(f"Error during intent classification: {e}", exc_info=True)
        # Fallback on error
        return {
            "intent": "unsupported",
            "api_response": {"message": "I had trouble understanding your request."},
            "next_node": "generate_response"
        }


def execute_tool(state: GraphState) -> Dict[str, Any]:
    """Executes the appropriate tool based on the classified intent and tool calls."""
    messages = state["messages"]
    # Safely get the last AI message if it exists and has tool calls
    last_ai_message = messages[-1] if messages and isinstance(messages[-1], AIMessage) else None
    tool_calls = last_ai_message.tool_calls if last_ai_message and hasattr(last_ai_message, 'tool_calls') else []
    intent = state.get("intent")
    logger.info(f"Executing tool for intent: {intent}")

    # Initialize results for this execution run
    api_response = None
    rag_context = None
    tool_error = None
    tool_message_content = "Tool execution did not proceed as expected." # Default if no path matches

    # --- Path 1: Execute based on LLM Tool Call ---
    if tool_calls:
        # Handle the first tool call (simplification for this example)
        call = tool_calls[0]
        tool_name = call['name']
        tool_args = call['args']
        tool_id = call['id']
        logger.info(f"Executing tool '{tool_name}' via LLM call with args: {tool_args}")

        try:
            # Find the corresponding tool function from the available tools list
            tool_func = next((t for t in available_tools if t.name == tool_name), None)
            if tool_func:
                # Invoke the actual tool function
                result = tool_func.invoke(tool_args)
                logger.info(f"Tool '{tool_name}' executed successfully.")

                # Process result based on tool type
                if tool_name == knowledge_base_lookup.name:
                    # Handle potential error dict returned by KB tool
                    if isinstance(result, dict) and "error" in result:
                        tool_error = result["error"]
                        rag_context = None
                        tool_message_content = f"Tool execution returned an error: {tool_error}"
                    elif isinstance(result, str) and result:
                        rag_context = result
                        tool_message_content = f"Successfully looked up knowledge base. Context length: {len(rag_context)}"
                    else: # Empty string or unexpected result from KB tool
                        rag_context = None
                        # We might not set tool_error here, generate_response handles empty context
                        tool_message_content = "Knowledge base lookup returned no specific results."
                else: # API tools
                    api_response = result if isinstance(result, dict) else {"result": str(result)}
                    # Check for errors returned *by the tool* itself (e.g., 404)
                    if "error" in api_response:
                        tool_error = api_response["error"]
                        tool_message_content = f"Tool execution returned an error: {tool_error}"
                    else:
                        tool_message_content = f"Successfully called API tool {tool_name}."
            else:
                # Tool name called by LLM not found in our list
                tool_error = f"Tool '{tool_name}' not found in available tools."
                logger.error(tool_error)
                tool_message_content = f"Error: Tool '{tool_name}' is not available."

        except Exception as e:
            # Catch unexpected errors during tool execution
            tool_error = f"Failed to execute tool '{tool_name}': {e}"
            logger.error(tool_error, exc_info=True)
            tool_message_content = f"An unexpected error occurred while executing tool {tool_name}."

        # Append ToolMessage with the outcome to the conversation history
        state["messages"].append(ToolMessage(content=tool_message_content, tool_call_id=tool_id))

    # --- Path 2: Execute RAG explicitly if intent is KB but no tool call ---
    elif intent == "knowledge_base_query":
        logger.info("Executing RAG tool explicitly for knowledge_base_query intent.")
        # Find the last human message to use as the query
        last_user_message = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        if last_user_message:
            try:
                # Invoke the KB lookup tool directly
                rag_context_result = knowledge_base_lookup.invoke({"query": last_user_message})
                # Process the result
                if isinstance(rag_context_result, dict) and "error" in rag_context_result:
                    tool_error = rag_context_result["error"]
                    rag_context = None
                elif isinstance(rag_context_result, str) and rag_context_result:
                    rag_context = rag_context_result
                    logger.info(f"Explicit RAG execution successful.")
                else:
                    rag_context = None
                    tool_error = "No relevant information found in the knowledge base."
                    logger.warning("Explicit RAG lookup returned no results.")

                # No ToolMessage needed here as it wasn't an LLM-driven tool call
            except Exception as e:
                 tool_error = f"Failed to execute explicit RAG lookup: {e}"
                 logger.error(tool_error, exc_info=True)
        else:
             tool_error = "Could not find user query for explicit RAG lookup."
             logger.warning(tool_error)

    # --- Path 3: Execute tools internally for multi-turn return flow ---
    elif intent == "initiate_return" and state.get("next_node") == "handle_return_step_1":
        order_id = state.get("order_id")
        if order_id:
             logger.info(f"Executing get_order_details for return step 1, order: {order_id}")
             # Invoke the get_order_details tool
             api_response = get_order_details.invoke({"order_id": order_id})
             # Check for errors or unsuitable order details
             if "error" in api_response:
                 tool_error = api_response["error"]
             elif not api_response.get("items"): # Check if items list is empty or missing
                 tool_error = f"No returnable items found for order {order_id} (order might not be delivered or items are ineligible)."
                 # Keep api_response to potentially inform the user in generate_response
             elif not api_response.get("delivered"): # Double check delivered status
                  tool_error = f"Order {order_id} is not marked as delivered yet, cannot initiate return."
                  # Keep api_response
        else:
            tool_error = "Order ID missing for initiating return."
            logger.warning(tool_error)

    elif intent == "return_reason_provided" and state.get("next_node") == "execute_tool": # Check marker set by handle_multi_turn
        # Execute the final return submission API call
        order_id = state.get("order_id")
        sku = state.get("item_sku_to_return")
        reason = state.get("return_reason")
        if order_id and sku:
            logger.info(f"Executing initiate_return_request for return step 3. Order: {order_id}, SKU: {sku}")
            # Invoke the initiate_return_request tool
            api_response = initiate_return_request.invoke({"order_id": order_id, "sku": sku, "reason": reason})
            # Check for errors returned by the API call
            if "error" in api_response:
                tool_error = api_response["error"]
        else:
            tool_error = "Missing Order ID or Item SKU for submitting return."
            logger.warning(tool_error)

    # --- Update State ---
    updates = {
        "api_response": api_response,
        "rag_context": rag_context,
        "tool_error": tool_error,
        "next_node": None # Clear marker unless explicitly set again later
    }
    # Clear multi-turn state *only if* the final return submission was successful
    if intent == "return_reason_provided" and not tool_error and api_response and "return_id" in api_response:
        logger.info(f"Return successful (ID: {api_response.get('return_id')}), clearing return state.")
        updates.update({
            "item_sku_to_return": None,
            "return_reason": None,
            "available_return_items": None,
            # Keep order_id? Maybe useful for follow-up? Let's clear it for now.
            # "order_id": None
        })

    return updates


def handle_multi_turn_return(state: GraphState) -> Dict[str, Any]:
    """
    Manages the state transitions within the multi-step return process.
    Determines if clarification is needed or if the process can proceed.
    """
    intent = state.get("intent")
    tool_error = state.get("tool_error")
    api_response = state.get("api_response") # Result from get_order_details if available
    order_id = state.get("order_id")
    # Get the marker that indicates which step we expected to handle
    # This marker is set by the 'classify_intent' node.
    expected_step_marker = state.get("next_node")

    logger.info(f"Handling multi-turn return. Intent: {intent}, Error: {tool_error}, Expected Step Marker: {expected_step_marker}")

    # If an error occurred getting details, pass it to generate_response
    if tool_error and expected_step_marker == "handle_return_step_1":
        logger.warning(f"Error occurred getting order details: {tool_error}")
        # Don't need clarification, just generate the error message
        return {"needs_clarification": False, "clarification_question": None, "next_node": "generate_response"}

    # --- Step 1 Outcome: Received Order Details, Ask for SKU ---
    # This path is entered after execute_tool successfully ran get_order_details
    if intent == "initiate_return" and api_response and api_response.get("items"):
        items = api_response["items"]
        logger.info(f"Order {order_id} details fetched. Items available for return: {items}")
        # Format item list for the user
        item_list_str = "\n".join([f"- {item['name']} (SKU: {item['sku']})" for item in items])
        question = f"Okay, I found order {order_id}. Which item would you like to return? Please provide the SKU:\n{item_list_str}"
        # Update state to ask the question
        return {
            "available_return_items": items, # Store items for validation later
            "needs_clarification": True,     # Set flag to indicate we need user input
            "clarification_question": question, # The question to ask
            "next_node": "generate_response" # Route to generate_response to ask
        }
    elif intent == "initiate_return" and expected_step_marker == "handle_return_step_1":
        # Handle case where get_order_details ran but found no items or error occurred (already checked tool_error above)
        error_msg = tool_error or (api_response.get("error") if api_response else None) or f"Sorry, I couldn't find any returnable items for order {order_id}. This might be because the order hasn't been delivered yet or items are ineligible."
        logger.warning(f"No returnable items found or error in details for {order_id}: {error_msg}")
        return {"tool_error": error_msg, "needs_clarification": False, "next_node": "generate_response"}


    # --- Step 2 Outcome: Received SKU, Ask for Reason ---
    # This path is entered after classify_intent successfully parsed the SKU
    elif intent == "return_item_selection" and expected_step_marker == "handle_return_step_2":
        sku = state.get("item_sku_to_return")
        logger.info(f"Item SKU {sku} selected by user. Asking for reason.")
        question = f"Got it, you want to return item {sku}. Could you briefly tell me why you're returning it? (Optional, press Enter or say 'skip' to skip)"
        # Update state to ask the question
        return {
            "needs_clarification": True, # Need the reason (or skip)
            "clarification_question": question,
            "next_node": "generate_response" # Route to generate_response to ask
        }

    # --- Step 3 Outcome: Received Reason, Proceed to Submit ---
    # This path is entered after classify_intent successfully parsed the reason
    elif intent == "return_reason_provided" and expected_step_marker == "handle_return_step_3":
        logger.info("Reason provided (or skipped). Ready to submit return request.")
        # We don't need clarification. Signal to proceed to the tool execution node.
        return {"needs_clarification": False, "clarification_question": None, "next_node": "execute_tool"}

    # --- Fallback ---
    # This case should ideally not be reached if the routing and state updates are correct
    logger.warning(f"Unexpected state in handle_multi_turn_return. Intent: {intent}, Marker: {expected_step_marker}. Routing to generate_response.")
    return {"needs_clarification": False, "clarification_question": None, "next_node": "generate_response"}


def generate_response(state: GraphState) -> Dict[str, Any]:
    """Generates the final response to the user based on the graph's current state."""
    messages = state["messages"]
    intent = state.get("intent")
    api_response = state.get("api_response")
    rag_context = state.get("rag_context")
    tool_error = state.get("tool_error")
    needs_clarification = state.get("needs_clarification")
    clarification_question = state.get("clarification_question")

    logger.info(f"Generating response. Intent: {intent}, Error: {tool_error}, Clarification: {needs_clarification}")

    response_text = "" # Initialize empty response

    # --- Priority 1: Ask Clarification Question ---
    if needs_clarification and clarification_question:
        response_text = clarification_question
        logger.info(f"Generated clarification response: {response_text}")

    # --- Priority 2: Report Tool Error ---
    elif tool_error:
        # Provide a user-friendly message based on the error
        response_text = f"I encountered an issue: {tool_error}"
        logger.warning(f"Generated error response: {response_text}")

    # --- Priority 3: Respond based on Intent and Results ---
    elif intent == "knowledge_base_query":
        if rag_context:
            logger.info("Generating response from RAG context.")
            # Find the most recent user query that likely triggered this
            last_user_message_content = "your question" # Default
            relevant_message_index = -1
            if messages and isinstance(messages[-1], HumanMessage):
                 last_user_message_content = messages[-1].content
                 relevant_message_index = -1
            elif messages and len(messages) > 1 and isinstance(messages[-2], HumanMessage) and isinstance(messages[-1], (AIMessage, ToolMessage)):
                 # Look back if the last message was AI/Tool response
                 last_user_message_content = messages[-2].content
                 relevant_message_index = -2
            elif messages: # Fallback to just the last message content if possible
                 last_user_message_content = messages[-1].content

            # Construct prompt for LLM to synthesize answer
            prompt = f"""Based *only* on the following information from our knowledge base:
            --- Knowledge Base Context ---
            {rag_context}
            --- End Context ---
            Answer the user's question: "{last_user_message_content}"
            Provide a concise and helpful answer based *strictly* on the provided context.
            If the context doesn't contain the answer, state that you couldn't find the specific detail in the knowledge base. Do not make up information.
            """
            try:
                # Use the base LLM (without tools) for synthesis
                ai_message = llm.invoke(prompt)
                response_text = ai_message.content
            except Exception as e:
                logger.error(f"LLM invocation failed during RAG response generation: {e}", exc_info=True)
                response_text = "I found some information in our knowledge base, but had trouble formulating a final answer. Could you perhaps rephrase your question?"
        else:
            # RAG ran but found no context (tool_error might have been set too)
            response_text = "I looked in our knowledge base, but couldn't find specific information about that."
            logger.warning("Generating response, but RAG context was empty or lookup failed.")

    elif api_response: # Handle responses from API tools
        logger.info(f"Generating response from API result: {api_response}")
        # Format API responses nicely based on the intent
        if intent == "get_order_status":
            response_text = f"The status for order {api_response.get('order_id', 'N/A')} is: {api_response.get('status', 'Unknown')}."
        elif intent == "get_tracking_info":
            if api_response.get('tracking_number'):
                response_text = (f"Tracking for order {api_response.get('order_id', 'N/A')}: "
                                 f"Number {api_response.get('tracking_number')}, "
                                 f"Carrier: {api_response.get('carrier', 'N/A')}, "
                                 f"Status: {api_response.get('status', 'N/A')}.")
            else: # Tracking not available yet case
                response_text = (f"Tracking information is not yet available for order {api_response.get('order_id', 'N/A')}. "
                                 f"Current status: {api_response.get('status', 'Unavailable')}.")
        # Check for successful return submission response
        elif intent == "return_reason_provided" and api_response.get("return_id"):
            response_text = f"Success! {api_response.get('message', 'Return initiated.')} Your return ID is {api_response.get('return_id')}."
        # Handle other potential API responses or fallback
        else:
            # Generic response if specific formatting isn't defined, or for errors missed earlier
            error_msg = api_response.get('error') or api_response.get('message')
            if error_msg:
                 response_text = f"There was an issue: {error_msg}"
            else:
                 response_text = f"I received the following details: {str(api_response)}"

    # --- Priority 4: Handle Simple Intents (Greeting/Goodbye) ---
    # These might be set by classify_intent if implemented there
    elif intent == "greeting":
        response_text = "Hello! How can I help you with your order or answer your questions?"
    elif intent == "goodbye":
        response_text = "Goodbye! Let me know if you need anything else."

    # --- Fallback Response ---
    else:
        # If none of the above conditions met (e.g., unsupported intent, unexpected state)
        response_text = "I'm sorry, I can't assist with that specific request right now. I can help with order status, tracking, returns, and answer general questions from our FAQ."
        # Optionally, use LLM's direct response if classify_intent produced one without tool calls
        last_ai_message = messages[-1] if messages and isinstance(messages[-1], AIMessage) else None
        if last_ai_message and last_ai_message.content and not hasattr(last_ai_message, 'tool_calls') or not last_ai_message.tool_calls:
             response_text = last_ai_message.content

    logger.info(f"Final generated response: {response_text}")

    # Return the final AIMessage to be appended to the state's messages list by LangGraph
    # Critical: Only return the state key(s) to be updated.
    return {"messages": [AIMessage(content=response_text)]}