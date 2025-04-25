# bot/nodes/generation.py
import logging
from typing import Dict, Any

from langchain_core.messages import AIMessage, HumanMessage

from ..llm import llm
from ..state import GraphState

logger = logging.getLogger(__name__)


def generate_final_response(state: GraphState) -> Dict[str, Any]:
    """
    Generates the final response to the user based on the current state,
    including API responses, RAG context, errors, or clarification questions.
    """
    messages = state["messages"]
    intent = state.get("intent")
    api_response = state.get("api_response")
    rag_context = state.get("rag_context")
    tool_error = state.get("tool_error")
    needs_clarification = state.get("needs_clarification")
    clarification_question = state.get("clarification_question")
    direct_llm_response = state.get("final_llm_response")  # From classify_intent if LLM gave direct answer

    response_text = ""

    # --- Prioritize Clarification ---
    if needs_clarification and clarification_question:
        logger.debug("Generating clarification question.")
        response_text = clarification_question

    # --- Prioritize Errors ---
    elif tool_error:
        logger.warning(f"Generating response based on tool error: {tool_error}")
        # Make error messages slightly more user-friendly
        if "not found" in tool_error.lower():
            response_text = f"Sorry, I couldn't find the requested information. Details: {tool_error}"
        elif "not eligible" in tool_error.lower() or "cannot return" in tool_error.lower():
            response_text = f"It seems there's an issue with eligibility for your request: {tool_error}"
        else:
            response_text = f"I encountered an issue processing your request: {tool_error}"

    # --- Handle Direct LLM Response ---
    elif direct_llm_response:
        logger.debug("Using direct response content from LLM.")
        response_text = direct_llm_response

    # --- Handle RAG ---
    elif intent == "knowledge_base_query" or rag_context:
        logger.debug(f"Generating response based on RAG context (present: {bool(rag_context)}).")
        if rag_context:
            last_user_message_content = "your question"  # Fallback
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    last_user_message_content = msg.content
                    break
            prompt = f"""Based *only* on the following information from our knowledge base:
            --- Knowledge Base Context ---
            {rag_context}
            --- End Context ---

            Answer the user's question: \"{last_user_message_content}\"

            Provide a concise and helpful answer based *strictly* on the provided context.
            If the context doesn't contain the answer, state that you couldn't find the specific detail in the knowledge base. Do not make up information.
            If the user's query is very general and the context provides relevant info (like return policy), summarize the key points from the context.
            """
            try:
                ai_message = llm.invoke(prompt)
                response_text = ai_message.content
                logger.info("Successfully generated response from RAG context.")
            except Exception as e:
                logger.error(f"Error during RAG response generation LLM call: {e}", exc_info=True)
                response_text = "I found some information, but had trouble formulating a final answer. Please ask again or try rephrasing."
        else:
            # RAG was attempted (intent=kb_query) but no context found/retrieved
            response_text = "I looked in our knowledge base, but couldn't find specific information about that topic."

    # --- Handle API Responses ---
    elif api_response:
        logger.debug(f"Generating response based on API response for intent: {intent}.")
        if intent == "get_order_status":
            response_text = f"The status for order {api_response.get('order_id', 'N/A')} is: {api_response.get('status', 'Unknown')}."
        elif intent == "get_tracking_info":
            if api_response.get('tracking_number'):
                response_text = (
                    f"Tracking for order {api_response.get('order_id', 'N/A')}: "
                    f"Number: {api_response.get('tracking_number')}, "
                    f"Carrier: {api_response.get('carrier', 'N/A')}, "
                    f"Status: {api_response.get('status', 'N/A')}."
                )
            else:
                # Handle cases like 'Tracking not available yet' gracefully
                status_msg = api_response.get('status', 'Unavailable')
                response_text = (
                    f"Tracking information for order {api_response.get('order_id', 'N/A')} "
                    f"is currently: {status_msg}."
                )
        # Check specifically for return success (handled by submit_return_request node now)
        elif intent == "initiate_return" and api_response.get("return_id"):
            response_text = f"Success! {api_response.get('message', 'Return initiated.')} Your return ID is {api_response.get('return_id')}."
        # Generic API response formatting (fallback)
        else:
            # Avoid showing generic dict if possible, check for 'message'
            if api_response.get('message'):
                response_text = api_response['message']
            elif api_response.get('status'):
                response_text = f"The current status is: {api_response['status']}"
            else:
                # Fallback, should be rare
                logger.warning(f"Generating generic response for API result: {api_response}")
                response_text = "I have processed your request based on the available information."

    # --- Handle Simple Intents ---
    elif intent == "greeting":
        response_text = "Hello! How can I assist you with orders, tracking, returns, or general questions today?"
    elif intent == "goodbye":
        response_text = "Goodbye! Feel free to reach out if you need anything else."

    # --- Fallback / Unsupported ---
    else:
        logger.info(f"Generating fallback/unsupported response for intent: {intent}.")
        response_text = "I'm sorry, I couldn't process that specific request. I can help with checking order status, tracking, initiating returns, or answering questions from our FAQ and policies."

    # Append the final response to messages
    if not response_text:
        logger.error("Response generation resulted in empty text. Providing fallback.")
        response_text = "I'm sorry, I encountered an unexpected issue. Could you please try rephrasing?"

    state["messages"].append(AIMessage(content=response_text))

    # Clear transient state fields before finishing
    final_state_update = {
        "api_response": None,
        "rag_context": None,
        "tool_error": None,
        "needs_clarification": False,  # Should be false unless explicitly set by a node for next turn
        "clarification_question": None,
        "final_llm_response": None,
        # Don't clear item_sku_to_return etc. here, handled by return nodes
    }

    # Only return the fields to update, LangGraph merges them
    return final_state_update
