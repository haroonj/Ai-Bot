# bot/nodes/execution.py
import logging
from typing import Dict, Any

from langchain_core.messages import AIMessage, ToolMessage, HumanMessage

from ..state import GraphState
from ..tools import available_tools, knowledge_base_lookup

logger = logging.getLogger(__name__)


def execute_tool_call(state: GraphState) -> Dict[str, Any]:
    """
    Executes the tool call found in the 'latest_ai_response' state field.
    Updates state with the API response or tool error and adds ToolMessage to history.
    """
    messages = state["messages"] # Still needed to add ToolMessage
    api_response = None
    tool_error = None
    tool_message_content = "Tool execution did not proceed as expected."

    # --- Read from the temporary state field ---
    last_ai_response = state.get("latest_ai_response")
    # --- End change ---

    # --- Check if the response exists and has tool calls ---
    if not last_ai_response or not last_ai_response.tool_calls:
        tool_error = "Attempted to execute tool, but no valid tool call found in latest AI response."
        logger.error(tool_error)
        # Clear the temporary field even on error
        return {"tool_error": tool_error, "api_response": None, "latest_ai_response": None}
    # --- End check ---

    # Execute the first tool call found
    call = last_ai_response.tool_calls[0]
    tool_name = call['name']
    tool_args = call['args']
    tool_id = call['id']
    logger.info(f"Executing tool '{tool_name}' from latest AI response with args: {tool_args}")

    try:
        tool_func = next((t for t in available_tools if t.name == tool_name), None)
        if tool_func:
            # RAG lookups should be routed directly to execute_rag_lookup by the graph,
            # so this check might be less critical now, but keep for safety.
            if tool_name == knowledge_base_lookup.name:
                 logger.warning(f"Tool call requested '{tool_name}', redirecting to RAG execution node.")
                 # Ideally graph routes this, but handle if it reaches here
                 rag_result = execute_rag_lookup(state, explicit_query=tool_args.get("query"))
                 # Make sure to clear latest_ai_response and return RAG result
                 rag_result["latest_ai_response"] = None
                 return rag_result

            # Execute API tool
            result = tool_func.invoke(tool_args)
            logger.info(f"Tool '{tool_name}' executed successfully via tool call.")

            api_response = result if isinstance(result, dict) else {"result": str(result)}
            if "error" in api_response:
                tool_error = api_response["error"]
                tool_message_content = f"Tool execution returned an error: {tool_error}"
                logger.warning(f"Tool '{tool_name}' resulted in error: {tool_error}")
            else:
                tool_message_content = f"Successfully called API tool {tool_name}."

        else:
            tool_error = f"Tool '{tool_name}' not found in available tools."
            logger.error(tool_error)
            tool_message_content = f"Error: Tool '{tool_name}' is not available."

    except Exception as e:
        tool_error = f"Failed to execute tool '{tool_name}' via tool call: {e}"
        logger.error(tool_error, exc_info=True)
        tool_message_content = f"An unexpected error occurred while executing tool {tool_name}."

    # --- Add the ToolMessage to the actual history ---
    # Only add if execution proceeded (even if it resulted in error)
    if tool_id:
        state["messages"].append(ToolMessage(content=tool_message_content, tool_call_id=tool_id))
    # --- End change ---

    # --- Clear the temporary AI response field ---
    return {
        "api_response": api_response,
        "tool_error": tool_error,
        "rag_context": None,
        "next_node": None,
        "latest_ai_response": None # Clear the temporary field
    }


def execute_rag_lookup(state: GraphState, explicit_query: str | None = None) -> Dict[str, Any]:
    """
    Performs a knowledge base lookup using the RAG retriever.
    Uses the last human message as the query unless an explicit_query is provided.
    Updates state with rag_context or tool_error.
    """
    messages = state["messages"]
    rag_context = None
    tool_error = None

    query = explicit_query
    if not query:
        # Find the last human message content for the query
        last_user_message = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        if not last_user_message:
            tool_error = "Could not find user query for RAG lookup."
            logger.warning(tool_error)
            return {"rag_context": None, "tool_error": tool_error}
        query = last_user_message

    logger.info(f"Executing RAG lookup for query: '{query}'")

    try:
        rag_context_result = knowledge_base_lookup.invoke({"query": query})

        if isinstance(rag_context_result, dict) and "error" in rag_context_result:
            tool_error = rag_context_result["error"]
            logger.warning(f"RAG lookup failed: {tool_error}")
        elif isinstance(rag_context_result, str) and rag_context_result.strip():
            rag_context = rag_context_result
            logger.info(f"RAG lookup successful. Context length: {len(rag_context)}")
            # Add a system/tool message indicating success? Optional.
            # state["messages"].append(ToolMessage(content=f"Successfully looked up knowledge base for '{query}'.", tool_call_id="rag_lookup"))
        else:
            rag_context = None  # Explicitly set to None if no results
            logger.warning(f"RAG lookup for '{query}' returned no results.")
            # Set a specific message? Or let generator handle no context?
            # tool_error = "No relevant information found in the knowledge base." # Option

    except Exception as e:
        tool_error = f"Failed to execute RAG lookup: {e}"
        logger.error(tool_error, exc_info=True)
        rag_context = None

    # At the end of execute_rag_lookup, add:
    final_updates = {
        "rag_context": rag_context,
        "tool_error": tool_error,
        "api_response": None,
        "next_node": "generate_final_response",
        "latest_ai_response": None # Also clear here for consistency
    }
    return final_updates