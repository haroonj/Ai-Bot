# bot/graph.py

import logging
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver # For potential stateful runs

# Make sure GraphState is explicitly imported if needed by the decision function type hint
from .state import GraphState
from .nodes import (
    classify_intent,
    execute_tool,
    handle_multi_turn_return,
    generate_response
)

logger = logging.getLogger(__name__)

# --- Decision function for handle_multi_turn_return node ---
def decide_next_node_after_multi_turn(state: GraphState) -> str:
    """Determines the next node after the multi-turn return handling node."""
    if state.get("needs_clarification"):
        # If the multi-turn node decided clarification is needed (e.g., asking for reason),
        # go generate that response/question.
        logger.debug("Routing from handle_multi_turn_return to generate_response (needs clarification)")
        return "generate_response"
    # FIX: Check intent or next_node signal from handle_multi_turn_return
    # If the intent indicates the reason was provided, or the node explicitly set next_node to execute_tool
    elif state.get("next_node") == "execute_tool" or state.get("intent") == "return_reason_provided":
        # The next step is to execute the tool to submit the return.
        logger.debug("Routing from handle_multi_turn_return to execute_tool (reason provided/ready to submit)")
        return "execute_tool"
    else:
        # Fallback case - If error occurred getting details or other unexpected path
        logger.warning("Unexpected state after handle_multi_turn_return, routing to generate_response as fallback.")
        return "generate_response"

# --- Graph Definition ---
def create_graph() -> StateGraph:
    """Creates and configures the LangGraph StateGraph."""
    workflow = StateGraph(GraphState)

    # --- Add Nodes ---
    logger.info("Adding nodes to the graph...")
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("execute_tool", execute_tool)
    workflow.add_node("handle_multi_turn_return", handle_multi_turn_return)
    workflow.add_node("generate_response", generate_response)

    # --- Define Edges ---
    logger.info("Defining edges for the graph...")

    # Entry point
    workflow.set_entry_point("classify_intent")

    # Conditional edges from classification
    workflow.add_conditional_edges(
        "classify_intent",
        # Decision function: inspects state to determine the next node
        lambda state: state.get("next_node") or "generate_response", # Default to response if no specific next node
        {
            "execute_tool": "execute_tool",              # Standard tool call or KB query
            "handle_return_step_1": "execute_tool",      # Trigger get_order_details (via execute_tool)
            "handle_return_step_2": "handle_multi_turn_return", # Go ask for reason
            "handle_return_step_3": "handle_multi_turn_return", # Go trigger submission (decision routes to execute_tool)
            "generate_response": "generate_response",    # For clarification or errors
        }
    )

    # Edges after tool execution
    workflow.add_conditional_edges(
        "execute_tool",
        # Decide based on intent and errors after tool runs
        lambda state:
            # If we just fetched details for a return successfully, go ask for SKU
            "handle_multi_turn_return" if state.get("intent") == "initiate_return" and not state.get("tool_error") and state.get("api_response") and state.get("api_response", {}).get("items")
            # Otherwise, generate response (covers success, errors, RAG results, successful return submission)
            else "generate_response",
        {
            "handle_multi_turn_return": "handle_multi_turn_return", # Go ask for SKU
            "generate_response": "generate_response",         # Show result/error/ask next
        }
    )


    # Edges after multi-turn return handling
    workflow.add_conditional_edges(
        "handle_multi_turn_return",
         # Use the dedicated decision function defined above
        decide_next_node_after_multi_turn,
        {
            # These map the *return values* of the decision function to node names
            "generate_response": "generate_response", # Ask clarification question or show error
            "execute_tool": "execute_tool",        # Submit the return request
        }
    )


    # Final response generation leads to the end
    workflow.add_edge("generate_response", END)

    logger.info("Graph definition complete.")
    return workflow

# --- Compile Graph ---
# MemorySaver can be added for stateful conversations across requests,
# but requires more setup (e.g., unique thread_id per user session).
# For stateless hackathon: memory = None
memory = None # MemorySaver()

try:
    app = create_graph().compile(checkpointer=memory)
    logger.info("LangGraph compiled successfully.")

    # --- Generate Graph Diagram ---
    try:
        graph_png = "ecommerce_bot_graph.png"
        # Check if 'app' exists and has 'get_graph' method before calling draw
        if app and hasattr(app, 'get_graph'):
            # Ensure necessary libraries are installed: pip install pygraphviz playwright
            # and run: playwright install
            # app.get_graph().draw_mermaid_png(output_file_path=graph_png) # Mermaid often fails locally
            app.get_graph().draw_graphviz(output_file_path=graph_png) # Try graphviz instead
            logger.info(f"Graph diagram saved to {graph_png}")
        else:
            logger.warning("Compiled app object is invalid, cannot generate graph diagram.")
    except ImportError as ie:
         logger.warning(f"Could not draw graph diagram. Missing libraries (try `pip install pygraphviz`): {ie}")
    except Exception as draw_error:
        logger.warning(f"Could not draw graph diagram (ensure Graphviz is installed and in PATH): {draw_error}")
        # Attempt ASCII fallback only if app is valid
        if app and hasattr(app, 'get_graph'):
            try:
                 # Fallback to ASCII
                 ascii_graph = app.get_graph().print_ascii()
                 print("\n--- Graph ASCII Diagram ---")
                 print(ascii_graph)
                 print("-------------------------\n")
            except Exception as ascii_error:
                 logger.warning(f"Could not print ASCII graph: {ascii_error}")
        else:
             logger.warning("Skipping ASCII graph print because app object is invalid.")


except Exception as compile_error:
    logger.critical(f"Failed to compile LangGraph: {compile_error}", exc_info=True)
    app = None # Ensure app is None if compilation fails

def get_runnable():
    """Returns the compiled LangGraph app."""
    if not app:
        raise RuntimeError("LangGraph application failed to compile.")
    return app