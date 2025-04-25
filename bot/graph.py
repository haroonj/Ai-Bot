# bot/graph.py
import logging

from langgraph.graph import StateGraph, END

# Import new node functions
from .nodes.classification import classify_intent
from .nodes.execution import execute_tool_call, execute_rag_lookup
from .nodes.return_flow import handle_return_step, submit_return_request
from .nodes.generation import generate_final_response

from .state import GraphState

logger = logging.getLogger(__name__)

# --- Conditional Edge Logic ---

def route_after_classification(state: GraphState) -> str:
    """Determines the next node based on the classification result."""
    next_node = state.get("next_node")
    logger.debug(f"Routing after classification. Determined next node: {next_node}")
    if next_node == "execute_tool_call":
        return "execute_tool_call"
    elif next_node == "execute_rag_lookup":
        return "execute_rag_lookup"
    elif next_node == "handle_return_step":
        return "handle_return_step"
    elif next_node == "generate_final_response":
        return "generate_final_response"
    else:
        logger.warning(f"Unknown next_node '{next_node}' after classification. Defaulting to generate_final_response.")
        return "generate_final_response"

def route_after_execution(state: GraphState) -> str:
    """Determines the next node after tool/RAG execution."""
    intent = state.get("intent")
    tool_error = state.get("tool_error")
    api_response = state.get("api_response") # Relevant for API calls

    if tool_error:
        logger.debug("Routing after execution: Tool error detected, going to generation.")
        return "generate_final_response"

    # If execution was part of a return flow (e.g., submitting), go generate
    if intent == "initiate_return" and state.get("item_sku_to_return") is None: # Indicates successful submission potentially cleared state
         logger.debug("Routing after execution: Return seems completed, going to generation.")
         return "generate_final_response"

    # If an API call successfully fetched order details for a return, route to return handler
    if intent == "initiate_return" and api_response and api_response.get("items") is not None:
         logger.debug("Routing after execution: API call fetched return details, going to handle_return_step.")
         # This condition might be redundant if classify routes directly to handle_return_step
         # Keep it as a fallback? Or simplify classify_intent edge. Let's rely on classify_intent first.
         # return "handle_return_step" # Revisit if needed

    # Default: After execution (API or RAG), generate the response
    logger.debug("Routing after execution: Defaulting to generate_final_response.")
    return "generate_final_response"


def route_after_return_step(state: GraphState) -> str:
    """ Routes after a step in the multi-turn return flow."""
    next_node = state.get("next_node") # Node function should set this
    needs_clarification = state.get("needs_clarification")

    logger.debug(f"Routing after return step. Next node hint: {next_node}, Needs Clarification: {needs_clarification}")

    if next_node == "submit_return_request":
        return "submit_return_request"
    elif next_node == "generate_final_response": # Usually when asking a question
        return "generate_final_response"
    else:
        # Fallback, should indicate an issue in handle_return_step logic
        logger.error(f"Unexpected state after handle_return_step. next_node='{next_node}'. Defaulting to generate.")
        return "generate_final_response"

# --- Graph Definition ---

def create_graph() -> StateGraph:
    """Creates and configures the LangGraph StateGraph with refactored nodes."""
    workflow = StateGraph(GraphState)

    logger.info("Adding refactored nodes to the graph...")
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("execute_tool_call", execute_tool_call)
    workflow.add_node("execute_rag_lookup", execute_rag_lookup)
    workflow.add_node("handle_return_step", handle_return_step)
    workflow.add_node("submit_return_request", submit_return_request)
    workflow.add_node("generate_final_response", generate_final_response)

    logger.info("Defining edges for the refactored graph...")

    workflow.set_entry_point("classify_intent")

    # Edges from Classification
    workflow.add_conditional_edges(
        "classify_intent",
        route_after_classification,
        {
            "execute_tool_call": "execute_tool_call",
            "execute_rag_lookup": "execute_rag_lookup",
            "handle_return_step": "handle_return_step",
            "generate_final_response": "generate_final_response",
        }
    )

    # Edges after Tool Execution (API or RAG)
    # Simple path for now: always generate response after execution
    workflow.add_edge("execute_tool_call", "generate_final_response")
    workflow.add_edge("execute_rag_lookup", "generate_final_response")
    # Revisit route_after_execution if more complex branching needed later

    # Edges within Return Flow
    workflow.add_conditional_edges(
        "handle_return_step",
        route_after_return_step,
        {
            "submit_return_request": "submit_return_request",
            "generate_final_response": "generate_final_response",
        }
    )

    # Edge after Submitting Return
    workflow.add_edge("submit_return_request", "generate_final_response")

    # Final Response Generation leads to End
    workflow.add_edge("generate_final_response", END)

    logger.info("Graph definition complete.")
    return workflow


# --- Compile and Export ---
memory = None # Add memory later if needed

try:
    app = create_graph().compile(checkpointer=memory)
    logger.info("LangGraph compiled successfully with refactored nodes.")

    # Optional: Draw graph diagram
    try:
        graph_png = "ecommerce_bot_graph_refactored.png"
        # Make sure the graph object can be drawn. For compiled graphs, this might need specific library versions or methods.
        # If app.get_graph() exists and works:
        if hasattr(app, 'get_graph') and callable(getattr(app, 'get_graph', None)):
             graph_viz = app.get_graph()
             # Check if draw_graphviz method exists
             if hasattr(graph_viz, 'draw_graphviz') and callable(getattr(graph_viz, 'draw_graphviz', None)):
                  graph_viz.draw_graphviz(output_file_path=graph_png)
                  logger.info(f"Graph diagram saved to {graph_png}")
             else:
                  logger.warning("Compiled graph object does not have 'draw_graphviz' method. Cannot generate PNG.")
        else:
            logger.warning("Cannot retrieve graph structure from compiled app. Cannot generate graph diagram.")

        # Fallback to ASCII print if PNG fails or isn't supported
        if hasattr(app, 'get_graph') and callable(getattr(app, 'get_graph', None)):
            graph_viz = app.get_graph()
            if hasattr(graph_viz, 'print_ascii') and callable(getattr(graph_viz, 'print_ascii', None)):
                 ascii_graph = graph_viz.print_ascii()
                 print("\n--- Graph ASCII Diagram (Refactored) ---")
                 print(ascii_graph)
                 print("----------------------------------------\n")
            else:
                 logger.warning("Cannot print ASCII graph, method not found.")
        else:
            logger.warning("Cannot retrieve graph structure for ASCII print.")

    except ImportError:
        logger.warning("Could not draw graph diagram. Missing libraries (e.g., `pip install pygraphviz`).")
    except Exception as draw_error:
        logger.warning(f"Could not draw graph diagram (ensure Graphviz is installed and in PATH, or library issue): {draw_error}")


except Exception as compile_error:
    logger.critical(f"Failed to compile refactored LangGraph: {compile_error}", exc_info=True)
    app = None


def get_runnable():
    """Returns the compiled LangGraph app."""
    if not app:
        raise RuntimeError("LangGraph application failed to compile.")
    return app