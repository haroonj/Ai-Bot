import logging

from langgraph.graph import StateGraph, END

from .nodes import (
    classify_intent,
    execute_tool,
    handle_multi_turn_return,
    generate_response
)
from .state import GraphState

logger = logging.getLogger(__name__)


def decide_next_node_after_multi_turn(state: GraphState) -> str:
    if state.get("needs_clarification"):
        logger.debug("Routing from handle_multi_turn_return to generate_response (needs clarification)")
        return "generate_response"
    elif state.get("next_node") == "execute_tool" or state.get("intent") == "return_reason_provided":
        logger.debug("Routing from handle_multi_turn_return to execute_tool (reason provided/ready to submit)")
        return "execute_tool"
    else:
        logger.warning("Unexpected state after handle_multi_turn_return, routing to generate_response as fallback.")
        return "generate_response"


def create_graph() -> StateGraph:
    """Creates and configures the LangGraph StateGraph."""
    workflow = StateGraph(GraphState)

    logger.info("Adding nodes to the graph...")
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("execute_tool", execute_tool)
    workflow.add_node("handle_multi_turn_return", handle_multi_turn_return)
    workflow.add_node("generate_response", generate_response)

    logger.info("Defining edges for the graph...")

    workflow.set_entry_point("classify_intent")

    workflow.add_conditional_edges(
        "classify_intent",
        lambda state: state.get("next_node") or "generate_response",
        {
            "execute_tool": "execute_tool",
            "handle_return_step_1": "execute_tool",
            "handle_return_step_2": "handle_multi_turn_return",
            "handle_return_step_3": "handle_multi_turn_return",
            "generate_response": "generate_response",
        }
    )

    workflow.add_conditional_edges(
        "execute_tool",
        lambda state:
        "handle_multi_turn_return" if state.get("intent") == "initiate_return" and not state.get(
            "tool_error") and state.get("api_response") and state.get("api_response", {}).get("items")
        else "generate_response",
        {
            "handle_multi_turn_return": "handle_multi_turn_return",
            "generate_response": "generate_response",
        }
    )

    workflow.add_conditional_edges(
        "handle_multi_turn_return",
        decide_next_node_after_multi_turn,
        {
            "generate_response": "generate_response",
            "execute_tool": "execute_tool",
        }
    )

    workflow.add_edge("generate_response", END)

    logger.info("Graph definition complete.")
    return workflow


memory = None

try:
    app = create_graph().compile(checkpointer=memory)
    logger.info("LangGraph compiled successfully.")

    try:
        graph_png = "ecommerce_bot_graph.png"
        if app and hasattr(app, 'get_graph'):
            app.get_graph().draw_graphviz(output_file_path=graph_png)
            logger.info(f"Graph diagram saved to {graph_png}")
        else:
            logger.warning("Compiled app object is invalid, cannot generate graph diagram.")
    except ImportError as ie:
        logger.warning(f"Could not draw graph diagram. Missing libraries (try `pip install pygraphviz`): {ie}")
    except Exception as draw_error:
        logger.warning(f"Could not draw graph diagram (ensure Graphviz is installed and in PATH): {draw_error}")
        if app and hasattr(app, 'get_graph'):
            try:
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
    app = None


def get_runnable():
    """Returns the compiled LangGraph app."""
    if not app:
        raise RuntimeError("LangGraph application failed to compile.")
    return app
