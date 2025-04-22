# bot/state.py

from typing import TypedDict, List, Optional, Dict, Any, Literal
from langchain_core.messages import BaseMessage

# Define possible intents the bot can recognize
Intent = Literal[
    "get_order_status",
    "get_tracking_info",
    "initiate_return",
    "knowledge_base_query",
    "clarification_needed", # Bot needs more info (e.g. missing order_id)
    "return_item_selection", # User provided item to return
    "return_reason_provided", # User provided reason
    "unsupported",
    "greeting",
    "goodbye"
]

class GraphState(TypedDict):
    """Represents the state of our graph."""

    messages: List[BaseMessage]  # Conversation history

    # Extracted information
    intent: Optional[Intent] = None
    order_id: Optional[str] = None
    item_sku_to_return: Optional[str] = None
    return_reason: Optional[str] = None

    # Information needed for multi-turn interactions
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    available_return_items: Optional[List[Dict[str, Any]]] = None # List of items eligible for return

    # Results from tools/nodes
    rag_context: Optional[str] = None       # Context retrieved from KB
    api_response: Optional[Dict[str, Any]] = None # Response from Mock API
    tool_error: Optional[str] = None        # Error message if a tool failed

    # Control flow
    next_node: Optional[str] = None         # Explicitly guide to next node if needed