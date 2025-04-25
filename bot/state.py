from typing import TypedDict, List, Optional, Dict, Any, Literal
# Keep AIMessage import if using it directly as type hint below
from langchain_core.messages import BaseMessage, AIMessage

Intent = Literal[
    "get_order_status",
    "get_tracking_info",
    "initiate_return",
    "knowledge_base_query",
    "clarification_needed",
    "return_item_selection",
    "return_reason_provided",
    "unsupported",
    "greeting",
    "goodbye"
]


class GraphState(TypedDict):
    messages: List[BaseMessage]
    intent: Optional[Intent] = None
    order_id: Optional[str] = None
    item_sku_to_return: Optional[str] = None
    return_reason: Optional[str] = None
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    available_return_items: Optional[List[Dict[str, Any]]] = None
    rag_context: Optional[str] = None
    api_response: Optional[Dict[str, Any]] = None
    tool_error: Optional[str] = None
    next_node: Optional[str] = None
    # --- Add temporary field ---
    latest_ai_response: Optional[AIMessage] = None # To pass tool calls without saving to history yet
    # --- End temporary field ---
    final_llm_response: Optional[str] = None # Added in generation.py test, ensure it's here