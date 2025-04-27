import logging
from typing import Optional, Dict, Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .vector_store import get_retriever
from mock_api.sample_data import get_order, create_return

logger = logging.getLogger(__name__)


class OrderInfoInput(BaseModel):
    order_id: str = Field(description="The unique identifier for the customer's order.")


class ReturnInput(BaseModel):
    order_id: str = Field(description="The unique identifier for the order containing the item to return.")
    sku: str = Field(description="The Stock Keeping Unit (SKU) identifier of the item to be returned.")
    reason: Optional[str] = Field(description="Optional reason provided by the customer for the return.", default=None)


@tool("get_order_status", args_schema=OrderInfoInput)
def get_order_status(order_id: str) -> Dict[str, Any]:
    """Looks up the current status of a specific order using its Order ID from internal mock data."""
    logger.info(f"Tool: Looking up status for order {order_id} in mock data.")
    order = get_order(order_id)
    if not order:
        logger.warning(f"Tool: Order ID '{order_id}' not found in mock data.")
        return {"error": f"Order ID '{order_id}' not found."}
    else:
        logger.info(f"Tool: Found status for order {order_id}: {order['status']}")
        # Return only the expected fields, matching the previous API response structure
        return {"order_id": order_id, "status": order["status"]}


@tool("get_tracking_info", args_schema=OrderInfoInput)
def get_tracking_info(order_id: str) -> Dict[str, Any]:
    """Retrieves the shipping tracking information (tracking number, carrier, status) for a specific order using its Order ID from internal mock data."""
    logger.info(f"Tool: Looking up tracking for order {order_id} in mock data.")
    order = get_order(order_id)
    if not order:
        logger.warning(f"Tool: Order ID '{order_id}' not found in mock data for tracking.")
        return {"error": f"Order ID '{order_id}' not found."}

    # Check if tracking info exists
    if order.get("tracking_number"):
        logger.info(f"Tool: Found tracking info for order {order_id}.")
        # Return the structure expected by the calling code
        return {
            "order_id": order_id,
            "tracking_number": order["tracking_number"],
            "carrier": order.get("carrier"),
            "status": order.get("tracking_status")
        }
    else:
        logger.info(f"Tool: Tracking info not available for order {order_id}.")
        # Return a specific structure indicating unavailability
        return {"order_id": order_id, "status": "Tracking not available yet"}


@tool("get_order_details", args_schema=OrderInfoInput)
def get_order_details(order_id: str) -> Dict[str, Any]:
    """Fetches the detailed information about an order, including the list of items, required for initiating a return from internal mock data."""
    logger.info(f"Tool: Looking up details for order {order_id} in mock data.")
    order = get_order(order_id)
    if not order:
        logger.warning(f"Tool: Order ID '{order_id}' not found in mock data for details.")
        return {"error": f"Order ID '{order_id}' not found."}

    logger.info(f"Tool: Found details for order {order_id}. Delivered: {order.get('delivered')}")
    details = {
        "order_id": order_id,
        "items": order.get("items", []),
        "status": order.get("status"),
        "delivered": order.get("delivered", False)
    }
    # Check eligibility based on details (e.g., must be delivered)
    if details["delivered"]:
        return details
    else:
        # Return error structure but include details for context if needed by LLM
        return {"error": f"Order {order_id} is not yet delivered or not eligible for return based on details.",
                "details": details}


@tool("initiate_return_request", args_schema=ReturnInput)
def initiate_return_request(order_id: str, sku: str, reason: Optional[str] = None) -> Dict[str, Any]:
    """Submits a request to initiate a return for a specific item (SKU) from a given order (Order ID) using internal mock logic. Optionally include a reason."""
    logger.info(f"Tool: Attempting to initiate return for order {order_id}, SKU {sku} in mock data.")
    return_id, message = create_return(order_id, sku, reason)

    if return_id:
        logger.info(f"Tool: Mock return successful: {return_id} - {message}")
        return {"return_id": return_id, "status": "Return Initiated", "message": message}
    else:
        # create_return returns the error message directly
        logger.warning(f"Tool: Mock return failed for order {order_id}, SKU {sku}: {message}")
        return {"error": message}


@tool("knowledge_base_lookup")
def knowledge_base_lookup(query: str) -> str:
    """Searches the e-commerce help documentation (knowledge base) for answers to general questions about policies, shipping, etc. Use for questions not related to specific orders."""
    logger.info(f"Performing RAG lookup for query: '{query}'")
    retriever = get_retriever()
    if not retriever:
        logger.error("RAG retriever is not available.")
        return "I apologize, my knowledge base is currently unavailable."
    try:
        results = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in results])
        if not context:
            logger.warning(f"RAG lookup for '{query}' returned no results.")
            return "I couldn't find specific information about that in our knowledge base."
        logger.info(f"RAG lookup for '{query}' successful.")
        return context
    except Exception as e:
        logger.error(f"Error during RAG retrieval for query '{query}': {e}", exc_info=True)
        return "I encountered an error while searching the knowledge base."


available_tools = [
    get_order_status,
    get_tracking_info,
    get_order_details,
    initiate_return_request,
    knowledge_base_lookup,
]

from .llm import llm

llm_with_tools = llm.bind_tools(available_tools)