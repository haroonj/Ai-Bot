import logging
from typing import Optional, Dict, Any

import httpx
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .config import settings
from .vector_store import get_retriever

logger = logging.getLogger(__name__)
MOCK_API_URL = str(settings.mock_api_base_url)
client = httpx.Client(base_url=MOCK_API_URL, timeout=10.0)


class OrderInfoInput(BaseModel):
    order_id: str = Field(description="The unique identifier for the customer's order.")


class ReturnInput(BaseModel):
    order_id: str = Field(description="The unique identifier for the order containing the item to return.")
    sku: str = Field(description="The Stock Keeping Unit (SKU) identifier of the item to be returned.")
    reason: Optional[str] = Field(description="Optional reason provided by the customer for the return.", default=None)


@tool("get_order_status", args_schema=OrderInfoInput)
def get_order_status(order_id: str) -> Dict[str, Any]:
    """Looks up the current status of a specific order using its Order ID."""
    try:
        response = client.get(f"/orders/{order_id}/status")
        response.raise_for_status()
        logger.info(f"API call get_order_status for {order_id} successful.")
        return response.json()
    except httpx.HTTPStatusError as e:
        logger.warning(f"API Error getting status for {order_id}: {e.response.status_code} - {e.response.text}")
        if e.response.status_code == 404:
            return {"error": f"Order ID '{order_id}' not found."}
        else:
            return {"error": f"API error fetching status for order {order_id}: {e.response.status_code}"}
    except httpx.RequestError as e:
        logger.error(f"Request Error getting status for {order_id}: {e}", exc_info=True)
        return {"error": f"Could not connect to the order system to fetch status for {order_id}."}


@tool("get_tracking_info", args_schema=OrderInfoInput)
def get_tracking_info(order_id: str) -> Dict[str, Any]:
    """Retrieves the shipping tracking information (tracking number, carrier, status) for a specific order using its Order ID."""
    try:
        response = client.get(f"/orders/{order_id}/tracking")
        response.raise_for_status()
        logger.info(f"API call get_tracking_info for {order_id} successful.")
        return response.json()
    except httpx.HTTPStatusError as e:
        logger.warning(f"API Error getting tracking for {order_id}: {e.response.status_code} - {e.response.text}")
        if e.response.status_code == 404:
            return {"error": f"Order ID '{order_id}' not found."}
        else:
            return {"error": f"API error fetching tracking for order {order_id}: {e.response.status_code}"}
    except httpx.RequestError as e:
        logger.error(f"Request Error getting tracking for {order_id}: {e}", exc_info=True)
        return {"error": f"Could not connect to the order system to fetch tracking for {order_id}."}


@tool("get_order_details", args_schema=OrderInfoInput)
def get_order_details(order_id: str) -> Dict[str, Any]:
    """Fetches the detailed information about an order, including the list of items, required for initiating a return."""
    try:
        response = client.get(f"/orders/{order_id}/details")
        response.raise_for_status()
        logger.info(f"API call get_order_details for {order_id} successful.")
        details = response.json()
        if details.get("delivered"):
            return details
        else:
            return {"error": f"Order {order_id} is not yet delivered or not eligible for return based on details.",
                    "details": details}
    except httpx.HTTPStatusError as e:
        logger.warning(f"API Error getting details for {order_id}: {e.response.status_code} - {e.response.text}")
        if e.response.status_code == 404:
            return {"error": f"Order ID '{order_id}' not found."}
        else:
            return {"error": f"API error fetching details for order {order_id}: {e.response.status_code}"}
    except httpx.RequestError as e:
        logger.error(f"Request Error getting details for {order_id}: {e}", exc_info=True)
        return {"error": f"Could not connect to the order system to fetch details for {order_id}."}


@tool("initiate_return_request", args_schema=ReturnInput)
def initiate_return_request(order_id: str, sku: str, reason: Optional[str] = None) -> Dict[str, Any]:
    """Submits a request to initiate a return for a specific item (SKU) from a given order (Order ID). Optionally include a reason."""
    payload = {"order_id": order_id, "sku": sku, "reason": reason}
    try:
        response = client.post("/returns", json=payload)
        response.raise_for_status()
        logger.info(f"API call initiate_return_request for order {order_id}, sku {sku} successful.")
        return response.json()
    except httpx.HTTPStatusError as e:
        logger.warning(
            f"API Error initiating return for {order_id}, sku {sku}: {e.response.status_code} - {e.response.text}")
        try:
            error_detail = e.response.json().get("detail", "Unknown API error during return.")
        except Exception:
            error_detail = f"API error initiating return: Status {e.response.status_code}"
        return {"error": error_detail}
    except httpx.RequestError as e:
        logger.error(f"Request Error initiating return for {order_id}, sku {sku}: {e}", exc_info=True)
        return {"error": f"Could not connect to the return system for order {order_id}."}


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
