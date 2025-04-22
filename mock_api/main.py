# mock_api/main.py

import logging
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from typing import List, Optional

from .sample_data import get_order, create_return

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Mock E-commerce API")

# --- Pydantic Models ---

class Item(BaseModel):
    sku: str
    name: str
    price: float

class OrderStatus(BaseModel):
    order_id: str
    status: str

class TrackingInfo(BaseModel):
    order_id: str
    tracking_number: Optional[str] = None
    carrier: Optional[str] = None
    status: Optional[str] = None # Different from order status, e.g., "In Transit"

class OrderDetails(BaseModel):
    order_id: str
    items: List[Item]
    status: str
    delivered: bool

class ReturnRequest(BaseModel):
    order_id: str
    sku: str
    reason: Optional[str] = None

class ReturnResponse(BaseModel):
    return_id: Optional[str] = None
    status: str
    message: str


# --- API Endpoints ---

@app.get("/orders/{order_id}/status", response_model=OrderStatus)
async def get_order_status_endpoint(order_id: str):
    logger.info(f"Received status request for order: {order_id}")
    order = get_order(order_id)
    if not order:
        logger.warning(f"Order not found: {order_id}")
        raise HTTPException(status_code=404, detail="Order not found")
    return OrderStatus(order_id=order_id, status=order["status"])

@app.get("/orders/{order_id}/tracking", response_model=TrackingInfo)
async def get_tracking_info_endpoint(order_id: str):
    logger.info(f"Received tracking request for order: {order_id}")
    order = get_order(order_id)
    if not order:
        logger.warning(f"Order not found: {order_id}")
        raise HTTPException(status_code=404, detail="Order not found")
    if not order["tracking_number"]:
         return TrackingInfo(order_id=order_id, status="Tracking not available yet")
    return TrackingInfo(
        order_id=order_id,
        tracking_number=order["tracking_number"],
        carrier=order["carrier"],
        status=order["tracking_status"]
    )

@app.get("/orders/{order_id}/details", response_model=OrderDetails)
async def get_order_details_endpoint(order_id: str):
    logger.info(f"Received details request for order: {order_id}")
    order = get_order(order_id)
    if not order:
        logger.warning(f"Order not found: {order_id}")
        raise HTTPException(status_code=404, detail="Order not found")
    return OrderDetails(
        order_id=order_id,
        items=[Item(**item) for item in order["items"]],
        status=order["status"],
        delivered=order["delivered"]
    )

@app.post("/returns", response_model=ReturnResponse)
async def initiate_return_endpoint(request: ReturnRequest = Body(...)):
    logger.info(f"Received return request: {request.dict()}")
    return_id, message = create_return(request.order_id, request.sku, request.reason)
    if return_id:
        logger.info(f"Return successful: {return_id} - {message}")
        return ReturnResponse(return_id=return_id, status="Return Initiated", message=message)
    else:
        logger.warning(f"Return failed for order {request.order_id}: {message}")
        # Determine appropriate status code based on message
        status_code = 404 if "not found" in message else 400
        raise HTTPException(status_code=status_code, detail=message)

@app.get("/")
async def root():
    return {"message": "Mock E-commerce API is running"}

# To run this: uvicorn mock_api.main:app --reload --port 8001