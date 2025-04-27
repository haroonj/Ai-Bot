# mock_api/sample_data.py

mock_orders = {
    "123": {
        "status": "Shipped",
        "tracking_number": "TRACK987",
        "carrier": "MockExpress",
        "tracking_status": "In Transit",
        "items": [
            {"sku": "ITEM001", "name": "Wireless Mouse", "price": 25.99},
            {"sku": "ITEM002", "name": "Keyboard", "price": 75.00},
        ],
        "delivered": False,
    },
    "456": {
        "status": "Processing",
        "tracking_number": None,
        "carrier": None,
        "tracking_status": None,
        "items": [
            {"sku": "ITEM003", "name": "Webcam", "price": 50.00},
        ],
        "delivered": False,
    },
    "789": {
        "status": "Delivered",
        "tracking_number": "TRACK111",
        "carrier": "MockPost",
        "tracking_status": "Delivered",
        "items": [
            {"sku": "ITEM004", "name": "Monitor", "price": 300.00},
        ],
        "delivered": True,  # Eligible for return
    },
}

mock_returns = {}
return_counter = 1


def get_order(order_id: str):
    return mock_orders.get(order_id)


def create_return(order_id: str, sku: str, reason: str | None = None):
    global return_counter
    order = get_order(order_id)
    if not order:
        return None, "Order not found"
    if not order["delivered"]:
        return None, "Order not yet delivered, cannot return items"

    item_found = any(item["sku"] == sku for item in order["items"])
    if not item_found:
        return None, f"Item SKU {sku} not found in order {order_id}"

    # Simple return ID generation
    return_id = f"RETN{return_counter:04d}"
    return_counter += 1
    mock_returns[return_id] = {
        "order_id": order_id,
        "sku": sku,
        "reason": reason,
        "status": "Return Initiated"
    }
    return return_id, "Return initiated successfully."
