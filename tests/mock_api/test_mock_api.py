# tests/mock_api/test_mock_api.py
import pytest
from fastapi.testclient import TestClient
import sys
import os

# Ensure mock_api package is findable if running pytest from root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from mock_api.main import app as mock_app
from mock_api import sample_data

@pytest.fixture(scope="module")
def client():
    return TestClient(mock_app)

@pytest.fixture(autouse=True)
def reset_returns():
    sample_data.mock_returns.clear()
    sample_data.return_counter = 1

def test_read_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Mock E-commerce API is running"}

# --- Order Status Tests ---
def test_get_order_status_success(client):
    response = client.get("/orders/ORD123/status")
    assert response.status_code == 200
    assert response.json() == {"order_id": "ORD123", "status": "Shipped"}

def test_get_order_status_not_found(client):
    response = client.get("/orders/NOTFOUND/status")
    assert response.status_code == 404
    assert response.json() == {"detail": "Order not found"}

# --- Tracking Info Tests ---
def test_get_tracking_info_success(client):
    response = client.get("/orders/ORD123/tracking")
    assert response.status_code == 200
    assert response.json() == {
        "order_id": "ORD123",
        "tracking_number": "TRACK987",
        "carrier": "MockExpress",
        "status": "In Transit"
    }

def test_get_tracking_info_not_available(client):
    response = client.get("/orders/ORD456/tracking")
    assert response.status_code == 200
    # FIX: Assert the exact response structure from endpoint logic
    # Corrected Assertion
    assert response.json() == {"order_id": "ORD456", "tracking_number": None, "carrier": None, "status": "Tracking not available yet"}


def test_get_tracking_info_not_found(client):
    response = client.get("/orders/NOTFOUND/tracking")
    assert response.status_code == 404
    assert response.json() == {"detail": "Order not found"}

# --- Order Details Tests ---
def test_get_order_details_success(client):
    response = client.get("/orders/ORD789/details")
    assert response.status_code == 200
    data = response.json()
    assert data["order_id"] == "ORD789"
    assert data["status"] == "Delivered"
    assert data["delivered"] is True
    assert len(data["items"]) == 1
    assert data["items"][0]["sku"] == "ITEM004"

def test_get_order_details_not_found(client):
    response = client.get("/orders/NOTFOUND/details")
    assert response.status_code == 404
    assert response.json() == {"detail": "Order not found"}

# --- Return Initiation Tests ---
def test_initiate_return_success(client):
    payload = {"order_id": "ORD789", "sku": "ITEM004", "reason": "Test reason"}
    response = client.post("/returns", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "Return Initiated"
    assert data["message"] == "Return initiated successfully."
    assert data["return_id"] == "RETN0001"
    assert "RETN0001" in sample_data.mock_returns
    assert sample_data.mock_returns["RETN0001"]["reason"] == "Test reason"

def test_initiate_return_order_not_found(client):
    payload = {"order_id": "NOTFOUND", "sku": "ITEM004"}
    response = client.post("/returns", json=payload)
    assert response.status_code == 404 # sample_data returns None -> 404 in endpoint
    assert response.json() == {"detail": "Order not found"}

def test_initiate_return_item_not_found(client):
    payload = {"order_id": "ORD789", "sku": "WRONGITEM"}
    response = client.post("/returns", json=payload)
    # FIX: sample_data returns specific message -> 404 in endpoint
    # Based on the logic: 'not found' in message triggers 404
    # Corrected Status Code
    assert response.status_code == 404
    assert response.json() == {"detail": "Item SKU WRONGITEM not found in order ORD789"}

def test_initiate_return_not_delivered(client):
    payload = {"order_id": "ORD123", "sku": "ITEM001"}
    response = client.post("/returns", json=payload)
    # FIX: sample_data returns specific message -> 400
    # Corrected Status Code
    assert response.status_code == 400
    assert response.json() == {"detail": "Order not yet delivered, cannot return items"}