import os
import sys
from unittest.mock import MagicMock

# Removed httpx import
# import httpx
import pytest

# Import sample data mocks from conftest
from tests.conftest import (
    mock_retriever_instance,
    MockGetRetriever,
    MockGetOrder_SampleData,
    MockCreateReturn_SampleData
)


@pytest.fixture
def tools_module():
    import importlib
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    import bot.tools
    importlib.reload(bot.tools) # Reload to ensure patches are applied
    return bot.tools

# Removed tools_config fixture as it's no longer needed for base URL

# --- Test Cases ---

def test_get_order_status_success(tools_module):
    # Setup mock for sample data function
    mock_order_data = {"status": "Shipped"}
    MockGetOrder_SampleData.return_value = mock_order_data

    # Invoke the tool
    result = tools_module.get_order_status.invoke({"order_id": "123"})

    # Assertions
    MockGetOrder_SampleData.assert_called_once_with("123")
    assert result == {"order_id": "123", "status": "Shipped"}


def test_get_order_status_not_found(tools_module):
    # Setup mock: get_order returns None
    MockGetOrder_SampleData.return_value = None

    # Invoke the tool
    result = tools_module.get_order_status.invoke({"order_id": "NOTFOUND"})

    # Assertions
    MockGetOrder_SampleData.assert_called_once_with("NOTFOUND")
    assert result == {"error": "Order ID 'NOTFOUND' not found."}


def test_get_tracking_info_success(tools_module):
    # Setup mock
    mock_order_data = {
        "tracking_number": "T123",
        "carrier": "C",
        "tracking_status": "In Transit"
    }
    MockGetOrder_SampleData.return_value = mock_order_data

    # Invoke
    result = tools_module.get_tracking_info.invoke({"order_id": "123"})

    # Assertions
    MockGetOrder_SampleData.assert_called_once_with("123")
    assert result == {
        "order_id": "123",
        "tracking_number": "T123",
        "carrier": "C",
        "status": "In Transit"
    }

def test_get_tracking_info_not_available(tools_module):
    # Setup mock: Order exists but no tracking number
    mock_order_data = {
        "status": "Processing",
        "tracking_number": None # Explicitly None
    }
    MockGetOrder_SampleData.return_value = mock_order_data

    # Invoke
    result = tools_module.get_tracking_info.invoke({"order_id": "456"})

    # Assertions
    MockGetOrder_SampleData.assert_called_once_with("456")
    assert result == {"order_id": "456", "status": "Tracking not available yet"}


def test_get_tracking_info_not_found(tools_module):
    # Setup mock: Order not found
    MockGetOrder_SampleData.return_value = None

    # Invoke
    result = tools_module.get_tracking_info.invoke({"order_id": "NF"})

    # Assertions
    MockGetOrder_SampleData.assert_called_once_with("NF")
    assert result == {"error": "Order ID 'NF' not found."}


def test_get_order_details_success_delivered(tools_module):
    # Setup mock
    mock_order_data = {
        "items": [{"sku": "S1", "name": "Item 1"}],
        "status": "Delivered",
        "delivered": True
    }
    MockGetOrder_SampleData.return_value = mock_order_data

    # Invoke
    result = tools_module.get_order_details.invoke({"order_id": "789"})

    # Assertions
    MockGetOrder_SampleData.assert_called_once_with("789")
    assert result["delivered"] is True
    assert result["order_id"] == "789"
    assert result["items"] == [{"sku": "S1", "name": "Item 1"}]
    assert "error" not in result


def test_get_order_details_not_delivered(tools_module):
    # Setup mock
    mock_order_data = {
        "items": [{"sku": "S2", "name": "Item 2"}],
        "status": "Shipped",
        "delivered": False
    }
    MockGetOrder_SampleData.return_value = mock_order_data

    # Invoke
    result = tools_module.get_order_details.invoke({"order_id": "123"})
    expected_details = { # Construct the expected details part of the error response
        "order_id": "123",
        "items": [{"sku": "S2", "name": "Item 2"}],
        "status": "Shipped",
        "delivered": False
    }

    # Assertions
    MockGetOrder_SampleData.assert_called_once_with("123")
    assert result["error"] == f"Order 123 is not yet delivered or not eligible for return based on details."
    assert result["details"] == expected_details


def test_get_order_details_not_found(tools_module):
    # Setup mock
    MockGetOrder_SampleData.return_value = None

    # Invoke
    result = tools_module.get_order_details.invoke({"order_id": "NF"})

    # Assertions
    MockGetOrder_SampleData.assert_called_once_with("NF")
    assert result == {"error": "Order ID 'NF' not found."}


def test_initiate_return_success(tools_module):
    # Setup mock for create_return
    MockCreateReturn_SampleData.return_value = ("RET1", "Mock return successful.")

    # Invoke
    payload_to_invoke = {"order_id": "789", "sku": "SKU1", "reason": "R"}
    result = tools_module.initiate_return_request.invoke(payload_to_invoke)

    # Assertions
    MockCreateReturn_SampleData.assert_called_once_with("789", "SKU1", "R")
    assert result == {"return_id": "RET1", "status": "Return Initiated", "message": "Mock return successful."}


def test_initiate_return_failure(tools_module):
    # Setup mock for create_return failure
    error_message = "Item not eligible for return (mock)"
    MockCreateReturn_SampleData.return_value = (None, error_message)

    # Invoke
    payload_to_invoke = {"order_id": "ORDX", "sku": "SKUX"} # Reason is optional
    result = tools_module.initiate_return_request.invoke(payload_to_invoke)

    # Assertions
    MockCreateReturn_SampleData.assert_called_once_with("ORDX", "SKUX", None) # Check reason defaults to None
    assert result == {"error": error_message}

# --- Knowledge Base Tests (Should remain largely unchanged) ---

def test_knowledge_base_lookup_success(tools_module):
    # Setup mock retriever
    MockGetRetriever.return_value = mock_retriever_instance
    mock_doc1 = MagicMock()
    mock_doc1.page_content = "Content 1"
    mock_doc2 = MagicMock()
    mock_doc2.page_content = "Content 2"
    mock_retriever_instance.invoke.return_value = [mock_doc1, mock_doc2]

    # Invoke
    query = "How long is shipping?"
    result = tools_module.knowledge_base_lookup.invoke({"query": query})

    # Assertions
    MockGetRetriever.assert_called_once()
    mock_retriever_instance.invoke.assert_called_once_with(query)
    assert result == "Content 1\n\nContent 2"


def test_knowledge_base_lookup_no_results(tools_module):
    # Setup
    MockGetRetriever.return_value = mock_retriever_instance
    mock_retriever_instance.invoke.return_value = []
    query = "Unknown topic?"

    # Invoke
    result = tools_module.knowledge_base_lookup.invoke({"query": query})

    # Assertions
    assert result == "I couldn't find specific information about that in our knowledge base."
    MockGetRetriever.assert_called_once()
    mock_retriever_instance.invoke.assert_called_once_with(query)


def test_knowledge_base_lookup_retriever_unavailable(tools_module):
    # Setup: Mock get_retriever to return None
    MockGetRetriever.return_value = None
    query = "Anything?"

    # Invoke
    result = tools_module.knowledge_base_lookup.invoke({"query": query})

    # Assertions
    assert result == "I apologize, my knowledge base is currently unavailable."
    MockGetRetriever.assert_called_once()
    mock_retriever_instance.invoke.assert_not_called() # Ensure the retriever itself wasn't called


def test_knowledge_base_lookup_retrieval_error(tools_module):
    # Setup: Mock retriever's invoke to raise an error
    MockGetRetriever.return_value = mock_retriever_instance
    mock_retriever_instance.invoke.side_effect = Exception("Vector DB connection lost")
    query = "Help!"

    # Invoke
    result = tools_module.knowledge_base_lookup.invoke({"query": query})

    # Assertions
    assert result == "I encountered an error while searching the knowledge base."
    MockGetRetriever.assert_called_once()
    mock_retriever_instance.invoke.assert_called_once_with(query)