import os
import sys
from unittest.mock import MagicMock

import httpx
import pytest

from tests.conftest import mock_httpx_client_instance, mock_retriever_instance, MockGetRetriever


@pytest.fixture
def tools_module():
    import importlib
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    import bot.tools
    importlib.reload(bot.tools)
    return bot.tools


@pytest.fixture
def tools_config():
    from bot.config import settings
    return settings


def test_get_order_status_success(tools_module):
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {"order_id": "ORD123", "status": "Shipped"}
    mock_response.raise_for_status.return_value = None
    mock_httpx_client_instance.get.return_value = mock_response
    result = tools_module.get_order_status.invoke({"order_id": "ORD123"})
    mock_httpx_client_instance.get.assert_called_once_with("/orders/ORD123/status")
    mock_response.raise_for_status.assert_called_once()
    assert result == {"order_id": "ORD123", "status": "Shipped"}


def test_get_order_status_not_found(tools_module):
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 404
    mock_response.text = '{"detail":"Not Found"}'
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError("NF", request=MagicMock(),
                                                                       response=mock_response)
    mock_httpx_client_instance.get.return_value = mock_response
    result = tools_module.get_order_status.invoke({"order_id": "NOTFOUND"})
    mock_httpx_client_instance.get.assert_called_once_with("/orders/NOTFOUND/status")
    mock_response.raise_for_status.assert_called_once()
    assert result == {"error": "Order ID 'NOTFOUND' not found."}


def test_get_order_status_api_error(tools_module):
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 500
    mock_response.text = 'Internal Server Error'
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError("SE", request=MagicMock(),
                                                                       response=mock_response)
    mock_httpx_client_instance.get.return_value = mock_response
    result = tools_module.get_order_status.invoke({"order_id": "ORD500"})
    mock_httpx_client_instance.get.assert_called_once_with("/orders/ORD500/status")
    assert result == {"error": "API error fetching status for order ORD500: 500"}


def test_get_order_status_request_error(tools_module):
    mock_httpx_client_instance.get.side_effect = httpx.RequestError("Connection failed", request=MagicMock())
    result = tools_module.get_order_status.invoke({"order_id": "CONNERR"})
    mock_httpx_client_instance.get.assert_called_once_with("/orders/CONNERR/status")
    assert result == {"error": "Could not connect to the order system to fetch status for CONNERR."}


def test_get_tracking_info_success(tools_module):
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {"order_id": "ORD123", "tracking_number": "T123", "carrier": "C",
                                       "status": "In Transit"}
    mock_response.raise_for_status.return_value = None
    mock_httpx_client_instance.get.return_value = mock_response
    result = tools_module.get_tracking_info.invoke({"order_id": "ORD123"})
    mock_httpx_client_instance.get.assert_called_once_with("/orders/ORD123/tracking")
    assert result["tracking_number"] == "T123"


def test_get_tracking_info_not_found(tools_module):
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 404
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError("NF", request=MagicMock(),
                                                                       response=mock_response)
    mock_httpx_client_instance.get.return_value = mock_response
    result = tools_module.get_tracking_info.invoke({"order_id": "NF"})
    mock_httpx_client_instance.get.assert_called_once_with("/orders/NF/tracking")
    assert result == {"error": "Order ID 'NF' not found."}


def test_get_order_details_success_delivered(tools_module):
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {"order_id": "ORD789", "items": [{"sku": "S1"}], "delivered": True}
    mock_response.raise_for_status.return_value = None
    mock_httpx_client_instance.get.return_value = mock_response
    result = tools_module.get_order_details.invoke({"order_id": "ORD789"})
    mock_httpx_client_instance.get.assert_called_once_with("/orders/ORD789/details")
    assert result["delivered"] is True
    assert "error" not in result


def test_get_order_details_not_delivered(tools_module):
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_details = {"order_id": "ORD123", "items": [], "delivered": False, "status": "Shipped"}
    mock_response.json.return_value = mock_details
    mock_response.raise_for_status.return_value = None
    mock_httpx_client_instance.get.return_value = mock_response
    result = tools_module.get_order_details.invoke({"order_id": "ORD123"})
    mock_httpx_client_instance.get.assert_called_once_with("/orders/ORD123/details")
    assert result["error"] == f"Order ORD123 is not yet delivered or not eligible for return based on details."
    assert result["details"] == mock_details


def test_get_order_details_not_found(tools_module):
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 404
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError("NF", request=MagicMock(),
                                                                       response=mock_response)
    mock_httpx_client_instance.get.return_value = mock_response
    result = tools_module.get_order_details.invoke({"order_id": "NF"})
    mock_httpx_client_instance.get.assert_called_once_with("/orders/NF/details")
    assert result == {"error": "Order ID 'NF' not found."}


def test_initiate_return_success(tools_module):
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {"return_id": "RET1", "status": "OK", "message": "Done"}
    mock_response.raise_for_status.return_value = None
    mock_httpx_client_instance.post.return_value = mock_response
    payload_to_invoke = {"order_id": "ORD789", "sku": "SKU1", "reason": "R"}
    result = tools_module.initiate_return_request.invoke(payload_to_invoke)
    mock_httpx_client_instance.post.assert_called_once_with("/returns", json=payload_to_invoke)
    assert result["return_id"] == "RET1"


def test_initiate_return_api_error_detail(tools_module):
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 400
    mock_response.json.return_value = {"detail": "Item not eligible for return"}
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError("BR", request=MagicMock(),
                                                                       response=mock_response)
    mock_httpx_client_instance.post.return_value = mock_response
    payload_to_invoke = {"order_id": "ORDX", "sku": "SKUX"}
    result = tools_module.initiate_return_request.invoke(payload_to_invoke)

    expected_post_payload = {"order_id": "ORDX", "sku": "SKUX", "reason": None}
    mock_httpx_client_instance.post.assert_called_once_with("/returns", json=expected_post_payload)
    assert result == {"error": "Item not eligible for return"}


def test_initiate_return_api_error_no_detail(tools_module):
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 500
    mock_response.json.side_effect = ValueError("Not JSON")
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError("SE", request=MagicMock(),
                                                                       response=mock_response)
    mock_httpx_client_instance.post.return_value = mock_response
    payload_to_invoke = {"order_id": "ORDY", "sku": "SKUY"}
    result = tools_module.initiate_return_request.invoke(payload_to_invoke)

    expected_post_payload = {"order_id": "ORDY", "sku": "SKUY", "reason": None}
    mock_httpx_client_instance.post.assert_called_once_with("/returns", json=expected_post_payload)
    assert result == {"error": "API error initiating return: Status 500"}


def test_knowledge_base_lookup_success(tools_module):
    MockGetRetriever.return_value = mock_retriever_instance
    mock_doc1 = MagicMock()
    mock_doc1.page_content = "Content 1"
    mock_doc2 = MagicMock()
    mock_doc2.page_content = "Content 2"
    mock_retriever_instance.invoke.return_value = [mock_doc1, mock_doc2]
    query = "How long is shipping?"
    result = tools_module.knowledge_base_lookup.invoke({"query": query})
    MockGetRetriever.assert_called_once()
    mock_retriever_instance.invoke.assert_called_once_with(query)
    assert result == "Content 1\n\nContent 2"


def test_knowledge_base_lookup_no_results(tools_module):
    MockGetRetriever.return_value = mock_retriever_instance
    mock_retriever_instance.invoke.return_value = []
    query = "Unknown topic?"
    result = tools_module.knowledge_base_lookup.invoke({"query": query})
    assert result == "I couldn't find specific information about that in our knowledge base."
    MockGetRetriever.assert_called_once()
    mock_retriever_instance.invoke.assert_called_once_with(query)


def test_knowledge_base_lookup_retriever_unavailable(tools_module):
    MockGetRetriever.return_value = None
    query = "Anything?"
    result = tools_module.knowledge_base_lookup.invoke({"query": query})
    assert result == "I apologize, my knowledge base is currently unavailable."
    MockGetRetriever.assert_called_once()
    mock_retriever_instance.invoke.assert_not_called()


def test_knowledge_base_lookup_retrieval_error(tools_module):
    MockGetRetriever.return_value = mock_retriever_instance
    mock_retriever_instance.invoke.side_effect = Exception("Vector DB connection lost")
    query = "Help!"
    result = tools_module.knowledge_base_lookup.invoke({"query": query})
    assert result == "I encountered an error while searching the knowledge base."
    MockGetRetriever.assert_called_once()
    mock_retriever_instance.invoke.assert_called_once_with(query)
