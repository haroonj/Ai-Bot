import os
import sys
from unittest.mock import MagicMock, ANY

import pytest

from tests.conftest import mock_pgvector_instance


@pytest.fixture
def vector_store_module():
    import importlib
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    import bot.vector_store
    importlib.reload(bot.vector_store)
    return bot.vector_store


@pytest.fixture
def bot_settings():
    from bot.config import settings
    return settings


def test_vector_store_initialization(vector_store_module):
    assert vector_store_module.vector_store is mock_pgvector_instance


def test_get_retriever_success(vector_store_module):
    mock_retriever = MagicMock(name="specific_mock_retriever_get")
    mock_pgvector_instance.as_retriever.return_value = mock_retriever

    retriever = vector_store_module.get_retriever(search_type="similarity", k=5)

    assert retriever is mock_retriever
    mock_pgvector_instance.as_retriever.assert_called_once_with(
        search_type="similarity", search_kwargs={'k': 5}
    )


def test_get_retriever_no_store(vector_store_module, mocker):
    mocker.patch('bot.vector_store.vector_store', None)
    logger_warning = mocker.patch('bot.vector_store.logger.warning')
    retriever = vector_store_module.get_retriever()
    assert retriever is None
    logger_warning.assert_called_with("Vector store not initialized. Cannot create retriever.")


def test_load_and_split_documents(vector_store_module, mocker):
    MockLocalDirectoryLoader = MagicMock(name="LocalDirLoader")
    mock_local_loader_instance = MockLocalDirectoryLoader.return_value
    MockLocalSplitter = MagicMock(name="LocalSplitter")
    mock_local_splitter_instance = MockLocalSplitter.return_value

    mocker.patch('bot.vector_store.DirectoryLoader', MockLocalDirectoryLoader)
    mocker.patch('bot.vector_store.RecursiveCharacterTextSplitter', MockLocalSplitter)

    mock_doc1 = MagicMock()
    mock_doc2 = MagicMock()
    mock_split_docs = [MagicMock(), MagicMock(), MagicMock()]
    mock_local_loader_instance.load.return_value = [mock_doc1, mock_doc2]
    mock_local_splitter_instance.split_documents.return_value = mock_split_docs

    test_path_str = "/fake/path/for/load_split"
    result = vector_store_module.load_and_split_documents(test_path_str)

    MockLocalDirectoryLoader.assert_called_once_with(
        test_path_str,
        glob="**/*.md",
        loader_cls=ANY,
        show_progress=True,
        use_multithreading=True
    )
    mock_local_loader_instance.load.assert_called_once()

    MockLocalSplitter.assert_called_once_with(chunk_size=1000, chunk_overlap=150)
    mock_local_splitter_instance.split_documents.assert_called_once_with([mock_doc1, mock_doc2])

    assert result == mock_split_docs


def test_add_documents_to_vector_store_success(vector_store_module):
    docs = [MagicMock(), MagicMock()]
    assert vector_store_module.vector_store is mock_pgvector_instance
    result = vector_store_module.add_documents_to_vector_store(docs)
    assert result is True
    mock_pgvector_instance.add_documents.assert_called_once_with(docs)


def test_add_documents_no_store(vector_store_module, mocker):
    mocker.patch('bot.vector_store.vector_store', None)
    logger_error = mocker.patch('bot.vector_store.logger.error')
    docs = [MagicMock()]
    result = vector_store_module.add_documents_to_vector_store(docs)
    assert result is False
    logger_error.assert_called_with("Vector store not initialized. Cannot add documents.")
    mock_pgvector_instance.add_documents.assert_not_called()


def test_add_documents_to_vector_store_failure_exception(vector_store_module):
    assert vector_store_module.vector_store is mock_pgvector_instance
    docs = [MagicMock()]
    mock_pgvector_instance.add_documents.side_effect = Exception("DB error")
    result = vector_store_module.add_documents_to_vector_store(docs)
    assert result is False
    mock_pgvector_instance.add_documents.assert_called_once_with(docs)
