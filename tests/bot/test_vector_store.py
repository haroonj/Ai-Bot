# tests/bot/test_vector_store.py
import pytest
from unittest.mock import MagicMock, patch, ANY
import sys
import os
from pathlib import Path

# Dependencies (PGVector, embeddings, Path, loaders) are mocked globally by conftest.py
from tests.conftest import mock_pgvector_instance, mock_retriever_instance # Import instances for assertions

@pytest.fixture
def vector_store_module():
    """Provides the vector_store module."""
    import importlib
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    if project_root not in sys.path:
         sys.path.insert(0, project_root)
    import bot.vector_store
    importlib.reload(bot.vector_store)
    return bot.vector_store

@pytest.fixture
def bot_settings():
    """Provides settings."""
    from bot.config import settings
    return settings

def test_vector_store_initialization(vector_store_module):
    """Test if PGVector instance in the module is the session mock."""
    assert vector_store_module.vector_store is mock_pgvector_instance

def test_get_retriever_success(vector_store_module):
    """Test successful retriever retrieval."""
    mock_retriever = MagicMock(name="specific_mock_retriever")
    mock_pgvector_instance.as_retriever.return_value = mock_retriever
    retriever = vector_store_module.get_retriever(search_type="similarity", k=5)
    assert retriever is mock_retriever
    mock_pgvector_instance.as_retriever.assert_called_once_with(
        search_type="similarity", search_kwargs={'k': 5}
    )

@patch('bot.vector_store.vector_store', None)
def test_get_retriever_no_store(vector_store_module, mocker):
    """Test retriever retrieval when store is None."""
    logger_warning = mocker.patch('logging.Logger.warning')
    retriever = vector_store_module.get_retriever()
    assert retriever is None
    logger_warning.assert_called_with("Vector store not initialized. Cannot create retriever.")

def test_load_and_split_documents(vector_store_module, mocker):
    """Test document loading and splitting logic."""
    # Use local mocks for loader/splitter/path as global ones might be complex
    MockLocalDirectoryLoader = MagicMock(name="LocalDirLoader")
    mock_local_loader_instance = MockLocalDirectoryLoader.return_value
    MockLocalSplitter = MagicMock(name="LocalSplitter")
    mock_local_splitter_instance = MockLocalSplitter.return_value
    MockLocalPathInstance = MagicMock(spec=Path)
    MockLocalPath = MagicMock(name="LocalPath", return_value=MockLocalPathInstance)

    # Patch locally for this test's scope
    mocker.patch('bot.vector_store.DirectoryLoader', MockLocalDirectoryLoader)
    mocker.patch('bot.vector_store.RecursiveCharacterTextSplitter', MockLocalSplitter)
    mocker.patch('langchain_community.document_loaders.directory.Path', MockLocalPath, create=True)

    mock_doc1 = MagicMock()
    mock_doc2 = MagicMock()
    mock_split_docs = [MagicMock(), MagicMock(), MagicMock()]
    mock_local_loader_instance.load.return_value = [mock_doc1, mock_doc2]
    mock_local_splitter_instance.split_documents.return_value = mock_split_docs
    MockLocalPathInstance.exists.return_value = True

    test_path = "/fake/path/for/load_split"
    result = vector_store_module.load_and_split_documents(test_path)

    MockLocalPath.assert_called_with(test_path)
    MockLocalPathInstance.exists.assert_called_once()
    MockLocalDirectoryLoader.assert_called_once()
    mock_local_loader_instance.load.assert_called_once()
    MockLocalSplitter.assert_called_once_with(chunk_size=1000, chunk_overlap=150)
    mock_local_splitter_instance.split_documents.assert_called_once_with([mock_doc1, mock_doc2])
    assert result == mock_split_docs

def test_add_documents_to_vector_store_success(vector_store_module):
    """Test adding documents successfully."""
    docs = [MagicMock(), MagicMock()]
    assert vector_store_module.vector_store is mock_pgvector_instance
    result = vector_store_module.add_documents_to_vector_store(docs)
    assert result is True
    mock_pgvector_instance.add_documents.assert_called_once_with(docs)

@patch('bot.vector_store.vector_store', None)
def test_add_documents_no_store(vector_store_module, mocker):
    """Test adding documents when store is None."""
    logger_error = mocker.patch('logging.Logger.error')
    docs = [MagicMock()]
    result = vector_store_module.add_documents_to_vector_store(docs)
    assert result is False
    logger_error.assert_called_with("Vector store not initialized. Cannot add documents.")
    mock_pgvector_instance.add_documents.assert_not_called()

def test_add_documents_to_vector_store_failure_exception(vector_store_module):
    """Test adding documents when the DB call fails."""
    assert vector_store_module.vector_store is mock_pgvector_instance
    docs = [MagicMock()]
    mock_pgvector_instance.add_documents.side_effect = Exception("DB error")
    result = vector_store_module.add_documents_to_vector_store(docs)
    assert result is False
    mock_pgvector_instance.add_documents.assert_called_once_with(docs)