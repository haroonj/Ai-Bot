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
    # Reload to ensure mocks from conftest are applied if module was imported before
    importlib.reload(bot.vector_store)
    return bot.vector_store

@pytest.fixture
def bot_settings():
    """Provides settings."""
    from bot.config import settings
    return settings

def test_vector_store_initialization(vector_store_module):
    """Test if PGVector instance in the module is the session mock."""
    # The instance should be the one from conftest due to session patching
    assert vector_store_module.vector_store is mock_pgvector_instance

def test_get_retriever_success(vector_store_module):
    """Test successful retriever retrieval."""
    # Use the mock instance defined in conftest
    mock_retriever = MagicMock(name="specific_mock_retriever_get")
    mock_pgvector_instance.as_retriever.return_value = mock_retriever

    retriever = vector_store_module.get_retriever(search_type="similarity", k=5)

    assert retriever is mock_retriever
    # Assert that the mock vector store instance's method was called
    mock_pgvector_instance.as_retriever.assert_called_once_with(
        search_type="similarity", search_kwargs={'k': 5}
    )

def test_get_retriever_no_store(vector_store_module, mocker):
    """Test retriever retrieval when store is None."""
    # Apply patch inside the test using mocker
    mocker.patch('bot.vector_store.vector_store', None)
    # Patch the logger within the module
    logger_warning = mocker.patch('bot.vector_store.logger.warning') # More specific patch target

    retriever = vector_store_module.get_retriever()

    assert retriever is None
    logger_warning.assert_called_with("Vector store not initialized. Cannot create retriever.")

def test_load_and_split_documents(vector_store_module, mocker):
    """Test document loading and splitting logic."""
    # Use local mocks for loader/splitter as their usage is being tested here
    MockLocalDirectoryLoader = MagicMock(name="LocalDirLoader")
    mock_local_loader_instance = MockLocalDirectoryLoader.return_value
    MockLocalSplitter = MagicMock(name="LocalSplitter")
    mock_local_splitter_instance = MockLocalSplitter.return_value

    # Patch locally for this test's scope
    # Patch the classes *as used within* bot.vector_store.py
    mocker.patch('bot.vector_store.DirectoryLoader', MockLocalDirectoryLoader)
    mocker.patch('bot.vector_store.RecursiveCharacterTextSplitter', MockLocalSplitter)

    # Configure mocks
    mock_doc1 = MagicMock()
    mock_doc2 = MagicMock()
    mock_split_docs = [MagicMock(), MagicMock(), MagicMock()]
    mock_local_loader_instance.load.return_value = [mock_doc1, mock_doc2]
    mock_local_splitter_instance.split_documents.return_value = mock_split_docs

    test_path_str = "/fake/path/for/load_split" # Pass a string as the function expects
    result = vector_store_module.load_and_split_documents(test_path_str)

    # --- Assertions ---
    # Assert DirectoryLoader was called with the path string
    MockLocalDirectoryLoader.assert_called_once_with(
        test_path_str, # Check the path argument
        glob="**/*.md",
        loader_cls=ANY, # Or import UnstructuredMarkdownLoader and check specifically
        show_progress=True,
        use_multithreading=True
    )
    # Assert load was called on the instance
    mock_local_loader_instance.load.assert_called_once()

    # Assert Splitter was called
    MockLocalSplitter.assert_called_once_with(chunk_size=1000, chunk_overlap=150)
    # Assert split_documents was called with the docs returned by loader.load
    mock_local_splitter_instance.split_documents.assert_called_once_with([mock_doc1, mock_doc2])

    # Assert the result is correct
    assert result == mock_split_docs


def test_add_documents_to_vector_store_success(vector_store_module):
    """Test adding documents successfully."""
    docs = [MagicMock(), MagicMock()]
    # Ensure the module uses the globally mocked vector_store instance
    assert vector_store_module.vector_store is mock_pgvector_instance

    result = vector_store_module.add_documents_to_vector_store(docs)

    assert result is True
    # Assert add_documents was called on the mock instance
    mock_pgvector_instance.add_documents.assert_called_once_with(docs)

def test_add_documents_no_store(vector_store_module, mocker):
    """Test adding documents when store is None."""
    # Apply patch inside the test using mocker
    mocker.patch('bot.vector_store.vector_store', None)
    logger_error = mocker.patch('bot.vector_store.logger.error') # More specific patch target
    docs = [MagicMock()]

    result = vector_store_module.add_documents_to_vector_store(docs)

    assert result is False
    # FIX: Remove exc_info=True from assertion as it's not used in the actual code
    logger_error.assert_called_with("Vector store not initialized. Cannot add documents.")
    # Check the globally mocked instance wasn't called
    mock_pgvector_instance.add_documents.assert_not_called()


def test_add_documents_to_vector_store_failure_exception(vector_store_module):
    """Test adding documents when the DB call fails."""
    # Ensure the module uses the globally mocked vector_store instance
    assert vector_store_module.vector_store is mock_pgvector_instance
    docs = [MagicMock()]
    # Configure the mock instance to raise an error
    mock_pgvector_instance.add_documents.side_effect = Exception("DB error")

    result = vector_store_module.add_documents_to_vector_store(docs)

    assert result is False
    # Assert add_documents was called
    mock_pgvector_instance.add_documents.assert_called_once_with(docs)