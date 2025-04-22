# tests/bot/test_vector_store.py
import pytest
from unittest.mock import MagicMock, patch, ANY
import sys
import os
from pathlib import Path # Import Path for mocking

# --- Mock External Dependencies EARLY ---
# Mock the PGVector class itself
mock_pgvector_instance = MagicMock(spec=True) # Use spec=True for better mocking
MockPGVector = MagicMock(return_value=mock_pgvector_instance)

# Mock Embeddings class
mock_embeddings_instance = MagicMock()
MockOpenAIEmbeddings = MagicMock(return_value=mock_embeddings_instance)

# Mock document loader and text splitter classes
MockDirectoryLoader = MagicMock()
mock_loader_instance = MockDirectoryLoader.return_value # Get the instance mock
MockRecursiveCharacterTextSplitter = MagicMock()
mock_splitter_instance = MockRecursiveCharacterTextSplitter.return_value # Get the instance mock

# Mock Path object methods used by DirectoryLoader
MockPathInstance = MagicMock(spec=Path)
MockPath = MagicMock(return_value=MockPathInstance)

# Apply patches using a fixture targeting specific import locations
@pytest.fixture(scope='module', autouse=True)
def patch_vector_store_dependencies(module_mocker):
    """Apply patches for vector store dependencies."""
    # Patch the class where it's imported and used for instantiation
    module_mocker.patch('bot.vector_store.PGVector', MockPGVector)
    module_mocker.patch('bot.vector_store.embeddings', mock_embeddings_instance) # Patch the instance directly
    # Patch loader/splitter where they are used in the functions
    module_mocker.patch('bot.vector_store.DirectoryLoader', MockDirectoryLoader)
    module_mocker.patch('bot.vector_store.RecursiveCharacterTextSplitter', MockRecursiveCharacterTextSplitter)
    # Patch pathlib.Path used within DirectoryLoader or related functions if needed
    module_mocker.patch('langchain_community.document_loaders.directory.Path', MockPath)
    module_mocker.patch('bot.vector_store.Path', MockPath) # If used directly in vector_store

    # Yield control to the tests
    yield

    # Optional: Clean up patches if needed, though module scope usually handles it
    module_mocker.stopall()


# Import the module *after* the fixture setup might run
# Using fixtures to provide the module helps ensure order
@pytest.fixture
def vector_store_module():
    """Provides the vector_store module, potentially reloaded with patches."""
    import importlib
    import bot.vector_store
    # Reload to ensure patches applied if already imported
    importlib.reload(bot.vector_store)
    return bot.vector_store

@pytest.fixture
def bot_settings():
    """Provides settings."""
    # Ensure dummy vars are set via .env or os.environ before this runs
    from bot.config import settings
    return settings

# Fixture to reset mocks before each test function
@pytest.fixture(autouse=True)
def reset_vector_store_mocks():
    MockPGVector.reset_mock()
    mock_pgvector_instance.reset_mock()
    mock_pgvector_instance.as_retriever.reset_mock()
    mock_pgvector_instance.add_documents.reset_mock()
    MockOpenAIEmbeddings.reset_mock()
    mock_embeddings_instance.reset_mock()
    MockDirectoryLoader.reset_mock()
    mock_loader_instance.reset_mock()
    mock_loader_instance.load.reset_mock()
    MockRecursiveCharacterTextSplitter.reset_mock()
    mock_splitter_instance.reset_mock()
    mock_splitter_instance.split_documents.reset_mock()
    MockPath.reset_mock()
    MockPathInstance.reset_mock()
    # Reset side effects
    mock_pgvector_instance.add_documents.side_effect = None
    # Default mock retriever
    mock_pgvector_instance.as_retriever.return_value = MagicMock(name="mock_retriever")
    # Default mock Path behavior
    MockPathInstance.exists.return_value = True # Prevent FileNotFoundError


def test_vector_store_initialization(vector_store_module, bot_settings):
    """Test if PGVector was instantiated (implicitly by module import)."""
    # Check that the *instance* within the reloaded module is our mock
    assert vector_store_module.vector_store is mock_pgvector_instance
    # Check constructor was called at least once during import/reload
    MockPGVector.assert_called()
    # More specific check if needed (might be fragile due to reload)
    # MockPGVector.assert_called_with(
    #     collection_name=bot_settings.vector_store_collection_name,
    #     connection_string=bot_settings.pgvector_connection_string,
    #     embedding_function=mock_embeddings_instance,
    # )

def test_get_retriever_success(vector_store_module):
    """Test successful retriever retrieval."""
    mock_retriever = MagicMock(name="specific_mock_retriever")
    mock_pgvector_instance.as_retriever.return_value = mock_retriever

    retriever = vector_store_module.get_retriever(search_type="similarity", k=5)

    # Now that vector_store should be the mock instance, this should work
    assert retriever is mock_retriever
    mock_pgvector_instance.as_retriever.assert_called_once_with(
        search_type="similarity", search_kwargs={'k': 5}
    )

# Correctly use @patch without injecting an argument
@patch('bot.vector_store.vector_store', None)
def test_get_retriever_no_store(vector_store_module, mocker):
    """Test retriever retrieval when store is None."""
    # NOTE: The patch decorator replaces bot.vector_store.vector_store with None *during this test*
    logger_warning = mocker.patch('logging.Logger.warning')
    retriever = vector_store_module.get_retriever()
    assert retriever is None
    logger_warning.assert_called_with("Vector store not initialized. Cannot create retriever.")


def test_load_and_split_documents(vector_store_module):
    """Test document loading and splitting logic."""
    mock_doc1 = MagicMock()
    mock_doc2 = MagicMock()
    mock_split_docs = [MagicMock(), MagicMock(), MagicMock()]

    # Configure the mocked instances
    mock_loader_instance.load.return_value = [mock_doc1, mock_doc2]
    mock_splitter_instance.split_documents.return_value = mock_split_docs
    MockPathInstance.exists.return_value = True # Ensure path exists check passes

    test_path = "/fake/path/for/test" # Use a more specific path
    result = vector_store_module.load_and_split_documents(test_path)

    # Assert mocks were called correctly
    MockPath.assert_called_with(test_path)
    MockPathInstance.exists.assert_called_once()
    MockDirectoryLoader.assert_called_once()
    mock_loader_instance.load.assert_called_once()
    MockRecursiveCharacterTextSplitter.assert_called_once_with(chunk_size=1000, chunk_overlap=150)
    mock_splitter_instance.split_documents.assert_called_once_with([mock_doc1, mock_doc2])
    assert result == mock_split_docs


def test_add_documents_to_vector_store_success(vector_store_module):
    """Test adding documents successfully."""
    docs = [MagicMock(), MagicMock()]
    # Ensure the module uses the mocked instance
    assert vector_store_module.vector_store is mock_pgvector_instance

    result = vector_store_module.add_documents_to_vector_store(docs)

    assert result is True
    mock_pgvector_instance.add_documents.assert_called_once_with(docs)


# Correctly use @patch without injecting an argument
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