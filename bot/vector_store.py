# bot/vector_store.py

import logging
from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader # Or TextLoader

from .config import settings
from .llm import embeddings

logger = logging.getLogger(__name__)

# --- Constants ---
CONNECTION_STRING = settings.pgvector_connection_string
COLLECTION_NAME = settings.vector_store_collection_name

# --- Vector Store Setup ---
try:
    vector_store = PGVector(
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        embedding_function=embeddings,
    )
    logger.info(f"Connected to PGVector collection '{COLLECTION_NAME}'")
except Exception as e:
    logger.error(f"Failed to connect to PGVector: {e}", exc_info=True)
    # Depending on the application, you might want to raise the exception
    # or handle it gracefully (e.g., run without RAG)
    vector_store = None # Indicate failure

def get_retriever(search_type="similarity", k=3) -> VectorStoreRetriever | None:
    """Gets a retriever instance from the configured vector store."""
    if not vector_store:
        logger.warning("Vector store not initialized. Cannot create retriever.")
        return None
    return vector_store.as_retriever(search_type=search_type, search_kwargs={'k': k})

# --- KB Loading Functionality (can be called from scripts/load_kb.py) ---
def load_and_split_documents(directory_path: str) -> list:
    """Loads and splits documents from a directory."""
    # Using DirectoryLoader with UnstructuredMarkdownLoader for .md files
    # Adjust loader_cls if using different file types (e.g., TextLoader for .txt)
    loader = DirectoryLoader(
        directory_path,
        glob="**/*.md", # Load only markdown files
        loader_cls=UnstructuredMarkdownLoader,
        show_progress=True,
        use_multithreading=True
    )
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} documents from {directory_path}")

    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    split_docs = text_splitter.split_documents(documents)
    logger.info(f"Split into {len(split_docs)} chunks")
    return split_docs

def add_documents_to_vector_store(documents: list):
    """Adds documents to the PGVector store."""
    if not vector_store:
        logger.error("Vector store not initialized. Cannot add documents.")
        return False
    try:
        vector_store.add_documents(documents)
        logger.info(f"Successfully added {len(documents)} chunks to collection '{COLLECTION_NAME}'")
        return True
    except Exception as e:
        logger.error(f"Failed to add documents to vector store: {e}", exc_info=True)
        return False