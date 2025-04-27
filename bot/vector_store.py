import logging
import os
from typing import List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document

from .config import settings
from .llm import embeddings

logger = logging.getLogger(__name__)

# Define the path for the FAISS index based on settings
FAISS_INDEX_PATH = settings.faiss_index_path
# Global variable to hold the loaded index
vector_store: Optional[FAISS] = None


def initialize_in_memory_vector_store():
    """Loads the FAISS index from the pre-built local path into memory."""
    global vector_store
    if vector_store:
        logger.info("In-memory vector store already initialized.")
        return

    if os.path.exists(FAISS_INDEX_PATH) and os.path.isdir(FAISS_INDEX_PATH):
        try:
            # FAISS uses pickle for serialization, which can be insecure if the source is untrusted.
            # Set allow_dangerous_deserialization=True as we trust the index file created by our script.
            vector_store = FAISS.load_local(
                folder_path=FAISS_INDEX_PATH,
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"Successfully loaded FAISS index from {FAISS_INDEX_PATH}")
        except Exception as e:
            logger.error(f"Failed to load FAISS index from {FAISS_INDEX_PATH}: {e}", exc_info=True)
            vector_store = None
    else:
        logger.warning(f"FAISS index directory not found at {FAISS_INDEX_PATH}. RAG features will be unavailable. "
                       f"Run 'python scripts/load_kb.py' to build the index.")
        vector_store = None


def get_retriever(search_type="similarity", k=3) -> VectorStoreRetriever | None:
    """Gets a retriever from the loaded in-memory vector store."""
    if not vector_store:
        logger.warning("In-memory vector store not initialized or failed to load. Cannot create retriever.")
        return None
    try:
        return vector_store.as_retriever(search_type=search_type, search_kwargs={'k': k})
    except Exception as e:
        logger.error(f"Failed to create retriever from vector store: {e}", exc_info=True)
        return None


def load_and_split_documents(directory_path: str) -> List[Document]:
    """Loads documents from a directory and splits them into chunks."""
    loader = DirectoryLoader(
        directory_path,
        glob="**/*.md",
        loader_cls=UnstructuredMarkdownLoader,
        show_progress=True,
        use_multithreading=True
    )
    try:
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} documents from {directory_path}")
        if not documents:
            return []
    except Exception as e:
        logger.error(f"Failed to load documents from {directory_path}: {e}", exc_info=True)
        return []

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    split_docs = text_splitter.split_documents(documents)
    logger.info(f"Split into {len(split_docs)} chunks")
    return split_docs


def create_and_save_vector_store(documents: List[Document]):
    """Creates a FAISS index from documents and saves it locally."""
    if not documents:
        logger.warning("No documents provided to create vector store.")
        return False
    try:
        logger.info(f"Creating FAISS index from {len(documents)} document chunks...")
        # Create the FAISS index from the document chunks and embeddings
        db = FAISS.from_documents(documents, embeddings)

        # Ensure the target directory exists
        os.makedirs(FAISS_INDEX_PATH, exist_ok=True)

        # Save the index locally
        db.save_local(FAISS_INDEX_PATH)
        logger.info(f"Successfully created and saved FAISS index to {FAISS_INDEX_PATH}")
        return True
    except Exception as exception:
        logger.error(f"Failed to create or save FAISS index: {exception}", exc_info=True)
        return False