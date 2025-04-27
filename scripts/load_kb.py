import logging
import os
import sys
import shutil # Import shutil for removing the old index

# Add project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.vector_store import load_and_split_documents, create_and_save_vector_store, FAISS_INDEX_PATH
from bot.config import settings # Import settings to ensure config is loaded

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Path to the knowledge base documents
KB_DOCS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kb_docs")

if __name__ == "__main__":
    logger.info(f"Starting FAISS index creation process using documents from: {KB_DOCS_PATH}")
    logger.info(f"Index will be saved to: {FAISS_INDEX_PATH}")

    # Check if the knowledge base directory exists
    if not os.path.isdir(KB_DOCS_PATH):
        logger.error(f"Knowledge base document directory not found: {KB_DOCS_PATH}")
        sys.exit(1)

    # Optional: Remove existing index before creating a new one
    if os.path.exists(FAISS_INDEX_PATH):
        logger.warning(f"Removing existing index directory at: {FAISS_INDEX_PATH}")
        try:
            shutil.rmtree(FAISS_INDEX_PATH)
        except OSError as e:
            logger.error(f"Error removing existing index directory {FAISS_INDEX_PATH}: {e}", exc_info=True)
            sys.exit(1)

    # Load and split the documents
    logger.info("Loading and splitting documents...")
    split_docs = load_and_split_documents(KB_DOCS_PATH)

    if not split_docs:
        logger.warning("No documents were loaded or split from the directory. Aborting index creation.")
        # Depending on requirements, you might want to exit(0) if an empty index is acceptable
        sys.exit(1) # Exit if no documents found is an error state

    # Create and save the FAISS index
    logger.info(f"Attempting to create and save FAISS index from {len(split_docs)} document chunks...")
    success = create_and_save_vector_store(split_docs)

    if success:
        logger.info("FAISS index creation process completed successfully.")
        sys.exit(0) # Explicitly exit with success code
    else:
        logger.error("FAISS index creation process failed.")
        sys.exit(1) # Explicitly exit with failure code