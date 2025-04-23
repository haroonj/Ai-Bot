import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.vector_store import load_and_split_documents, add_documents_to_vector_store

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

KB_DOCS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kb_docs")

if __name__ == "__main__":
    logger.info(f"Starting knowledge base loading process from: {KB_DOCS_PATH}")

    if not os.path.isdir(KB_DOCS_PATH):
        logger.error(f"Knowledge base directory not found: {KB_DOCS_PATH}")
        sys.exit(1)

    logger.info("Loading and splitting documents...")
    split_docs = load_and_split_documents(KB_DOCS_PATH)

    if not split_docs:
        logger.warning("No documents were loaded or split. Aborting.")
        sys.exit(1)

    logger.info(f"Attempting to add {len(split_docs)} document chunks to the vector store...")
    success = add_documents_to_vector_store(split_docs)

    if success:
        logger.info("Knowledge base loading process completed successfully.")
    else:
        logger.error("Knowledge base loading process failed.")
        sys.exit(1)

    sys.exit(0)
