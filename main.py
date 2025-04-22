# main.py

import logging
from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
import uuid # For generating unique conversation IDs if needed

# Import necessary components from the bot module
from bot.config import settings # To potentially access settings if needed
from bot.graph import get_runnable
from bot.state import GraphState # For typing hints
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- FastAPI App Setup ---
app = FastAPI(title="E-commerce Support Bot API", version="1.0.0")

# Add CORS middleware if you plan to call this from a browser frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for simplicity in hackathon
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Variables / Resources ---
try:
    langgraph_runnable = get_runnable()
    logger.info("LangGraph runnable loaded successfully.")
except Exception as e:
    logger.critical(f"Failed to initialize LangGraph runnable: {e}", exc_info=True)
    langgraph_runnable = None # Prevent API from starting if graph fails

# --- Request/Response Models ---
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender ('user' or 'assistant')")
    content: str = Field(..., description="Content of the message")

class ChatRequest(BaseModel):
    query: str = Field(..., description="The user's latest message/query.")
    conversation_id: Optional[str] = Field(None, description="Optional ID to maintain conversation state (if using checkpointer).")
    # Include past messages if maintaining state client-side (for stateless backend)
    # history: Optional[List[ChatMessage]] = Field(None, description="Past conversation messages")

class ChatResponse(BaseModel):
    reply: str = Field(..., description="The bot's response message.")
    conversation_id: str = Field(..., description="ID for the current conversation thread.")
    # Potentially return updated history or other state info if needed
    # history: List[ChatMessage]

# --- API Endpoints ---

@app.on_event("startup")
async def startup_event():
    if langgraph_runnable is None:
         # This won't actually stop FastAPI startup easily, but logs critical error.
         # Proper handling might involve a readiness probe in a k8s setup.
        logger.critical("CRITICAL: LangGraph runnable failed to load. API may not function.")
    logger.info("FastAPI application startup complete.")
    logger.info(f"Mock API URL configured: {settings.mock_api_base_url}")
    logger.info(f"Using LLM: {settings.llm_model_name}, Embeddings: {settings.embedding_model_name}")

@app.get("/", summary="Health Check")
async def root():
    """Simple health check endpoint."""
    return {"status": "ok", "message": "E-commerce Support Bot API is running."}

@app.post("/chat", response_model=ChatResponse, summary="Process User Query")
async def chat_endpoint(request: ChatRequest = Body(...)):
    """
    Receives a user query, processes it through the LangGraph agent,
    and returns the bot's reply.
    """
    if langgraph_runnable is None:
        logger.error("Attempted to call /chat endpoint but LangGraph is not available.")
        raise HTTPException(status_code=503, detail="Service Unavailable: Bot engine not initialized.")

    user_query = request.query
    conversation_id = request.conversation_id or str(uuid.uuid4()) # Generate new ID if none provided

    logger.info(f"Received chat request for conversation_id: {conversation_id}")
    logger.info(f"User query: {user_query}")

    # --- Prepare Input for LangGraph ---
    # For a stateless approach per request: Start with just the human message.
    # The graph's internal state doesn't persist between calls unless a checkpointer is active.
    initial_state: GraphState = {
        "messages": [HumanMessage(content=user_query)],
        # Initialize other state keys to None or default values
        "intent": None,
        "order_id": None,
        "item_sku_to_return": None,
        "return_reason": None,
        "needs_clarification": False,
        "clarification_question": None,
        "available_return_items": None,
        "rag_context": None,
        "api_response": None,
        "tool_error": None,
        "next_node": None,
    }

    # --- Invoke LangGraph ---
    # If using a checkpointer (like MemorySaver), you'd pass thread_id here:
    # config = {"configurable": {"thread_id": conversation_id}}
    # For stateless:
    config = {}

    try:
        # The `invoke` method runs the graph from entry to end for a given input.
        final_state = langgraph_runnable.invoke(initial_state, config=config)

        # Extract the last message added by the graph (should be the AI's response)
        if final_state and final_state.get("messages"):
            last_message = final_state["messages"][-1]
            if isinstance(last_message, AIMessage):
                bot_reply = last_message.content
                logger.info(f"Bot reply generated: {bot_reply}")
            else:
                 # This might happen if the graph ends unexpectedly or state is malformed
                 logger.error(f"Graph execution finished, but last message is not AIMessage: {last_message}")
                 bot_reply = "I'm sorry, I encountered an issue generating a response."
        else:
            logger.error("Graph execution finished with no final state or messages.")
            bot_reply = "I'm sorry, something went wrong while processing your request."

    except Exception as e:
        logger.exception(f"Error during LangGraph invocation for conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error: Failed to process the request.")

    # --- Return Response ---
    return ChatResponse(reply=bot_reply, conversation_id=conversation_id)


# --- Main Execution ---
if __name__ == "__main__":
    # Run using Uvicorn
    # Example: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    # The '--reload' flag is useful for development. Remove for production.
    logger.info("Starting E-commerce Support Bot API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)