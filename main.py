# main.py

import logging
import json
import uuid
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Body, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from bot.config import settings
from bot.graph import get_runnable
from bot.state import GraphState
# Corrected Import: Import messages_to_dict
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, messages_to_dict, messages_from_dict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- FastAPI App Setup ---
app = FastAPI(title="E-commerce Support Bot", version="1.0.0")

# Mount static files directory (optional if only using CDN)
# app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Add CORS middleware (optional for pure server-rendered, but good practice)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    langgraph_runnable = None

# --- Helper Functions ---
def format_messages_for_template(messages: List[BaseMessage]) -> List[dict]:
    """Converts BaseMessage list to simple dicts suitable for template."""
    formatted = []
    for msg in messages:
        # Filter out ToolMessages or other types if you don't want them displayed
        if isinstance(msg, (HumanMessage, AIMessage)):
            simple_dict = {
                "type": msg.type, # 'human' or 'ai'
                "content": msg.content
            }
            formatted.append(simple_dict)
    return formatted

# --- API Endpoints ---

@app.on_event("startup")
async def startup_event():
    if langgraph_runnable is None:
        logger.critical("CRITICAL: LangGraph runnable failed to load. API may not function.")
    logger.info("FastAPI application startup complete.")
    logger.info(f"Mock API URL configured: {settings.mock_api_base_url}")
    logger.info(f"Using LLM: {settings.llm_model_name}, Embeddings: {settings.embedding_model_name}")

# --- Serve the HTML Chat Interface ---
@app.get("/", response_class=HTMLResponse, summary="Chat Interface")
async def get_chat_interface(request: Request):
    """Serves the main HTML chat page."""
    logger.info("Serving initial chat interface.")
    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "messages": [], # Start with empty history
            "history_json": "[]", # Empty history JSON for the form
            "error": None
        }
    )

# --- Handle Chat Form Submission ---
@app.post("/chat", response_class=HTMLResponse, summary="Process User Query (Form)")
async def handle_chat_form(
    request: Request,
    query: str = Form(...),
    history_json: str = Form("[]") # Receive serialized history
):
    """
    Receives user query from form, processes it, re-renders the chat page.
    """
    if langgraph_runnable is None:
        logger.error("Attempted to process chat but LangGraph is not available.")
        # Re-render page with error
        return templates.TemplateResponse(
            "chat.html",
            {"request": request, "messages": [], "history_json": "[]", "error": "Bot engine not initialized."}
        )

    logger.info(f"Received form query: {query}")
    error_message = None
    current_messages: List[BaseMessage] = []

    # --- Reconstruct History ---
    try:
        history_dicts = json.loads(history_json)
        current_messages = messages_from_dict(history_dicts)
        logger.debug(f"Reconstructed history with {len(current_messages)} messages.")
    except json.JSONDecodeError:
        logger.warning("Could not decode history_json, starting fresh conversation.")
        # current_messages remains []
    except Exception as e:
        logger.error(f"Error reconstructing message history: {e}", exc_info=True)
        error_message = "Error loading previous conversation state."
        # current_messages remains []

    # Add current user query
    current_messages.append(HumanMessage(content=query))

    # Prepare initial template context (before graph run)
    final_messages_for_template = format_messages_for_template(current_messages)
    # FIX 1: Use messages_to_dict here for the state before potential graph failure
    new_history_json = json.dumps(messages_to_dict(current_messages))

    if langgraph_runnable and not error_message: # Only run graph if ready and no prior error
        initial_state: GraphState = {
            "messages": current_messages, # Pass reconstructed + new message
            "intent": None, "order_id": None, "item_sku_to_return": None,
            "return_reason": None, "needs_clarification": False,
            "clarification_question": None, "available_return_items": None,
            "rag_context": None, "api_response": None, "tool_error": None,
            "next_node": None,
        }
        config = {} # Stateless invocation

        try:
            # --- Invoke LangGraph ---
            logger.debug("Invoking LangGraph...")
            final_state = langgraph_runnable.invoke(initial_state, config=config)
            logger.debug("LangGraph invocation complete.")

            # --- Process Final State (Successful Graph Run) ---
            if final_state and final_state.get("messages"):
                final_messages_lc: List[BaseMessage] = final_state["messages"]
                final_messages_for_template = format_messages_for_template(final_messages_lc)
                # FIX 2: Use messages_to_dict for final successful state
                new_history_json = json.dumps(messages_to_dict(final_messages_lc))
                if final_state.get("tool_error"):
                    error_message = final_state["tool_error"]
                    logger.warning(f"Graph execution finished with tool_error: {error_message}")
                elif not isinstance(final_messages_lc[-1], AIMessage):
                    error_message = "Bot failed to generate a final response."
                    logger.error(f"Graph execution finished, but last message is not AIMessage: {final_messages_lc[-1]}")

            else: # Graph ran but returned empty/invalid state
                error_message = "Something went wrong processing your request."
                logger.error("Graph execution finished with no final state or messages.")
                # Keep template/history as user message only
                final_messages_for_template = format_messages_for_template(current_messages)
                # FIX 3: Use messages_to_dict for this error case
                new_history_json = json.dumps(messages_to_dict(current_messages))

        except Exception as e: # Catch errors during graph invocation
            logger.exception(f"Error during LangGraph invocation: {e}")
            error_message = "Internal server error processing your request."
            # Keep template/history as user message only
            final_messages_for_template = format_messages_for_template(current_messages)
            # FIX 4: Use messages_to_dict for this exception case
            new_history_json = json.dumps(messages_to_dict(current_messages))

    # --- Render Response Template ---
    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "messages": final_messages_for_template,
            "history_json": new_history_json, # Send back the correct history JSON
            "error": error_message
        }
    )

# --- Main Execution ---
if __name__ == "__main__":
    logger.info("Starting E-commerce Support Bot API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)