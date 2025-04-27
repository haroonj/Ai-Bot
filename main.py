import json
import logging
from typing import List

import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, messages_to_dict, messages_from_dict

from bot.config import settings
from bot.graph import get_runnable
from bot.state import GraphState
from bot.vector_store import initialize_in_memory_vector_store # Import the initializer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="E-commerce Support Bot", version="1.0.0")

templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LangGraph Runnable
langgraph_runnable = None
try:
    langgraph_runnable = get_runnable()
    logger.info("LangGraph runnable loaded successfully.")
except Exception as e:
    logger.critical(f"Failed to initialize LangGraph runnable: {e}", exc_info=True)
    # Allow startup to continue, but log critical failure


def format_messages_for_template(messages: List[BaseMessage]) -> List[dict]:
    formatted = []
    for msg in messages:
        if isinstance(msg, (HumanMessage, AIMessage)):
            simple_dict = {
                "type": msg.type,
                "content": msg.content
            }
            formatted.append(simple_dict)
    return formatted


@app.on_event("startup")
async def startup_event():
    # Load the FAISS index into memory on startup
    logger.info("Application startup: Initializing vector store...")
    initialize_in_memory_vector_store()

    if langgraph_runnable is None:
        logger.critical("CRITICAL: LangGraph runnable failed to load during startup. API may not function correctly.")

    logger.info("FastAPI application startup complete.")
    logger.info(f"Using LLM: {settings.llm_model_name}, Embeddings: {settings.embedding_model_name}")
    logger.info(f"FAISS index path: {settings.faiss_index_path}")


@app.get("/", response_class=HTMLResponse, summary="Chat Interface")
async def get_chat_interface(request: Request):
    logger.info("Serving initial chat interface.")
    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "messages": [],
            "history_json": "[]",
            "error": None
        }
    )


@app.post("/chat", response_class=HTMLResponse, summary="Process User Query (Form)")
async def handle_chat_form(
        request: Request,
        query: str = Form(...),
        history_json: str = Form("[]")
):
    if langgraph_runnable is None:
        logger.error("Attempted to process chat but LangGraph runnable is not available.")
        # Format existing history (if any) and show error
        current_messages: List[BaseMessage] = []
        try:
            history_dicts = json.loads(history_json)
            current_messages = messages_from_dict(history_dicts)
            current_messages.append(HumanMessage(content=query)) # Add user query for context
        except Exception:
            logger.warning("Could not decode history_json on error path.")
            current_messages = [HumanMessage(content=query)]

        final_messages_for_template = format_messages_for_template(current_messages)
        new_history_json = json.dumps(messages_to_dict(current_messages))

        return templates.TemplateResponse(
            "chat.html",
            {
                "request": request,
                "messages": final_messages_for_template,
                "history_json": new_history_json,
                "error": "Bot engine is not available or failed to load. Please contact support."
            }
        )

    logger.info(f"Received form query: {query}")
    error_message = None
    current_messages: List[BaseMessage] = []

    try:
        history_dicts = json.loads(history_json)
        current_messages = messages_from_dict(history_dicts)
        logger.debug(f"Reconstructed history with {len(current_messages)} messages.")
    except json.JSONDecodeError:
        logger.warning("Could not decode history_json, starting fresh conversation.")
        current_messages = [] # Start fresh if history is bad
    except Exception as exception:
        logger.error(f"Error reconstructing message history: {exception}", exc_info=True)
        error_message = "Error loading previous conversation state. Starting fresh."
        current_messages = [] # Start fresh on other errors

    # Append the new user message
    current_messages.append(HumanMessage(content=query))

    # Prepare state for the graph
    initial_state: GraphState = {
        "messages": current_messages,
        "intent": None, "order_id": None, "item_sku_to_return": None,
        "return_reason": None, "needs_clarification": False,
        "clarification_question": None, "available_return_items": None,
        "rag_context": None, "api_response": None, "tool_error": None,
        "next_node": None,
    }
    config = {} # Add any necessary config for LangGraph run

    try:
        # Invoke the LangGraph runnable
        final_state = langgraph_runnable.invoke(initial_state, config=config)

        if final_state and final_state.get("messages"):
            final_messages_lc: List[BaseMessage] = final_state["messages"]
            final_messages_for_template = format_messages_for_template(final_messages_lc)
            new_history_json = json.dumps(messages_to_dict(final_messages_lc))

            # Check for errors reported by the graph itself
            if final_state.get("tool_error"):
                # Append tool error to the displayed error messages if needed, or just log it
                error_message = f"Tool Error: {final_state['tool_error']}" # Or handle differently
                logger.warning(f"Graph execution finished with tool_error: {final_state['tool_error']}")
            elif not isinstance(final_messages_lc[-1], AIMessage):
                error_message = "Bot failed to generate a final response."
                logger.error(f"Graph execution finished, but last message is not AIMessage: {final_messages_lc[-1]}")

        else:
            # Handle cases where the graph invocation fails to return a valid state
            error_message = "Something went wrong processing your request. Graph returned invalid state."
            logger.error("Graph execution finished with no final state or messages.")
            # Fallback to showing just the user's message
            final_messages_lc = current_messages
            final_messages_for_template = format_messages_for_template(final_messages_lc)
            new_history_json = json.dumps(messages_to_dict(final_messages_lc))

    except Exception as exception:
        # Catch errors during the graph invocation itself
        logger.exception(f"Error during LangGraph invocation: {exception}")
        error_message = "Internal server error processing your request. Please try again later."
        # Fallback to showing just the user's message
        final_messages_lc = current_messages
        final_messages_for_template = format_messages_for_template(final_messages_lc)
        new_history_json = json.dumps(messages_to_dict(final_messages_lc))

    # Render the template with the final state
    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "messages": final_messages_for_template,
            "history_json": new_history_json,
            "error": error_message
        }
    )


if __name__ == "__main__":
    logger.info("Starting E-commerce Support Bot API server...")
    # Note: Uvicorn reload might not trigger the startup event correctly every time
    # For production, run without --reload
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) # Added reload=True for dev