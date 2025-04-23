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

try:
    langgraph_runnable = get_runnable()
    logger.info("LangGraph runnable loaded successfully.")
except Exception as e:
    logger.critical(f"Failed to initialize LangGraph runnable: {e}", exc_info=True)
    langgraph_runnable = None


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
    if langgraph_runnable is None:
        logger.critical("CRITICAL: LangGraph runnable failed to load. API may not function.")
    logger.info("FastAPI application startup complete.")
    logger.info(f"Mock API URL configured: {settings.mock_api_base_url}")
    logger.info(f"Using LLM: {settings.llm_model_name}, Embeddings: {settings.embedding_model_name}")


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
        logger.error("Attempted to process chat but LangGraph is not available.")
        return templates.TemplateResponse(
            "chat.html",
            {"request": request, "messages": [], "history_json": "[]", "error": "Bot engine not initialized."}
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
    except Exception as exception:
        logger.error(f"Error reconstructing message history: {exception}", exc_info=True)
        error_message = "Error loading previous conversation state."

    current_messages.append(HumanMessage(content=query))

    final_messages_for_template = format_messages_for_template(current_messages)
    new_history_json = json.dumps(messages_to_dict(current_messages))

    if langgraph_runnable and not error_message:
        initial_state: GraphState = {
            "messages": current_messages,
            "intent": None, "order_id": None, "item_sku_to_return": None,
            "return_reason": None, "needs_clarification": False,
            "clarification_question": None, "available_return_items": None,
            "rag_context": None, "api_response": None, "tool_error": None,
            "next_node": None,
        }
        config = {}

        try:
            final_state = langgraph_runnable.invoke(initial_state, config=config)
            if final_state and final_state.get("messages"):
                final_messages_lc: List[BaseMessage] = final_state["messages"]
                final_messages_for_template = format_messages_for_template(final_messages_lc)
                new_history_json = json.dumps(messages_to_dict(final_messages_lc))
                if final_state.get("tool_error"):
                    error_message = final_state["tool_error"]
                    logger.warning(f"Graph execution finished with tool_error: {error_message}")
                elif not isinstance(final_messages_lc[-1], AIMessage):
                    error_message = "Bot failed to generate a final response."
                    logger.error(
                        f"Graph execution finished, but last message is not AIMessage: {final_messages_lc[-1]}")

            else:
                error_message = "Something went wrong processing your request."
                logger.error("Graph execution finished with no final state or messages.")
                final_messages_for_template = format_messages_for_template(current_messages)
                new_history_json = json.dumps(messages_to_dict(current_messages))

        except Exception as exception:
            logger.exception(f"Error during LangGraph invocation: {exception}")
            error_message = "Internal server error processing your request."
            final_messages_for_template = format_messages_for_template(current_messages)
            new_history_json = json.dumps(messages_to_dict(current_messages))

    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "messages": final_messages_for_template,
            "history_json": new_history_json,  # Send back the correct history JSON
            "error": error_message
        }
    )


if __name__ == "__main__":
    logger.info("Starting E-commerce Support Bot API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
