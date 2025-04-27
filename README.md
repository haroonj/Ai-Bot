# 💡 AI E-Commerce Support Bot

A backend AI-powered chatbot for handling customer support inquiries on e-commerce platforms using **LangGraph**, *
*LangChain**, and **PGVector**, with a simple web interface provided by **FastAPI** and **Jinja2**.

---

## 🚀 Setup & Installation

### Prerequisites

- Python **3.12+**
- Docker & Docker Compose
- Git

---

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Ai-Bot
```

---

### 2. Set Up a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Configure Environment Variables

1. Create a `.env` file in the project root.
2. Use `.env.example` (if available) as a reference.
3. Required variables:

```dotenv
# OpenAI
OPENAI_API_KEY="sk-..."

# PostgreSQL (Vector Store)
POSTGRES_HOST="localhost"
POSTGRES_PORT="5333"
POSTGRES_DB="ragbot"
POSTGRES_USER="raguser"
POSTGRES_PASSWORD="ragpass"
VECTOR_STORE_COLLECTION_NAME="ecommerce_kb"

# Mock API
MOCK_API_BASE_URL="http://localhost:8001"

# Optional LangSmith
# LANGCHAIN_TRACING_V2="true"
# LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
# LANGCHAIN_API_KEY="..."
# LANGCHAIN_PROJECT="..."
```

---

### 5. Start the Vector Database

```bash
docker compose up -d
```

This starts PostgreSQL with PGVector on port `5333`.

---

### 6. Load the Knowledge Base

```bash
python scripts/load_kb.py
```

This script:

- Reads markdown files from `kb_docs/`
- Splits them into chunks
- Generates embeddings via OpenAI
- Stores them in PGVector

---

## 🤖 Running the Bot & Mock API

Open **two terminals**:

### Terminal 1: Run the Mock E-commerce API

```bash
uvicorn mock_api.main:app --reload --port 8001
```

### Terminal 2: Run the Main Chatbot Server

```bash
uvicorn main:app --reload --port 8000
```

Access the chatbot at:  
👉 http://localhost:8000

---

## 💬 Chat Usage (POST API)

### Endpoint: `POST /chat`

```json
{
  "query": "What is the status of order 123?",
  "conversation_id": "optional-uuid-string"
}
```

### Response:

```json
{
  "reply": "The status for order 123 is: Shipped.",
  "conversation_id": "uuid"
}
```

**Curl Example:**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I return an item?"}'
```

---

## 🧪 Running Tests

This project uses `pytest`.

```bash
pytest tests/
```

- All external dependencies (OpenAI, PGVector, API) are mocked.
- Some tests use `BeautifulSoup` to parse HTML responses.
- Reports:
    - JUnit XML: `backend-report.xml`
    - HTML: `backend-report.html`

---

## 🔁 CI/CD (GitHub Actions)

Workflow: `.github/workflows/backend-ci.yml`

What it does:

- Installs Python & dependencies
- Runs tests with `pytest`
- Generates JUnit & HTML reports
- Uploads HTML as artifact
- Displays test summary via `dorny/test-reporter`
