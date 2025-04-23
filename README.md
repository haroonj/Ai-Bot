# 💡 AI E-Commerce Support Bot

A backend AI-powered chatbot for handling customer support inquiries on e-commerce platforms using **LangGraph**, **LangChain**, and **PGVector**.

---

## 🚀 Setup & Installation

### **Prerequisites**

- Python **3.12+**
- Docker & Docker Compose
- Git

### **1. Clone the Repository**
```bash
git clone <repository-url>
cd Ai-Bot
```

### **2. Set Up a Virtual Environment**
```bash
python3 -m venv .venv
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4. Configure Environment Variables**
Create a `.env` file in the root directory:

```dotenv
# OpenAI
OPENAI_API_KEY="sk-..."

# PGVector Database
POSTGRES_HOST="localhost"
POSTGRES_PORT="5333"
POSTGRES_DB="ragbot"
POSTGRES_USER="raguser"
POSTGRES_PASSWORD="ragpass"
VECTOR_STORE_COLLECTION_NAME="ecommerce_kb" # Optional

# Mock API
MOCK_API_BASE_URL="http://localhost:8001"

# Optional LangSmith Tracing
# LANGCHAIN_TRACING_V2="true"
# LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
# LANGCHAIN_API_KEY="..."
# LANGCHAIN_PROJECT="..."
```

Use `.env.example` if provided.

### **5. Start the Vector Database**
Make sure Docker is running, then:
```bash
docker-compose up -d
```
This will spin up PostgreSQL with PGVector on port `5333`.

### **6. Load Knowledge Base**
```bash
python scripts/load_kb.py
```
This script reads files from `kb_docs/`, splits them, generates OpenAI embeddings, and stores them in PGVector.

---

## 🤖 Running the Bot & Mock API

### **1. Mock E-commerce API**
```bash
uvicorn mock_api.main:app --reload --port 8001
```

### **2. Main Chatbot API**
```bash
uvicorn main:app --reload --port 8000
```

Bot is now available at:  
📉 `http://localhost:8000`

---

## 📡 API Usage

### **POST /chat**
Send messages to the bot.

#### **Request Example**
```json
{
  "query": "What is the status of order ORD123?",
  "conversation_id": "optional-uuid"
}
```

#### **Response Example**
```json
{
  "reply": "The status for order ORD123 is: Shipped.",
  "conversation_id": "auto-generated-uuid"
}
```

### **cURL Example**
```bash
curl -X POST http://localhost:8000/chat \
-H "Content-Type: application/json" \
-d '{"query": "How do I return an item?"}'
```

---

## 🧪 Testing

This project uses **pytest**.

### **Run all tests**
```bash
pytest tests/
```

### **Notes**
- Tests are located in `tests/`
- Mocks for OpenAI, PGVector, and external APIs are centralized in `tests/conftest.py`
- Tests are fast and do **not require real credentials or databases**

---

## 🔁 CI/CD with GitHub Actions

The workflow in `.github/workflows/run_tests.yml` automatically:

- Checks out your code
- Sets up Python and dependencies
- Runs all tests
- Generates test reports (JUnit XML + HTML)
- Uploads the HTML report as an artifact
- Shows results in the GitHub UI via [dorny/test-reporter](https://github.com/dorny/test-reporter)

---