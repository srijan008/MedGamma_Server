# MedGamma AI Chatbot 🏥🤖 - Backend

MedGamma's backend is a robust FastAPI application serving an advanced AI assistant capable of RAG (Document Analysis), Web Search, and critical Emergency Response via Twilio.

## ✨ Backend Features
- **FastAPI Core**: High-performance async API.
- **LangChain & Cohere**: Powered by Cohere's Command R+ models and Embeddings.
- **ChromaDB**: Vector storage for PDF knowledge retrieval.
- **Tools System**: Integrated `WebSearchTool` (DuckDuckGo), `EmergencyCallTool`, and `EmergencySmsTool` (Twilio).
- **Disk Offloading**: Specific optimization for running large Hugging Face models (like MedGemma) on limited hardware.

## 🛠️ Tech Stack
- **Python 3.10+**
- **FastAPI / Uvicorn**
- **LangChain / LangChain-Cohere**
- **Twilio** (Voice & SMS)
- **Hugging Face Transformers** & **Accelerate**
- **Prisma** (Database ORM)

---

## 🚀 Setup & Installation

### 1. Prerequisites
- Python 3.10 or higher installed.
- Twilio Account (SID, Auth Token, Verified Numbers).
- Cohere API Key.
- Hugging Face Token (for gated models like MedGemma).

### 2. Installation
```bash
cd backend
python -m venv venv

# Activate venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install Dependencies
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the `backend/` directory with the following keys:

```ini
# --- Database ---
DATABASE_URL="postgresql://user:password@host:port/db?sslmode=require" # Or sqlite

# --- AI Services ---
cohere_api_key="your_cohere_key"
GOOGLE_API_KEY="your_google_key" # If used

# --- Vector Store (Chroma) ---
# Leave these blank if using local in-memory Chroma
CHROMA_API_KEY="" 
CHROMA_TENANT=""
CHROMA_DATABASE=""

# --- Twilio (Emergency System) ---
TWILIO_ACCOUNT_SID="AC..."
TWILIO_AUTH_TOKEN="your_token"
TWILIO_FROM_NUMBER="+1234567890"
TWILIO_TO_NUMBER="+0987654321"

# --- Hugging Face ---
hugging_face="hf_your_token" # Required for gated models
```

### 4. Running Locally
```bash
uvicorn main:app --reload
```
API will be available at `http://localhost:8000`.
Docs at `http://localhost:8000/docs`.

---

## ☁️ Deployment (Render.com)

This backend is correctly configured for deployment on Render.

**Settings:**
- **Build Command:** `pip install -r requirements.txt` (Ensure `twilio`, `transformers`, etc. are in requirements.txt)
- **Start Command:** `uvicorn main:app --host 0.0.0.0 --port 10000`

**Environment Variables:**
Ensure all variables from your `.env` (especially `TWILIO_...` keys) are added to the Environment Variables section in your Render dashboard.

### ⚠️ Note on Large Models
If attempting to run the **MedGemma** model (4B params) on Render's free tier, it will likely OOM (Out of Memory). The `medgemma_test.py` script includes optimization for disk offloading, but standard deployment containers are ephemeral and resource-constrained.
