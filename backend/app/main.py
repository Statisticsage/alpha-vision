from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.routes.auth_routes import auth_router
from app.routes.file_processing_routes import router as file_processing_routes
from app.routes.analysis import analysis_router
from app.routes.prediction_routes import prediction_router
from app.database import engine, Base
from openai import OpenAI  # ✅ Updated for OpenAI v1+
import logging
import os
from dotenv import load_dotenv
from typing import List, Dict, Any

# 📦 Load environment variables
load_dotenv()

# ✅ Initialize OpenAI client (v1+ syntax)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 🚀 Initialize FastAPI
app = FastAPI(
    title="Alpha Vision API",
    description="Handles data analysis & predictions",
    version="1.0.0"
)

# 🌍 CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📁 Static files
app.mount("/prediction_results", StaticFiles(directory="prediction_results"), name="prediction_results")
app.mount("/analysis_results", StaticFiles(directory="analysis_results"), name="analysis_results")
# 🧱 Create DB tables
Base.metadata.create_all(bind=engine)

# 🔗 Include routers
app.include_router(auth_router)
app.include_router(file_processing_routes)
app.include_router(analysis_router)
app.include_router(prediction_router)

# 🧠 Chatbot Model
class ChatRequest(BaseModel):
    message: str
    data: List[Dict[str, Any]]  # Expecting a list of dashboard records

# 💬 AI Chat Endpoint
@app.post("/api/chat")
async def chat_endpoint(chat_request: ChatRequest):
    logging.info("📨 AI Chat request: %s", chat_request.message)

    try:
        if not isinstance(chat_request.data, list) or not chat_request.data:
            raise HTTPException(status_code=400, detail="⚠️ Dashboard data must be a non-empty list.")

        user_question = chat_request.message
        dashboard_data = chat_request.data[:5]  # Just a sample for context

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful AI assistant that analyzes dashboard data and user questions. "
                    "Give insightful, easy-to-understand answers based on trends, patterns, or anomalies in the data."
                )
            },
            {
                "role": "user",
                "content": f"Here is a sample of the dashboard data:\n{dashboard_data}\n\nQuestion: {user_question}"
            }
        ]

        # ✅ Modern OpenAI SDK usage (v1+)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )

        answer = response.choices[0].message.content  # ✅ Correct access pattern
        return {"response": answer}

    except Exception as e:
        logging.error("❌ AI Chatbot error: %s", str(e))
        raise HTTPException(status_code=500, detail="AI failed to process the request")

# 🌐 HTTP Error Handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

# 🏠 Base Endpoint
@app.get("/")
def read_root():
    logging.info("🏠 Home endpoint hit")
    return {"message": "Welcome to the Analysis and Prediction world!"}
