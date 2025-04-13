import logging
import os
import pandas as pd
import cchardet
import asyncio
import httpx
import aiofiles
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from sqlalchemy.orm import Session
from werkzeug.utils import secure_filename
from app.database import get_db
from app.models.uploaded_file import UploadedFile

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)
MAX_FILE_SIZE_MB = 5  # Max file size in MB
VALID_TASKS = ["prediction", "analysis"]
SERVICE_HOST = os.getenv("SERVICE_HOST", "http://localhost:8000")  # Configurable microservice host

# FastAPI Router
router = APIRouter()

# 🔍 Read file into Pandas DataFrame with encoding detection
# 🔍 Read file into Pandas DataFrame with encoding detection
def read_file_into_dataframe(file_path: str, filename: str):
    try:
        if filename.endswith(".csv"):
            # ✅ Detect encoding
            with open(file_path, "rb") as f:
                raw_data = f.read()
                encoding_info = cchardet.detect(raw_data)
                detected_encoding = encoding_info.get("encoding", "utf-8")
                confidence = encoding_info.get("confidence", 0)

            # ✅ Use ISO-8859-1 for low confidence
            if confidence < 0.7:
                detected_encoding = "ISO-8859-1"

            logger.info(f"📌 Using encoding: {detected_encoding} (confidence: {confidence})")

            # ✅ Read CSV with detected encoding
            return pd.read_csv(file_path, encoding=detected_encoding)

        elif filename.endswith((".xls", ".xlsx")):
            # ✅ Read Excel normally (encoding is auto-handled)
            return pd.read_excel(file_path)

        else:
            raise ValueError(f"Unsupported file format: {filename}")

    except Exception as e:
        logger.error(f"❌ Error reading file {filename}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")
# 💾 Save file info to database
def save_file_to_db(filename: str, filepath: str, task: str, db: Session):
    db_file = UploadedFile(filename=filename, filepath=filepath, task=task)
    db.add(db_file)
    db.commit()
    db.refresh(db_file)
    logger.info(f"File info saved to database: {filename}")
    return db_file

# 🔄 Forward request to a microservice with retry logic
async def forward_request_to_service(endpoint: str, filepath: str, max_retries: int = 3, backoff_factor: float = 2.0):
    url = f"{SERVICE_HOST}{endpoint}"
    headers = {"Content-Type": "multipart/form-data"}

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                async with aiofiles.open(filepath, "rb") as f:
                    file_content = await f.read()
                    files = {"file": (os.path.basename(filepath), file_content, "application/octet-stream")}

                    response = await client.post(url, headers=headers, files=files)
                    response.raise_for_status()

                    return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error (Attempt {attempt + 1}/{max_retries}): {e}")
            if e.response.status_code >= 500:  # Retry only for server errors
                await asyncio.sleep(backoff_factor * (2 ** attempt))
                continue
            raise HTTPException(status_code=e.response.status_code, detail=f"Service error: {e}")

        except (httpx.ConnectError, httpx.ReadTimeout) as e:
            logger.warning(f"Network issue (Attempt {attempt + 1}/{max_retries}): {e}")
            await asyncio.sleep(backoff_factor * (2 ** attempt))
            continue

        except Exception as e:
            logger.error(f"Unexpected error (Attempt {attempt + 1}/{max_retries}): {e}")
            raise HTTPException(status_code=500, detail=f"Error forwarding request: {str(e)}")

    logger.error(f"Failed to connect to {url} after {max_retries} attempts.")
    raise HTTPException(status_code=503, detail=f"Service unavailable after {max_retries} retries.")

# 🚀 FastAPI POST endpoint to process files efficiently
@router.post("/process")
async def process_file(
    file: UploadFile = File(...),
    task: str = Form(...),
    db: Session = Depends(get_db)
):
    logger.info(f"Received file: {file.filename} for task: {task}")

    # Validate task type
    if task not in VALID_TASKS:
        raise HTTPException(status_code=400, detail=f"Invalid task: {task}")

    # Validate file extension
    allowed_extensions = {".csv", ".xls", ".xlsx"}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Unsupported file format. Only CSV/Excel allowed.")

    # Check file size before saving
    content_length = file.size or 0
    if content_length > MAX_FILE_SIZE_MB * 1024 * 1024:
        logger.error(f"File size exceeds {MAX_FILE_SIZE_MB}MB limit.")
        raise HTTPException(status_code=400, detail=f"File size exceeds {MAX_FILE_SIZE_MB}MB limit.")

    try:
        # Secure the filename
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_DIR, filename)

        # Save the file in chunks
        total_size = 0
        CHUNK_SIZE = 512 * 1024  # 512 KB chunks

        async with aiofiles.open(filepath, "wb") as out_file:
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > MAX_FILE_SIZE_MB * 1024 * 1024:
                    logger.error(f"File too large: {total_size} bytes")
                    raise HTTPException(status_code=400, detail=f"File size exceeds {MAX_FILE_SIZE_MB}MB limit.")
                await out_file.write(chunk)

        logger.info(f"File saved successfully: {filepath}")

        # Read file into DataFrame
        df = read_file_into_dataframe(filepath, file.filename)

        # Save file details in database
        db_file = save_file_to_db(filename, filepath, task, db)

        # Forward the request to the appropriate service
        if task == "analysis":
            analysis_response = await forward_request_to_service(f"/analysis/{db_file.id}", filepath)
            return {"message": "Analysis task processed", "file_id": db_file.id, "analysis_result": analysis_response}

        elif task == "prediction":
            prediction_response = await forward_request_to_service(f"/prediction/{db_file.id}", filepath)
            return {"message": "Prediction task processed", "file_id": db_file.id, "prediction_result": prediction_response}

    except Exception as e:
        logger.exception(f"Error processing file {file.filename}")  # Logs full stack trace
        raise HTTPException(status_code=500, detail="Internal server error")
