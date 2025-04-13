import base64
import logging
import numpy as np
import pandas as pd
import dask.dataframe as dd
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import torch
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from starlette.concurrency import run_in_threadpool
import json
import os
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.uploaded_file import UploadedFile
from app.utils.helpers import (
    load_dataframe,
    make_column_names_unique,
    infer_data_types,
    fig_to_base64,
    generate_outlier_plot,
    perform_customer_segmentation,
    generate_cluster_plot,
    generate_ai_summary
)
import aiofiles
from typing import Any

router = APIRouter()
logger = logging.getLogger(__name__)
analysis_router = APIRouter()

try:
    # Load the model and tokenizer
    model_name = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Apply dynamic quantization to improve CPU inference speed
    model_quantized = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    # Create the pipeline with the quantized model
    summarizer_pipeline = pipeline("summarization", model=model_quantized, tokenizer=tokenizer, device=-1)
    logger.info("✅ Quantized T5-small summarizer pipeline loaded successfully for CPU inference.")
except Exception as e:
    logger.error(f"🚨 Failed to load quantized summarizer: {e}")
    # Fallback to non-quantized model if quantization fails
    try:
        summarizer_pipeline = pipeline("summarization", model="t5-small", device=-1)
        logger.info("✅ T5-small summarizer pipeline loaded successfully (without quantization).")
    except Exception as e2:
        logger.error(f"🚨 Failed to load T5-small summarizer: {e2}")
        summarizer_pipeline = None

ALLOWED_MIME_TYPES = {
    ".csv": "text/csv",
    ".xls": "application/vnd.ms-excel",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".json": "application/json",
    ".parquet": "application/x-parquet",
}

# --- New Helper Functions ---
def make_column_names_unique(df: pd.DataFrame) -> pd.DataFrame:
    seen = {}
    new_cols = []
    for col in df.columns:
        if col in seen:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            new_cols.append(col)
    df.columns = new_cols
    return df

def infer_data_types(df: pd.DataFrame):
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime = df.select_dtypes(include=["datetime", "datetime64[ns]"]).columns.tolist()
    return numeric, categorical, datetime
# --- End of New Helper Functions ---

def detect_file_type(filepath: str) -> str:
    _, ext = os.path.splitext(filepath)
    ext = ext.lower()
    if ext not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=400, detail=f"🚨 Unsupported file format: {ext}")
    logger.info(f"📌 Detected MIME type: {ALLOWED_MIME_TYPES[ext]} for file: {filepath}")
    return ALLOWED_MIME_TYPES[ext]

def load_dataframe(filepath: str) -> pd.DataFrame:
    file_size = os.path.getsize(filepath)
    if file_size > 100 * 1024 * 1024:
        try:
            df = dd.read_csv(filepath).compute()
            logger.info(f"✅ Loaded file with Dask | Shape: {df.shape}")
            return df
        except Exception as e:
            logger.warning(f"Dask loading failed: {e}, falling back to Pandas.")
    mime_type = detect_file_type(filepath)
    encodings = ["utf-8", "ISO-8859-1", "windows-1252"]
    for encoding in encodings:
        try:
            if mime_type == "text/csv":
                df = pd.read_csv(filepath, encoding=encoding, engine="python", on_bad_lines="skip", low_memory=True)
            elif mime_type in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
                df = pd.read_excel(filepath, engine="openpyxl")
            elif mime_type == "application/json":
                df = pd.read_json(filepath)
            elif mime_type == "application/x-parquet":
                df = pd.read_parquet(filepath)
            else:
                raise ValueError("🚨 Unsupported file format")
            logger.info(f"✅ Loaded file with encoding: {encoding} | Shape: {df.shape}")
            return df
        except UnicodeDecodeError:
            logger.warning(f"⚠️ Encoding {encoding} failed. Trying next...")
    raise HTTPException(status_code=400, detail="🚨 Unable to decode file with any known encoding.")
def generate_ai_summary(df: pd.DataFrame, business_insights: str) -> str:
    if summarizer_pipeline is None:
        return "AI summarization is currently unavailable."

    sample_data = df.head(10).to_csv(index=False)

    context = (
        f"You are a business intelligence analyst. Analyze the following data and provide meaningful business insights.\n\n"
        f"Sample data (CSV):\n{sample_data}\n\n"
        f"Business context: {business_insights}\n\n"
        f"Focus on trends, anomalies, customer behavior patterns, economic signals, and operational red flags."
    )

    prompt = "summarize: " + context

    try:
        summary = summarizer_pipeline(prompt, max_length=220, min_length=80, do_sample=False)[0]["summary_text"]
    except Exception as e:
        logger.error(f"🚨 AI Summarization Error: {str(e)}")
        summary = "AI summarization failed. Please review the raw insights."

    return summary.strip()

def perform_customer_segmentation(df):
    if "customer_lifetime_value" in df.columns and "credit_score" in df.columns:
        features = df[['customer_lifetime_value', 'credit_score']].dropna()
        if features.empty:
            return df
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(features_scaled)
        df.loc[features.index, 'customer_segment'] = clusters
    return df

def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

def generate_cluster_plot(df):
    if "customer_segment" in df.columns:
        fig = px.scatter(
            df,
            x="customer_lifetime_value",
            y="credit_score",
            color="customer_segment",
            title="Customer Segmentation",
            labels={"customer_lifetime_value": "Customer Lifetime Value", "credit_score": "Credit Score"}
        )
        return fig.to_json()
    return None

def generate_outlier_plot(df, column):
    outliers = detect_outliers_iqr(df, column)
    fig = px.box(df, y=column, points="all", title=f"Outlier Detection for {column}")
    fig.add_scatter(y=outliers[column], mode="markers", marker=dict(color="red"), name="Outliers")
    return fig.to_json()

# New helper function to convert Plotly figures to base64 images using Kaleido
def fig_to_base64(fig):
    try:
        img_bytes = fig.to_image(format="png")
        encoded = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
    except Exception as e:
        logger.error(f"🚨 Plot export error: {e}")
        return None

def perform_analysis(filepath: str) -> dict[str, Any]:
    df = load_dataframe(filepath)
    if df.empty:
        logger.error("🚨 Dataset is empty! Cannot perform analysis.")
        raise HTTPException(status_code=400, detail="Uploaded file contains no data.")
    
    df = make_column_names_unique(df)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    numeric_cols, categorical_cols, datetime_cols = infer_data_types(df)
    
    # 🔍 Business Insights
    business_insights = ""
    if "gdp_growth" in df.columns:
        business_insights += f"Average GDP growth is {df['gdp_growth'].mean():.2f}%. "
    if "customer_lifetime_value" in df.columns:
        business_insights += f"Average customer lifetime value is ${df['customer_lifetime_value'].mean():,.2f}. "
    if "credit_score" in df.columns:
        low_credit_count = df[df["credit_score"] < 600].shape[0]
        business_insights += f"{low_credit_count} customers have low credit scores. "
    if "transaction_amount" in df.columns:
        df["fraud_score"] = np.abs(zscore(df["transaction_amount"]))
        fraud_cases = df[df["fraud_score"] > 3].shape[0]
        business_insights += f"{fraud_cases} potential fraud cases detected. "
    if "revenue" in df.columns:
        business_insights += f"Average revenue is ${df['revenue'].mean():,.2f}. "
    if "unemployment_rate" in df.columns:
        unemployment_trend = df["unemployment_rate"].rolling(window=3).mean()
        business_insights += f"Recent unemployment rate is {unemployment_trend.iloc[-1]:.2f}%. "
    
    if not business_insights:
        business_insights = f"Dataset contains {df.shape[0]} records across {df.shape[1]} columns."
    
    # 📊 Static Plots
    plots = {}
    if datetime_cols and numeric_cols:
        df_sorted = df.sort_values(by=datetime_cols[0])
        fig_trend = px.line(df_sorted, x=datetime_cols[0], y=numeric_cols[0], title=f"Trend: {numeric_cols[0]}")
        plots[f"trend_{numeric_cols[0]}"] = fig_trend.to_json()
    
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        fig_corr = px.imshow(corr_matrix, text_auto=True, title="Correlation Heatmap")
        plots["correlation_heatmap"] = fig_corr.to_json()
    
    if "fraud_score" in df.columns:
        fig_box = px.box(df, y="fraud_score", title="Fraud Score Distribution")
        plots["fraud_boxplot"] = fig_box.to_json()
        if "transaction_amount" in df.columns:
            plots["transaction_outliers"] = generate_outlier_plot(df, "transaction_amount")
    
    if "customer_lifetime_value" in df.columns and "credit_score" in df.columns:
        df = perform_customer_segmentation(df)
        cluster_plot = generate_cluster_plot(df)
        if cluster_plot:
            plots["customer_segmentation"] = cluster_plot
    
    # 📈 Dynamic Visuals
    numeric_cols, categorical_cols, datetime_cols = infer_data_types(df)
    dynamic_plots = {}
    
    if datetime_cols and numeric_cols:
        fig_line = px.line(df.sort_values(by=datetime_cols[0]), x=datetime_cols[0], y=numeric_cols[0],
                           title=f"Line Chart: {numeric_cols[0]} over {datetime_cols[0]}")
        dynamic_plots["line_chart"] = fig_to_base64(fig_line)
    
    if numeric_cols:
        fig_hist = px.histogram(df, x=numeric_cols[0], title=f"Histogram: {numeric_cols[0]}")
        dynamic_plots["histogram"] = fig_to_base64(fig_hist)
    
    if categorical_cols and numeric_cols:
        group = df.groupby(categorical_cols[0])[numeric_cols[0]].mean().reset_index()
        group = make_column_names_unique(group)
        fig_bar = px.bar(group, x=group.columns[0], y=group.columns[1],
                         title=f"Bar Chart: {group.columns[1]} by {group.columns[0]}")
        dynamic_plots["bar_chart"] = fig_to_base64(fig_bar)
    
    if categorical_cols:
        value_counts = df[categorical_cols[0]].value_counts().reset_index()
        value_counts.columns = [categorical_cols[0], 'count']
        fig_pie = px.pie(value_counts, names=categorical_cols[0], values='count',
                         title=f"Pie Chart: Distribution of {categorical_cols[0]}")
        dynamic_plots["pie_chart"] = fig_to_base64(fig_pie)
    
    if numeric_cols:
        fig_box = px.box(df, y=numeric_cols[0], title=f"Box Plot: {numeric_cols[0]}")
        dynamic_plots["box_plot"] = fig_to_base64(fig_box)
    
    if len(numeric_cols) >= 2:
        fig_scatter = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1],
                                 title=f"Scatter Plot: {numeric_cols[1]} vs {numeric_cols[0]}")
        dynamic_plots["scatter_plot"] = fig_to_base64(fig_scatter)
    
    ai_summary = generate_ai_summary(df, business_insights)
    
    return {
        "columns": df.columns.tolist(),
        "shape": df.shape,
        "summary": ai_summary,
        "insights": business_insights,
        "plots": plots,
        "dynamic_visuals": dynamic_plots
    }

from fastapi import BackgroundTasks

@analysis_router.post("/analysis/{file_id}")
async def analyze_file(file_id: int, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    file_info = db.query(UploadedFile).filter(UploadedFile.id == file_id).first()

    if not file_info or not os.path.exists(file_info.filepath):
        logger.error(f"🚨 File with ID {file_id} not found or missing on disk!")
        raise HTTPException(status_code=404, detail="File not found.")

    logger.info(f"📂 Starting background analysis for file: {file_info.filename}")
    status_path = f"storage/statuses/{file_id}.status"
    result_path = f"storage/results/{file_id}.json"
    
    os.makedirs(os.path.dirname(status_path), exist_ok=True)
    
    # Write initial status to the file
    async with aiofiles.open(status_path, mode='w') as status_file:
        await status_file.write("processing")
    
    # Run background analysis task
    background_tasks.add_task(run_analysis_and_store, file_info.filepath, result_path, status_path)
    
    return {
        "status": "processing",
        "file_id": file_id,
        "filename": file_info.filename,
        "message": "Analysis started. Please wait while the system processes the file."
    }



async def run_analysis_and_store(input_file_path: str, result_path: str, status_path: str):
        try:
            logger.info(f"⚙️ Analysis started for: {input_file_path}")

            # 💾 Ensure directories exist
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            os.makedirs(os.path.dirname(status_path), exist_ok=True)

            # 🧠 Run analysis logic in thread pool to prevent event loop blocking
            from app.routes.analysis import perform_analysis
            result_data = await run_in_threadpool(perform_analysis, input_file_path)

            # ✅ Store results to JSON file asynchronously
            async with aiofiles.open(result_path, mode='w') as result_file:
                await result_file.write(json.dumps(result_data, indent=2))

            # ✅ Update status to completed
            async with aiofiles.open(status_path, mode='w') as status_file:
                await status_file.write("completed")

            logger.info(f"✅ Analysis completed and stored at: {result_path}")

        except Exception as e:
            logger.error(f"❌ Analysis failed during background task: {str(e)}")
            # Write failure status
            async with aiofiles.open(status_path, mode='w') as status_file:
                await status_file.write("failed")

# ✅ GET analysis result
@analysis_router.get("/analysis/result/{file_id}")
async def get_analysis_result(file_id: int):
    result_path = f"storage/results/{file_id}.json"
    status_path = f"storage/statuses/{file_id}.status"

    # Check the status of the analysis
    if not os.path.exists(status_path):
        logger.error(f"❌ Status file not found for file_id: {file_id}")
        raise HTTPException(status_code=404, detail="Analysis status not found.")

    async with aiofiles.open(status_path, mode="r") as status_file:
        status_content = await status_file.read()

    if status_content.strip() == "processing":
        return {
            "status": "processing",
            "message": "Analysis is still processing. Please wait."
        }

    if not os.path.exists(result_path):
        logger.error(f"❌ Analysis result not found for file_id: {file_id}")
        raise HTTPException(status_code=404, detail="Analysis result not found.")

    async with aiofiles.open(result_path, mode='r') as result_file:
        content = await result_file.read()
        return json.loads(content)
