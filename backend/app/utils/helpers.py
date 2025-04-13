import pandas as pd
import base64
import io
import plotly.io as pio
import plotly.graph_objects as go
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import logging

logger = logging.getLogger(__name__)

# -------------------- Load DataFrame --------------------
def load_dataframe(filepath: str) -> pd.DataFrame:
    try:
        if filepath.endswith(".csv"):
            return pd.read_csv(filepath)
        elif filepath.endswith(".xlsx"):
            return pd.read_excel(filepath)
        elif filepath.endswith(".parquet"):
            return pd.read_parquet(filepath)
        else:
            raise ValueError("Unsupported file format.")
    except Exception as e:
        logger.error(f"❌ Failed to load dataframe: {e}")
        return pd.DataFrame()

# -------------------- Unique Column Names --------------------
def make_column_names_unique(df: pd.DataFrame) -> pd.DataFrame:
    seen = set()
    new_columns = []
    for col in df.columns:
        orig_col = col
        i = 1
        while col in seen:
            col = f"{orig_col}_{i}"
            i += 1
        seen.add(col)
        new_columns.append(col)
    df.columns = new_columns
    return df

# -------------------- Data Type Inference --------------------
def infer_data_types(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64', 'datetime']).columns.tolist()
    
    for col in df.columns:
        if col not in numeric_cols + categorical_cols + datetime_cols:
            try:
                df[col] = pd.to_datetime(df[col])
                datetime_cols.append(col)
            except:
                continue
    return numeric_cols, categorical_cols, datetime_cols

# -------------------- Plot to Base64 --------------------
def fig_to_base64(fig) -> str:
    try:
        img_bytes = pio.to_image(fig, format="png")
        return base64.b64encode(img_bytes).decode("utf-8")
    except Exception as e:
        logger.warning(f"⚠️ Could not render figure to base64: {e}")
        return ""

# -------------------- Outlier Plot --------------------
def generate_outlier_plot(df: pd.DataFrame, column: str) -> str:
    try:
        fig = go.Figure()
        fig.add_trace(go.Box(y=df[column], name=column))
        return fig.to_json()
    except Exception as e:
        logger.error(f"❌ Outlier plot generation failed: {e}")
        return ""

# -------------------- Customer Segmentation --------------------
def perform_customer_segmentation(df: pd.DataFrame) -> pd.DataFrame:
    try:
        features = ["customer_lifetime_value", "credit_score"]
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df[features])
        kmeans = KMeans(n_clusters=3, random_state=42)
        df["segment"] = kmeans.fit_predict(df_scaled)
        return df
    except Exception as e:
        logger.error(f"❌ Segmentation failed: {e}")
        return df

# -------------------- Cluster Plot --------------------
def generate_cluster_plot(df: pd.DataFrame) -> str:
    try:
        import plotly.express as px
        fig = px.scatter(
            df,
            x="customer_lifetime_value",
            y="credit_score",
            color="segment",
            title="Customer Segmentation"
        )
        return fig.to_json()
    except Exception as e:
        logger.error(f"❌ Cluster plot failed: {e}")
        return ""
# -------------------- AI Summary --------------------
def generate_ai_summary(df: pd.DataFrame, insights: str) -> str:
    try:
        rows, cols = df.shape
        return f"📊 This dataset contains {rows} rows and {cols} columns. {insights}"
    except Exception as e:
        logger.error(f"❌ Summary generation failed: {e}")
        return "No summary available."
