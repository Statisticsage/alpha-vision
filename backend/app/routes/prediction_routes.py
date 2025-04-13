# --- Imports ---
import os, json, logging, asyncio, joblib
import pandas as pd
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import plotly.express as px
import plotly.graph_objects as go
from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_regression
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.uploaded_file import UploadedFile

# New imports for advanced models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

# --- Config ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
RESULTS_DIR = "prediction_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Define proper serializable model wrappers ---
# These need to be top-level classes that can be properly serialized

class KerasModelWrapper:
    """Wrapper for Keras models that can be serialized by joblib"""
    def __init__(self, model=None, scaler=None):
        self.model = model
        self.scaler = scaler
        
    def predict(self, X):
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.model.predict(X).flatten()
    
    def __getstate__(self):
        """Custom serialization method"""
        model_path = os.path.join(RESULTS_DIR, f"temp_keras_model_{id(self)}.h5")
        if self.model is not None:
            self.model.save(model_path)
        
        state = self.__dict__.copy()
        state['model'] = model_path if self.model is not None else None
        return state
    
    def __setstate__(self, state):
        """Custom deserialization method"""
        self.__dict__ = state
        model_path = state['model']
        if model_path is not None and os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            os.remove(model_path)  # Clean up temp file
        else:
            self.model = None

class TorchModelWrapper:
    """Wrapper for PyTorch models that can be serialized by joblib"""
    def __init__(self, model=None, scaler=None):
        self.model = model
        self.scaler = scaler
        
    def predict(self, X):
        if self.scaler is not None:
            X = self.scaler.transform(X)
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.model.eval()
        with torch.no_grad():
            return self.model(X_tensor).numpy().flatten()
    
    def __getstate__(self):
        """Custom serialization method"""
        model_path = os.path.join(RESULTS_DIR, f"temp_torch_model_{id(self)}.pt")
        if self.model is not None:
            torch.save(self.model.state_dict(), model_path)
        
        # Store model class and init parameters
        model_class = self.model.__class__
        model_params = {}
        if hasattr(self.model, 'input_dim'):
            model_params['input_dim'] = self.model.input_dim
        if hasattr(self.model, 'embed_dim'):
            model_params['embed_dim'] = self.model.embed_dim
        if hasattr(self.model, 'num_layers'):
            model_params['num_layers'] = self.model.num_layers
        if hasattr(self.model, 'num_heads'):
            model_params['num_heads'] = self.model.num_heads
        
        state = self.__dict__.copy()
        state['model'] = None
        state['model_path'] = model_path
        state['model_class'] = model_class
        state['model_params'] = model_params
        return state
    
    def __setstate__(self, state):
        """Custom deserialization method"""
        self.__dict__ = state
        model_class = state.pop('model_class', None)
        model_params = state.pop('model_params', {})
        model_path = state.pop('model_path', None)
        
        if model_class and model_path and os.path.exists(model_path):
            # Recreate the model instance
            self.model = model_class(**model_params)
            self.model.load_state_dict(torch.load(model_path))
            os.remove(model_path)  # Clean up temp file
        else:
            self.model = None

# --- Transformer Model Definition ---
class TransformerTimeSeries(nn.Module):
    def __init__(self, input_dim, embed_dim=64, num_layers=2, num_heads=4):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        self.input_proj = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(embed_dim, 1)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.dropout(x)
        x = self.encoder(x)
        return self.fc(x.mean(dim=1))

# --- FastAPI Setup ---
app = FastAPI()
prediction_router = APIRouter()
app.mount("/prediction_results", StaticFiles(directory=RESULTS_DIR), name="prediction_results")
app.include_router(prediction_router)

# --- Load Data Helper ---
def load_dataframe(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(file_path)
    elif ext in [".xls", ".xlsx"]:
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format")

# --- Trigger Prediction Endpoint ---
@prediction_router.post("/prediction/{file_id}")
async def make_prediction(file_id: int, background_tasks: BackgroundTasks):
    db = next(get_db())
    try:
        file = db.query(UploadedFile).filter(UploadedFile.id == file_id).first()
        if not file or not os.path.exists(file.filepath):
            raise HTTPException(status_code=404, detail="File not found")
        background_tasks.add_task(perform_prediction, file.filepath, file_id)
        return {"message": "Prediction started. Results will be available soon."}
    finally:
        db.close()

# --- Get Prediction Result ---
@prediction_router.get("/result/{file_id}")
async def get_result(file_id: int):
    result_path = os.path.join(RESULTS_DIR, f"result_{file_id}.json")
    if not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Result generating please wait/refresh")
    
    with open(result_path) as f:
        result = json.load(f)
    
    # Add logging to check if plots exist and their format
    if 'prediction_result' in result and 'plots' in result['prediction_result']:
        plots = result['prediction_result']['plots']
        logger.info(f"Found {len(plots)} plots in result for file_id={file_id}")
        
        # Check if plots are valid JSON
        for i, plot in enumerate(plots):
            try:
                # Try to parse the plot JSON to verify it's valid
                plot_data = json.loads(plot)
                logger.info(f"Plot {i} is valid JSON with {len(plot_data.get('data', []))} data traces")
            except json.JSONDecodeError as e:
                logger.error(f"Plot {i} contains invalid JSON: {e}")
    else:
        logger.warning(f"No plots found in result for file_id={file_id}")
    
    return JSONResponse(content=result)

# --- Automated Target Selection ---
def select_best_target(df):
    """
    Automatically select the best target column for prediction based on:
    1. Data completeness (fewer missing values)
    2. Variance (higher variance indicates more information)
    3. Correlation with other features
    """
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if len(numeric_cols) < 2:
        raise ValueError("Dataset must contain at least two numeric columns for prediction.")
    
    # Calculate metrics for each potential target
    target_scores = {}
    
    for col in numeric_cols:
        # Skip columns with too many missing values (>30%)
        missing_pct = df[col].isna().mean()
        if missing_pct > 0.3:
            continue
            
        # Calculate normalized variance (higher is better)
        variance = df[col].var()
        if variance == 0:
            continue
            
        # Calculate mean absolute correlation with other features
        other_cols = [c for c in numeric_cols if c != col]
        if not other_cols:
            continue
            
        correlations = df[other_cols].corrwith(df[col]).abs()
        mean_correlation = correlations.mean()
        
        # Calculate a composite score (higher is better)
        # We want: low missing values, high variance, moderate to high correlation
        completeness_score = 1 - missing_pct
        variance_score = min(variance / df[col].mean() if df[col].mean() != 0 else 0, 1)
        correlation_score = mean_correlation
        
        composite_score = (0.4 * completeness_score + 
                          0.3 * variance_score + 
                          0.3 * correlation_score)
        
        target_scores[col] = composite_score
    
    if not target_scores:
        raise ValueError("No suitable target columns found in the dataset.")
    
    # Select the column with the highest score
    best_target = max(target_scores.items(), key=lambda x: x[1])[0]
    logger.info(f"Automatically selected target column: {best_target} (score: {target_scores[best_target]:.3f})")
    
    return best_target

# --- Core Prediction Logic ---
async def perform_prediction(file_path: str, file_id: int):
    try:
        logger.info(f"Starting prediction for file_id={file_id}")
        df = await asyncio.to_thread(load_dataframe, file_path)
        
        # Automatically select the best target column
        target_column = select_best_target(df)
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # Drop rows with missing values
        df_clean = df[numeric_cols].dropna()
        
        # Prepare features and target
        X = df_clean.drop(columns=[target_column])
        y = df_clean[target_column]
        
        # Split data into train and test sets (80/20 split)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model_dir = os.path.join(os.path.dirname(__file__), "../models")
        os.makedirs(model_dir, exist_ok=True)
        
        # Choose model based on dataset size
        n_samples = len(X_train)
        
        if n_samples < 100:
            model, predictions, metrics = await asyncio.to_thread(
                train_linear_regression, X_train, y_train, X_test, y_test
            )
            model_type = "LinearRegression"
        elif n_samples < 500:
            model, predictions, metrics = await asyncio.to_thread(
                train_decision_tree, X_train, y_train, X_test, y_test
            )
            model_type = "DecisionTree"
        elif n_samples < 1000:
            model, predictions, metrics = await asyncio.to_thread(
                train_svm, X_train, y_train, X_test, y_test
            )
            model_type = "SVM"
        elif n_samples < 5000:
            model, predictions, metrics = await asyncio.to_thread(
                train_mlp, X_train, y_train, X_test, y_test
            )
            model_type = "MLP"
        else:
            model, predictions, metrics = await asyncio.to_thread(
                train_transformer, X_train, y_train, X_test, y_test
            )
            model_type = "Transformer"
        
        # Save the model
        model_path = os.path.join(model_dir, f"model_{model_type}_{file_id}.pkl")
        await asyncio.to_thread(joblib.dump, model, model_path)
        
        # Add target column info to metrics
        metrics['target_column'] = target_column
        
        # Generate plots
        plots = generate_plotly_plots(df, y_test, predictions, numeric_cols, metrics)
        
        # Prepare result
        result = {
            "file_id": file_id,
            "status": "success",
            "prediction_result": {
                **metrics,
                "model_type": model_type,
                "plots": plots
            }
        }
        
        with open(os.path.join(RESULTS_DIR, f"result_{file_id}.json"), "w") as f:
            json.dump(result, f)
        
        logger.info(f"Prediction complete for file_id={file_id}")
    
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        with open(os.path.join(RESULTS_DIR, f"result_{file_id}.json"), "w") as f:
            json.dump({
                "file_id": file_id, 
                "status": "error", 
                "error_message": str(e)
            }, f)

# --- Advanced Model Trainers with Hyperparameter Tuning ---
def train_linear_regression(X_train, y_train, X_test, y_test):
    # Create a pipeline with preprocessing and model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])
    
    # Define hyperparameters to tune
    param_grid = {
        'model__fit_intercept': [True, False],
    }
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='neg_mean_squared_error'
    )
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Make predictions on test set
    predictions = best_model.predict(X_test)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, predictions)
    metrics['best_params'] = grid_search.best_params_
    
    # Calculate feature importance (coefficients for linear regression)
    feature_importance = np.abs(best_model.named_steps['model'].coef_)
    metrics['feature_importance'] = feature_importance.tolist()
    metrics['feature_names'] = X_train.columns.tolist()
    
    return best_model, predictions, metrics

def train_decision_tree(X_train, y_train, X_test, y_test):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', DecisionTreeRegressor(random_state=42))
    ])
    
    param_grid = {
        'model__max_depth': [None, 5, 10, 15, 20],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    }
    
    # Use RandomizedSearchCV for efficiency with larger parameter space
    random_search = RandomizedSearchCV(
        pipeline, param_grid, n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42
    )
    random_search.fit(X_train, y_train)
    
    best_model = random_search.best_estimator_
    predictions = best_model.predict(X_test)
    
    metrics = calculate_metrics(y_test, predictions)
    metrics['best_params'] = random_search.best_params_
    
    # Get feature importance
    feature_importance = best_model.named_steps['model'].feature_importances_
    metrics['feature_importance'] = feature_importance.tolist()
    metrics['feature_names'] = X_train.columns.tolist()
    
    return best_model, predictions, metrics

def train_svm(X_train, y_train, X_test, y_test):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVR())
    ])
    
    param_grid = {
        'model__C': [0.1, 1, 10, 100],
        'model__kernel': ['linear', 'rbf'],
        'model__gamma': ['scale', 'auto', 0.1, 0.01]
    }
    
    # Use RandomizedSearchCV for efficiency
    random_search = RandomizedSearchCV(
        pipeline, param_grid, n_iter=8, cv=3, scoring='neg_mean_squared_error', random_state=42
    )
    random_search.fit(X_train, y_train)
    
    best_model = random_search.best_estimator_
    predictions = best_model.predict(X_test)
    
    metrics = calculate_metrics(y_test, predictions)
    metrics['best_params'] = random_search.best_params_
    
    # For SVR, we don't have direct feature importance, so use permutation importance
    # or just skip this part
    metrics['feature_importance'] = None
    
    return best_model, predictions, metrics

def train_mlp(X_train, y_train, X_test, y_test):
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define model architecture with early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
    
    # Try different architectures
    architectures = [
        [64, 32],
        [128, 64, 32],
        [64, 64, 32]
    ]
    
    best_val_loss = float('inf')
    best_model = None
    best_arch = None
    
    # Create validation set
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=42
    )
    
    for arch in architectures:
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)))
        
        for units in arch:
            model.add(tf.keras.layers.Dense(units, activation='relu'))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dropout(0.2))
        
        model.add(tf.keras.layers.Dense(1))
        
        model.compile(
            loss="mse", 
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
        )
        
        # Use validation split for early stopping
        history = model.fit(
            X_train_final, y_train_final, 
            epochs=50, 
            batch_size=32, 
            verbose=0,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping]
        )
        
        val_loss = min(history.history['val_loss'])
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            best_arch = arch    
    predictions = best_model.predict(X_test_scaled).flatten()
    
    metrics = calculate_metrics(y_test, predictions)
    metrics['architecture'] = str(best_arch)
    
    # Create a properly serializable model wrapper
    model_wrapper = KerasModelWrapper(best_model, scaler)
    
    return model_wrapper, predictions, metrics

def train_transformer(X_train, y_train, X_test, y_test):
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(1)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(1)
    
    # Define hyperparameters to try
    configs = [
        {'embed_dim': 64, 'num_layers': 2, 'num_heads': 4, 'lr': 0.001},
        {'embed_dim': 128, 'num_layers': 3, 'num_heads': 8, 'lr': 0.0005},
        {'embed_dim': 64, 'num_layers': 4, 'num_heads': 4, 'lr': 0.0001}
    ]
    
    best_val_loss = float('inf')
    best_model = None
    best_config = None
    
    # Create a validation set
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_tensor, y_train_tensor, test_size=0.2, random_state=42
    )
    
    for config in configs:
        model = TransformerTimeSeries(
            input_dim=X_train_scaled.shape[1],
            embed_dim=config['embed_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads']
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        criterion = nn.MSELoss()
        
        # Training loop with validation
        patience = 5
        counter = 0
        best_epoch_loss = float('inf')
        best_epoch_weights = None
        
        for epoch in range(30):
            # Training
            model.train()
            optimizer.zero_grad()
            output = model(X_train_final)
            loss = criterion(output, y_train_final)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_output = model(X_val)
                val_loss = criterion(val_output, y_val)
            
            # Early stopping check
            if val_loss < best_epoch_loss:
                best_epoch_loss = val_loss
                best_epoch_weights = model.state_dict().copy()
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    break
        
        # Load best weights from early stopping
        if best_epoch_weights:
            model.load_state_dict(best_epoch_weights)
        
        # Compare with best model so far
        if best_epoch_loss < best_val_loss:
            best_val_loss = best_epoch_loss
            best_model = model
            best_config = config
    
    # Make predictions with the best model
    best_model.eval()
    with torch.no_grad():
        predictions = best_model(X_test_tensor).numpy().flatten()
    
    metrics = calculate_metrics(y_test, predictions)
    metrics['best_config'] = best_config
    
    # Create a properly serializable model wrapper
    model_wrapper = TorchModelWrapper(best_model, scaler)
    
    return model_wrapper, predictions, metrics

def train_lstm(X_train, y_train, X_test, y_test):
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define hyperparameters to try
    configs = [
        {'units': [50, 50], 'dropout': 0.2, 'lr': 0.001},
        {'units': [100, 50], 'dropout': 0.3, 'lr': 0.0005},
        {'units': [64, 32, 16], 'dropout': 0.2, 'lr': 0.001}
    ]
    
    best_val_loss = float('inf')
    best_model = None
    best_config = None
    
    # Create validation set
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=42
    )
    
    for config in configs:
        # Build model
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)))
        model.add(tf.keras.layers.Reshape((1, X_train_scaled.shape[1])))
        
        for i, units in enumerate(config['units']):
            return_sequences = i < len(config['units']) - 1
            model.add(tf.keras.layers.LSTM(units, return_sequences=return_sequences))
            model.add(tf.keras.layers.Dropout(config['dropout']))
        
        model.add(tf.keras.layers.Dense(1))
        
        # Compile
        model.compile(
            loss="mse", 
            optimizer=tf.keras.optimizers.Adam(learning_rate=config['lr'])
        )
        
        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
        
        # Train
        history = model.fit(
            X_train_final, y_train_final,
            epochs=30,
            batch_size=32,
            verbose=0,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping]
        )
        
        val_loss = min(history.history['val_loss'])
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            best_config = config
    
    # Make predictions
    predictions = best_model.predict(X_test_scaled).flatten()
    
    metrics = calculate_metrics(y_test, predictions)
    metrics['best_config'] = best_config
    
    # Create a properly serializable model wrapper
    model_wrapper = KerasModelWrapper(best_model, scaler)
    
    return model_wrapper, predictions, metrics

# --- Metrics Calculation Helper ---
def calculate_metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "avg_prediction": float(np.mean(y_pred))
    }

# --- Visualization Generator ---
def generate_plotly_plots(df, y_true, y_pred, numeric_cols, metrics):
    plots = []
    
    # Get target column
    target_column = metrics.get('target_column', numeric_cols[0])
    
    # Histogram of the target column
    if target_column in df.columns:
        hist = px.histogram(df, x=target_column, nbins=30, title=f"Distribution of {target_column}")
        plot_json = hist.to_json()
        logger.info(f"Generated histogram plot, size: {len(plot_json)} bytes")
        plots.append(plot_json)
    
    # Actual vs Predicted scatter plot
    pred_fig = px.scatter(
        x=y_true, y=y_pred,
        labels={"x": "Actual", "y": "Predicted"},
        title=f"Actual vs Predicted (R² = {metrics['R2']:.3f})"
    )
    # Add perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    pred_fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        )
    )
    plot_json = pred_fig.to_json()
    logger.info(f"Generated scatter plot, size: {len(plot_json)} bytes")
    plots.append(plot_json)
    
    # Time series plot for all numeric columns
    time_fig = go.Figure()
    for col in numeric_cols:
        if col in df.columns:  # Make sure column exists
            time_fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=col))
    time_fig.update_layout(title="Feature Trends", xaxis_title="Time", yaxis_title="Value")
    plot_json = time_fig.to_json()
    logger.info(f"Generated time series plot, size: {len(plot_json)} bytes")
    plots.append(plot_json)
    
    # Residuals Plot (Actual - Predicted vs Predicted)
    residuals = y_true - np.array(y_pred).squeeze()
    resid_fig = px.scatter(
        x=y_pred,
        y=residuals,
        title=f"Residuals vs Predicted Values (RMSE = {metrics['RMSE']:.3f})",
        labels={"x": "Predicted Values", "y": "Residuals"}
    )
    # Add horizontal line at y=0
    resid_fig.add_hline(y=0, line_dash="dash", line_color="red")
    plot_json = resid_fig.to_json()
    logger.info(f"Generated residuals plot, size: {len(plot_json)} bytes")
    plots.append(plot_json)
    
    # Correlation Heatmap of the dataset
    # Create a copy of the dataframe with numeric columns only
    corr_df = df[numeric_cols].copy()
    
    corr_fig = px.imshow(
        corr_df.corr(),
        text_auto=True,
        title="Correlation Heatmap",
        color_continuous_scale="RdBu_r"
    )
    plot_json = corr_fig.to_json()
    logger.info(f"Generated correlation heatmap, size: {len(plot_json)} bytes")
    plots.append(plot_json)
    
    logger.info(f"Total plots generated: {len(plots)}")
    return plots
