import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed

# Ensure directories exist
model_dir = os.path.join(os.path.dirname(__file__), "../models")
os.makedirs(model_dir, exist_ok=True)
uploads_dir = os.path.join(os.path.dirname(__file__), "../uploads")
os.makedirs(uploads_dir, exist_ok=True)

# Check for dataset
data_files = [f for f in os.listdir(uploads_dir) if f.endswith(".csv")]
if not data_files:
    print(f"⚠️ No dataset found in {uploads_dir}. Please upload a CSV file and retry.")
    exit()

data_path = os.path.join(uploads_dir, data_files[0])
print(f"📂 Using dataset: {data_path}")

# Load dataset
df = pd.read_csv(data_path)

# Detect target column (first non-numeric column)
potential_targets = df.select_dtypes(exclude=["number"]).columns.tolist()
if not potential_targets:
    raise ValueError("❌ No categorical column found for the target.")

target_column = potential_targets[0]
print(f"✅ Auto-detected target column: {target_column}")

# Encode target
le = LabelEncoder()
df[target_column] = le.fit_transform(df[target_column])
joblib.dump(le, os.path.join(model_dir, "label_encoder.pkl"))

# Prepare features and target
X = df.drop(columns=[target_column])
y = df[target_column]

# Handle missing values
imputer = SimpleImputer(strategy="most_frequent")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
joblib.dump(imputer, os.path.join(model_dir, "imputer.pkl"))

# Encode categorical features
categorical_features = X.select_dtypes(include=["object"]).columns
if not categorical_features.empty:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_encoded = encoder.fit_transform(X[categorical_features])
    X = X.drop(columns=categorical_features)
    X_encoded = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out())
    X = pd.concat([X.reset_index(drop=True), X_encoded.reset_index(drop=True)], axis=1)
    joblib.dump(encoder, os.path.join(model_dir, "one_hot_encoder.pkl"))

# Standardize numerical data
scaler = StandardScaler()
X = scaler.fit_transform(X)
joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models with better parameters
models = {
    "logistic_regression": LogisticRegression(max_iter=2000, solver='liblinear', C=1.0),
    "random_forest": RandomForestClassifier(n_estimators=150, max_depth=None, min_samples_split=5, n_jobs=-1, random_state=42),
    "svm": SVC(kernel='rbf', C=1.0, gamma='scale')
}

def train_and_save_model(name, model, X_train, y_train, X_test, y_test, model_dir):
    print(f"🚀 Training {name} model...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Save model
    model_path = os.path.join(model_dir, f"{name}.pkl")
    joblib.dump(model, model_path)
    
    print(f"✅ {name} accuracy: {accuracy:.4f} (Model saved at {model_path})")
    return name, accuracy, model

# Train models in parallel
results = Parallel(n_jobs=-1)(delayed(train_and_save_model)(name, model, X_train, y_train, X_test, y_test, model_dir) for name, model in models.items())

# Select best model
best_model_name, best_accuracy, best_model = max(results, key=lambda x: x[1])

# Save best model
best_model_path = os.path.join(model_dir, "best_model.pkl")
joblib.dump(best_model, best_model_path)
print(f"🔥 Best model: {best_model_name} (Accuracy: {best_accuracy:.4f}) saved as {best_model_path}")
