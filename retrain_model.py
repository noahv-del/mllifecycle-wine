"""
Retrain and save the model with current environment
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

print("Loading data...")
# Load datasets
red = pd.read_csv('winequality-red.csv', sep=';')
white = pd.read_csv('winequality-white.csv', sep=';')

# Add color column
red['color'] = 'red'
white['color'] = 'white'
df = pd.concat([red, white], ignore_index=True)

print(f"Dataset shape: {df.shape}")

# Add missing values (as in original notebook)
np.random.seed(42)
for col in ['residual sugar', 'chlorides', 'sulphates']:
    missing_indices = df.sample(frac=0.01).index
    df.loc[missing_indices, col] = np.nan

# Add outliers
outlier_indices = df.sample(frac=0.005).index
df.loc[outlier_indices, 'alcohol'] *= 3
df.loc[outlier_indices, 'density'] *= 2

# Handle missing values
num_features = df.select_dtypes(include=['float64', 'int64']).columns.drop('quality')
df[num_features] = df[num_features].fillna(df[num_features].median())

# Handle outliers
for col in num_features:
    lower, upper = df[col].quantile([0.01, 0.99])
    df[col] = np.clip(df[col], lower, upper)

# Feature engineering
df["sulfur_ratio"] = df["free sulfur dioxide"] / (df["total sulfur dioxide"] + 1e-6)
df["sulfur_ratio"] = df["sulfur_ratio"].replace([np.inf, -np.inf], np.nan).fillna(df["sulfur_ratio"].median())

print("Creating target variable...")
# Define target (binary classification)
df["quality_label"] = (df["quality"] >= 7).astype(int)
y = df["quality_label"]
X = df.drop(columns=["quality", "quality_label"])

print(f"Class distribution: {y.value_counts().to_dict()}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training size: {X_train.shape}, Test size: {X_test.shape}")

# Define feature types
num_features_list = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
cat_features = ["color"]

print("Building pipeline...")
# Preprocessing pipelines
log_transformer = FunctionTransformer(np.log1p, validate=False)

numeric_pipeline = Pipeline(steps=[
    ("log", log_transformer),
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.95))
])

categorical_pipeline = Pipeline(steps=[
    ("onehot", OneHotEncoder(drop="first"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, num_features_list),
        ("cat", categorical_pipeline, cat_features)
    ]
)

# Full pipeline with model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    max_depth=None
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", model)
])

print("Training model...")
pipeline.fit(X_train, y_train)

# Evaluate
train_score = pipeline.score(X_train, y_train)
test_score = pipeline.score(X_test, y_test)
print(f"Training accuracy: {train_score:.4f}")
print(f"Test accuracy: {test_score:.4f}")

# Save models
print("Saving models...")
joblib.dump(preprocessor, "preprocessor.pkl")
joblib.dump(pipeline, "pipeline_base.pkl")
joblib.dump(model, "random_forest_model.pkl")

joblib.dump({
    "X_train": X_train,
    "X_test": X_test,
    "y_train": y_train,
    "y_test": y_test
}, "train_test_data.pkl")

print("✓ Models saved successfully!")
print("Files created:")
print("  - preprocessor.pkl")
print("  - pipeline_base.pkl")
print("  - random_forest_model.pkl")
print("  - train_test_data.pkl")
