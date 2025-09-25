# ==============================
# Week 5–6: Data Preprocessing
# ==============================

# Step 1: Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from pathlib import Path

# Step 2: Load Dataset
data_path = Path("data/raw/diabetic_data.csv")
df = pd.read_csv(data_path)

# Step 3: Drop ID-like columns (non-predictive)
df = df.drop(columns=['encounter_id', 'patient_nbr'])

# Step 4: Handle missing values (replace '?' with NaN)
df = df.replace('?', np.nan)

# Step 5: Define target variable (readmitted)
# Convert to binary: <30 → 1, otherwise 0
df['readmitted_binary'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)

# Step 6: Separate features & target
X = df.drop(columns=['readmitted', 'readmitted_binary'])
y = df['readmitted_binary']

# Step 7: Identify column types
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Step 8: Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Step 9: Fit the preprocessor and transform dataset
X_processed = preprocessor.fit_transform(X)

# Step 10: Convert to dense DataFrame
try:
    X_dense = X_processed.toarray()  # works if sparse
except AttributeError:
    X_dense = X_processed  # already dense

# Get feature names after encoding
cat_cols = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_features)
all_features = np.concatenate([numeric_features, cat_cols])

df_processed = pd.DataFrame(X_dense, columns=all_features)

# Add target column
df_processed['readmitted'] = y.values

# Step 11: Save processed dataset
output_path = Path("data/processed")
output_path.mkdir(parents=True, exist_ok=True)
df_processed.to_csv(output_path / "diabetic_clean.csv", index=False)

print("✅ Preprocessing complete. Cleaned dataset saved at: data/processed/diabetic_clean.csv")
print("Final shape:", df_processed.shape)