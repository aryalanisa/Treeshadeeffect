import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

df = pd.read_csv ("File/Newdataset.csv")

# Features: numeric + categorical (one-hot encode)
numeric_feats = ["Agriculture",
    "Forest",
    "Grassland",
    "Water",
    "CanopyCover",
    "Imperviousdensity",
    "MeanCanopyHeight",
    "CanopyH_cv_20",
    "CanopyH_p95p5_20",
    "Shannon_landcover",
    "Vegetation",
    "GreenImperviousRatio",
    "log_Distancetowater",
    "log_Distancetoforest",
     "Hour",
    ]
categorical_feats = ['Surface','Location','Year', 'Landcover']


X = pd.get_dummies(df[numeric_feats + categorical_feats], drop_first = False)
print(X.columns)

#Comparing regression models
corr_threshold = 0.85
random_state = 42


# =========================
# 2. LOAD X, y
# =========================
X = df[numeric_feats + categorical_feats].copy()
y = df['Temperature_difference'].copy()


# =========================
# 4. TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_state
)


# =========================
# 5. PREPROCESSING
# =========================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_feats),
        ("cat", categorical_transformer, categorical_feats)
    ]
)


# =========================
# 6. MODELS
# =========================
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1
    ),
    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=random_state
    )
}


# =========================
# 7. TRAIN + EVALUATE
# =========================
results = []

fitted_pipelines = {}

for name, model in models.items():
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    results.append({
        "Model": name,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae
    })

    fitted_pipelines[name] = pipe

results_df = pd.DataFrame(results).sort_values("R2", ascending=False)

print(results_df)
