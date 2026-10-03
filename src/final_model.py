import os
import json
import pandas as pd

from xgboost import XGBClassifier


# =============================================================
# FILE PATHS
# =============================================================

BASE_FOLDER = "/Users/berraktozak/Desktop/Krediler Data/Excel"

ENCODED_FILE = os.path.join(
    BASE_FOLDER,
    "encoded_modeling_data.xlsx"
)

MODEL_FILE = os.path.join(
    BASE_FOLDER,
    "final_xgboost_fraud_model.json"
)

FEATURE_FILE = os.path.join(
    BASE_FOLDER,
    "final_model_features.json"
)


# =============================================================
# LOAD FINAL MODELING DATA
# =============================================================

df = pd.read_excel(
    ENCODED_FILE
)

X = df.drop(
    columns=["TARGET"]
)

y = df["TARGET"]


# =============================================================
# DEFINE FINAL XGBOOST MODEL
# =============================================================

final_model = XGBClassifier(
    learning_rate=0.001,
    n_estimators=5000,
    max_depth=9,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10,
    gamma=0.5,
    random_state=42,
    tree_method="hist",
    n_jobs=-1
)


# =============================================================
# TRAIN ON ALL AVAILABLE DATA
# =============================================================
#
# Evaluation is already complete.
#
# The final production model can therefore be trained using
# all available labeled observations.
# =============================================================

print(
    "Training final model on all available data..."
)

final_model.fit(
    X,
    y
)


# =============================================================
# SAVE XGBOOST MODEL
# =============================================================

final_model.save_model(
    MODEL_FILE
)


# =============================================================
# SAVE EXPECTED FEATURE LIST
# =============================================================
#
# This is VERY important for future inference.
#
# The model expects exactly these columns in exactly
# this order.
# =============================================================

feature_names = X.columns.tolist()

with open(
    FEATURE_FILE,
    "w"
) as file:

    json.dump(
        feature_names,
        file,
        indent=4
    )

loaded_model = XGBClassifier()

loaded_model.load_model(MODEL_FILE)

original_prob = final_model.predict_proba(X.head(100))[:, 1]

loaded_prob = loaded_model.predict_proba(
    X.head(100))[:, 1]

max_difference = abs(
    original_prob - loaded_prob
).max()


# =============================================================
# FINAL INFORMATION
# =============================================================

print("\n" + "=" * 75)
print("FINAL MODEL SAVED SUCCESSFULLY")
print("=" * 75)

print(
    f"Training Applications: "
    f"{len(X):,}"
)

print(
    f"Frauds:                "
    f"{int(y.sum()):,}"
)

print(
    f"Fraud Rate:            "
    f"{y.mean():.2%}"
)

print(
    f"Number of Features:    "
    f"{X.shape[1]}"
)

print(
    f"\nModel saved to:\n"
    f"{MODEL_FILE}"
)

print(
    f"\nFeature schema saved to:\n"
    f"{FEATURE_FILE}"
)

print(
    f"\nReload Verification "
    f"Max Probability Difference: "
    f"{max_difference:.10f}"
)

if max_difference < 1e-8:

    print(
        "Verification: PASSED"
    )

else:

    print(
        "Verification: CHECK MODEL"
    )