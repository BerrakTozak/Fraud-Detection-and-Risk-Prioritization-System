import os
import json
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)

from xgboost import XGBClassifier


# =============================================================
# CONFIGURATION
# =============================================================

BASE_FOLDER = "/Users/berraktozak/Desktop/Krediler Data/Excel"

INPUT_FILE = os.path.join(
    BASE_FOLDER,
    "final_feature_engineered_data_before_split.xlsx"
)

MODEL_FILE = os.path.join(
    BASE_FOLDER,
    "final_xgboost_fraud_model.json"
)

FEATURE_SCHEMA_FILE = os.path.join(
    BASE_FOLDER,
    "final_model_features.json"
)

OUTPUT_FILE = os.path.join(
    BASE_FOLDER,
    "final_fraud_review_queue.xlsx"
)


# -------------------------------------------------------------
# Operational review capacity
# -------------------------------------------------------------

REVIEW_CAPACITY = 0.05


# =============================================================
# APPROVED MODEL FEATURES
# =============================================================

model_columns = [

    # ---------------------------------------------------------
    # Original categorical features
    # ---------------------------------------------------------

    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",

    # ---------------------------------------------------------
    # Original numerical / binary features
    # ---------------------------------------------------------

    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "DAYS_EMPLOYED",
    "DAYS_LAST_PHONE_CHANGE",
    "REGION_RATING_CLIENT",
    "REGION_RATING_CLIENT_W_CITY",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "FLAG_WORK_PHONE",
    "FLAG_CONT_MOBILE",
    "FLAG_PHONE",
    "FLAG_EMAIL",

    # ---------------------------------------------------------
    # Final engineered features
    # ---------------------------------------------------------

    "CURRENT_TO_PREV_AVG_CREDIT_RATIO",
    "CURRENT_TO_PREV_MAX_CREDIT_RATIO",
    "CURRENT_TO_PREV_AVG_GOODS_RATIO",
    "CURRENT_FINANCING_RATIO",
    "FINANCING_RATIO_CHANGE",
    "EXT_SOURCE_DISAGREEMENT",
    "HIGH_AMOUNT_LOW_SCORE_FLAG",
    "NEW_CUSTOMER_FLAG",
    "CURRENT_TO_PROFILE_AVG_CREDIT_RATIO",
    "CREDIT_Z_SCORE_CAPPED_50"
]


categorical_features = [
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE"
]


median_features = [
    "CNT_FAM_MEMBERS",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "DAYS_LAST_PHONE_CHANGE"
]


# =============================================================
# 1. LOAD SAVED MODEL
# =============================================================

print("Loading saved XGBoost model...")

model = XGBClassifier()

model.load_model(
    MODEL_FILE
)

print("Model loaded successfully.")


# =============================================================
# 2. LOAD EXPECTED 129-FEATURE SCHEMA
# =============================================================

with open(
    FEATURE_SCHEMA_FILE,
    "r"
) as file:

    expected_features = json.load(
        file
    )

print(
    f"Expected model features: "
    f"{len(expected_features)}"
)


# =============================================================
# 3. LOAD FEATURE-ENGINEERED APPLICATION DATA
# =============================================================

print("\nLoading applications...")

df = pd.read_excel(
    INPUT_FILE
)

print(
    f"Applications loaded: "
    f"{len(df):,}"
)


# =============================================================
# 4. PRESERVE IDENTIFIERS / TARGET IF AVAILABLE
# =============================================================
#
# These are NOT given to XGBoost.
# They are retained only so the output can identify applications.
# =============================================================

if "SK_ID_CURR" in df.columns:

    application_ids = df[
        "SK_ID_CURR"
    ].copy()

else:

    application_ids = pd.Series(
        range(1, len(df) + 1),
        name="Application_ID"
    )


if "TARGET" in df.columns:

    historical_target = df[
        "TARGET"
    ].copy()

else:

    historical_target = None


# =============================================================
# 5. VALIDATE PRE-ENCODING INPUT FEATURES
# =============================================================

missing_input_features = [
    feature
    for feature in model_columns
    if feature not in df.columns
]

if missing_input_features:

    print(
        "\nMissing required input features:"
    )

    for feature in missing_input_features:

        print(
            f"  - {feature}"
        )

    raise KeyError(
        "Input data does not contain all "
        "required model variables."
    )


# =============================================================
# 6. REPRODUCE VALIDATED MISSING-VALUE HANDLING
# =============================================================

processed_df = df.copy()


# -------------------------------------------------------------
# OCCUPATION TYPE
# -------------------------------------------------------------

processed_df["OCCUPATION_TYPE"] = (
    processed_df["OCCUPATION_TYPE"]
    .fillna("Unknown")
)


# -------------------------------------------------------------
# MEDIAN IMPUTATION
# -------------------------------------------------------------
#
# These transformations reproduce the preprocessing used
# for the validated historical modeling dataset.
# =============================================================

for feature in median_features:

    processed_df[feature] = (
        processed_df[feature]
        .fillna(
            processed_df[feature].median()
        )
    )


# =============================================================
# 7. SELECT ONLY APPROVED MODEL VARIABLES
# =============================================================

model_df = processed_df[
    model_columns
].copy()


# =============================================================
# 8. PRE-ENCODING MISSING-VALUE CHECK
# =============================================================

missing_values = (
    model_df
    .isna()
    .sum()
)

missing_values = (
    missing_values[
        missing_values > 0
    ]
)

if not missing_values.empty:

    print(
        "\nMissing values remain:"
    )

    print(
        missing_values
    )

    raise ValueError(
        "Preprocessing failed because missing "
        "values remain in model features."
    )


# =============================================================
# 9. ONE-HOT ENCODE CATEGORICAL FEATURES
# =============================================================

encoded_df = pd.get_dummies(
    model_df,
    columns=categorical_features,
    drop_first=True,
    dtype=int
)


# =============================================================
# 10. CHECK ENCODED SCHEMA
# =============================================================

generated_features = (
    encoded_df.columns.tolist()
)

missing_expected_features = [
    feature
    for feature in expected_features
    if feature not in generated_features
]

unexpected_features = [
    feature
    for feature in generated_features
    if feature not in expected_features
]


# =============================================================
# 11. ALIGN TO EXACT TRAINING SCHEMA
# =============================================================
#
# Missing dummy variables are safely created as zero.
#
# Unexpected columns are discarded because the final model
# only accepts the schema it was trained with.
# =============================================================

X_model = encoded_df.reindex(
    columns=expected_features,
    fill_value=0
)


# =============================================================
# 12. FINAL INPUT VALIDATION
# =============================================================

if X_model.shape[1] != len(expected_features):

    raise ValueError(
        "Final feature count does not match "
        "the saved model schema."
    )


if X_model.columns.tolist() != expected_features:

    raise ValueError(
        "Final feature order does not match "
        "the saved model schema."
    )


remaining_missing = int(
    X_model.isna()
    .sum()
    .sum()
)

if remaining_missing > 0:

    raise ValueError(
        f"{remaining_missing} missing values remain "
        "after preprocessing."
    )


print("\n" + "=" * 75)
print("PREPROCESSING VALIDATION")
print("=" * 75)

print(
    f"Generated encoded features: "
    f"{len(generated_features)}"
)

print(
    f"Expected model features:    "
    f"{len(expected_features)}"
)

print(
    f"Missing encoded features:   "
    f"{len(missing_expected_features)}"
)

print(
    f"Unexpected features:        "
    f"{len(unexpected_features)}"
)

print(
    f"Final aligned features:     "
    f"{X_model.shape[1]}"
)

print(
    f"Remaining missing values:   "
    f"{remaining_missing}"
)

print(
    "Schema validation:         PASSED"
)


# =============================================================
# 13. GENERATE FRAUD PROBABILITIES
# =============================================================

print(
    "\nScoring applications..."
)

fraud_probability = (
    model.predict_proba(
        X_model
    )[:, 1]
)


# =============================================================
# 14. BUILD SCORING RESULTS
# =============================================================

results = pd.DataFrame({
    "Application_ID":
        application_ids.to_numpy(),

    "Fraud_Probability":
        fraud_probability
})


# =============================================================
# OPTIONAL HISTORICAL TARGET
# =============================================================
#
# TARGET is included only because the demonstration file
# contains historical labels.
#
# Genuine new applications would not have this column.
# =============================================================

if historical_target is not None:

    results["Historical_TARGET"] = (
        historical_target.to_numpy()
    )


# =============================================================
# 15. RISK RANKING
# =============================================================

results["Risk_Rank"] = (
    results["Fraud_Probability"]
    .rank(
        ascending=False,
        method="first"
    )
    .astype(int)
)


# =============================================================
# 16. REVIEW CAPACITY
# =============================================================

max_reviews = int(
    len(results)
    * REVIEW_CAPACITY
)

results["Review_Decision"] = (
    results["Risk_Rank"]
    <= max_reviews
).map({
    True: "REVIEW",
    False: "NO REVIEW"
})


# =============================================================
# 17. SORT HIGHEST RISK FIRST
# =============================================================

results = (
    results
    .sort_values(
        by="Fraud_Probability",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


# =============================================================
# 18. IMPLIED OPERATIONAL THRESHOLD
# =============================================================

if max_reviews > 0:

    implied_threshold = (
        results.iloc[
            max_reviews - 1
        ]["Fraud_Probability"]
    )

else:

    implied_threshold = None


# =============================================================
# 18.5 HISTORICAL PERFORMANCE DIAGNOSTICS
# =============================================================
#
# IMPORTANT:
# These metrics are calculated only when historical TARGET
# values are available.
#
# Because the final production model was retrained on the full
# historical dataset, these are IN-SAMPLE diagnostics.
#
# Official model-performance claims should come from the
# held-out test set / repeated split / bootstrap analysis.
# =============================================================

if historical_target is not None:

    print("\n" + "=" * 80)
    print("HISTORICAL PERFORMANCE DIAGNOSTICS")
    print("=" * 80)

    y_true = historical_target.to_numpy()

    y_prob = fraud_probability


    # ---------------------------------------------------------
    # Probability-based metrics
    # ---------------------------------------------------------

    roc_auc = roc_auc_score(
        y_true,
        y_prob
    )

    pr_auc = average_precision_score(
        y_true,
        y_prob
    )


    # ---------------------------------------------------------
    # Capacity-based prediction
    # ---------------------------------------------------------
    #
    # Instead of using an arbitrary 0.50 threshold,
    # REVIEW cases are treated as positive predictions.
    # ---------------------------------------------------------

    y_pred_capacity = (
        results["Review_Decision"]
        .eq("REVIEW")
        .astype(int)
        .to_numpy()
    )

    # IMPORTANT:
    # results was sorted by risk, so TARGET must be taken
    # from the sorted results as well.

    y_true_sorted = (
        results["Historical_TARGET"]
        .astype(int)
        .to_numpy()
    )


    # ---------------------------------------------------------
    # Classification metrics at operational capacity
    # ---------------------------------------------------------

    accuracy = accuracy_score(
        y_true_sorted,
        y_pred_capacity
    )

    precision = precision_score(
        y_true_sorted,
        y_pred_capacity,
        zero_division=0
    )

    recall = recall_score(
        y_true_sorted,
        y_pred_capacity,
        zero_division=0
    )

    f1 = f1_score(
        y_true_sorted,
        y_pred_capacity,
        zero_division=0
    )


    # ---------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true_sorted,
            y_pred_capacity
        ).ravel()
    )


    # ---------------------------------------------------------
    # Print metrics
    # ---------------------------------------------------------

    print(
        f"ROC-AUC:                     "
        f"{roc_auc:.6f}"
    )

    print(
        f"PR-AUC:                      "
        f"{pr_auc:.6f}"
    )

    print(
        f"Fraud prevalence:            "
        f"{y_true.mean():.2%}"
    )

    print(
        f"\nOperational capacity:        "
        f"{REVIEW_CAPACITY:.1%}"
    )

    print(
        f"Accuracy:                    "
        f"{accuracy:.2%}"
    )

    print(
        f"Precision:                   "
        f"{precision:.2%}"
    )

    print(
        f"Recall:                      "
        f"{recall:.2%}"
    )

    print(
        f"F1 Score:                    "
        f"{f1:.4f}"
    )


    print("\nConfusion Matrix")

    print(
        f"True Negatives:              "
        f"{tn:,}"
    )

    print(
        f"False Positives:             "
        f"{fp:,}"
    )

    print(
        f"False Negatives:             "
        f"{fn:,}"
    )

    print(
        f"True Positives:              "
        f"{tp:,}"
    )


    # ---------------------------------------------------------
    # Fraud concentration / lift
    # ---------------------------------------------------------

    review_fraud_rate = precision

    overall_fraud_rate = (
        y_true_sorted.mean()
    )

    lift = (
        review_fraud_rate
        / overall_fraud_rate
        if overall_fraud_rate > 0
        else float("nan")
    )

    print(
        f"\nFraud rate overall:          "
        f"{overall_fraud_rate:.2%}"
    )

    print(
        f"Fraud rate in review queue:  "
        f"{review_fraud_rate:.2%}"
    )

    print(
        f"Review-queue lift:           "
        f"{lift:.2f}x"
    )


    # =========================================================
    # ROC CURVE
    # =========================================================

    fpr, tpr, roc_thresholds = (
        roc_curve(
            y_true,
            y_prob
        )
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        fpr,
        tpr,
        label=f"XGBoost ROC-AUC = {roc_auc:.3f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC Curve"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    ROC_OUTPUT_FILE = os.path.join(
        BASE_FOLDER,
        "final_model_roc_curve.png"
    )

    plt.savefig(
        ROC_OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


    # =========================================================
    # PRECISION-RECALL CURVE
    # =========================================================

    pr_precision, pr_recall, pr_thresholds = (
        precision_recall_curve(
            y_true,
            y_prob
        )
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        pr_recall,
        pr_precision,
        label=f"XGBoost PR-AUC = {pr_auc:.3f}"
    )

    plt.axhline(
        y=overall_fraud_rate,
        linestyle="--",
        label=(
            f"Prevalence Baseline = "
            f"{overall_fraud_rate:.3f}"
        )
    )

    plt.xlabel(
        "Recall"
    )

    plt.ylabel(
        "Precision"
    )

    plt.title(
        "Precision-Recall Curve"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    PR_OUTPUT_FILE = os.path.join(
        BASE_FOLDER,
        "final_model_precision_recall_curve.png"
    )

    plt.savefig(
        PR_OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


    print(
        f"\nROC curve saved to:\n"
        f"{ROC_OUTPUT_FILE}"
    )

    print(
        f"\nPrecision-Recall curve saved to:\n"
        f"{PR_OUTPUT_FILE}"
    )

# =============================================================
# 19. CREATE RISK BANDS
# =============================================================
#
# These bands are descriptive only.
# The operational REVIEW decision is still determined
# by ranking/capacity.
# =============================================================

results["Risk_Band"] = pd.cut(
    results["Fraud_Probability"],
    bins=[
        -float("inf"),
        0.10,
        0.20,
        0.30,
        float("inf")
    ],
    labels=[
        "Low",
        "Moderate",
        "High",
        "Very High"
    ]
)


# =============================================================
# 20. EXPORT REVIEW QUEUE
# =============================================================

results.to_excel(
    OUTPUT_FILE,
    index=False
)


# =============================================================
# 21. FINAL PIPELINE SUMMARY
# =============================================================

print("\n" + "=" * 80)
print("END-TO-END FRAUD SCORING PIPELINE COMPLETE")
print("=" * 80)

print(
    f"Applications processed:     "
    f"{len(results):,}"
)

print(
    f"Model features used:        "
    f"{X_model.shape[1]}"
)

print(
    f"Review capacity:            "
    f"{REVIEW_CAPACITY:.1%}"
)

print(
    f"Applications sent to review:"
    f" {max_reviews:,}"
)


if implied_threshold is not None:

    print(
        f"Implied risk threshold:     "
        f"{implied_threshold:.4f}"
    )


print(
    f"Maximum predicted risk:     "
    f"{results['Fraud_Probability'].max():.2%}"
)

print(
    f"Minimum predicted risk:     "
    f"{results['Fraud_Probability'].min():.2%}"
)

print(
    f"\nReview queue saved to:\n"
    f"{OUTPUT_FILE}"
)


# =============================================================
# 22. DISPLAY TOP 20 RISK CASES
# =============================================================

print("\n" + "=" * 80)
print("TOP 20 HIGHEST-RISK APPLICATIONS")
print("=" * 80)

display_columns = [
    "Application_ID",
    "Fraud_Probability",
    "Risk_Rank",
    "Risk_Band",
    "Review_Decision"
]

if historical_target is not None:

    display_columns.insert(
        2,
        "Historical_TARGET"
    )


print(
    results[
        display_columns
    ]
    .head(20)
    .to_string(
        index=False,
        formatters={
            "Fraud_Probability":
                "{:.2%}".format
        }
    )
)