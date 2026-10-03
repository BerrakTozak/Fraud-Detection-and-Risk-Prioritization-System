# Credit Risk Prioritization with XGBoost

An end-to-end machine learning project for prioritizing higher-risk
credit applications under limited manual-review capacity.

The project uses the public Home Credit credit-application dataset from
Kaggle. Instead of treating the task only as a binary classification
problem, the model produces risk scores that can be used to rank
applications and create a capacity-constrained manual-review queue.

> **Important:** The dataset target represents repayment difficulty /
> credit risk and should not be interpreted as confirmed fraud. This
> repository therefore describes the project as credit-risk
> prioritization.

## Project Overview

The workflow covers:

-   Data exploration and cleaning
-   Missing-value handling
-   Historical application aggregation
-   Feature engineering
-   One-hot encoding of categorical variables
-   XGBoost model training
-   ROC-AUC and PR-AUC evaluation
-   Capacity-based review analysis
-   Calibration analysis
-   SHAP explainability
-   Robustness testing across multiple random splits
-   Bootstrap confidence intervals
-   Hypothetical cost-sensitivity analysis
-   Saved-model validation
-   Reproducible scoring and review-queue generation

The final encoded modeling dataset contains **307,511 applications** and
**129 predictive features**. The positive class represents approximately
**8.07%** of the applications.

## Technologies

-   Python
-   pandas
-   NumPy
-   scikit-learn
-   XGBoost
-   SHAP
-   Matplotlib
-   openpyxl

The project was developed as Python scripts in **PyCharm** rather than
notebooks.

## Model

The final model uses `XGBClassifier` with the following configuration:

``` python
XGBClassifier(
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
```

## Held-Out Performance

The main evaluation was performed on held-out data.

  Metric                   Result
  --------------------- ---------
  ROC-AUC                  0.7532
  PR-AUC                   0.2393
  Positive prevalence     \~8.07%

Accuracy is not treated as the primary metric because the target is
imbalanced. A classifier predicting the majority class for nearly every
application could obtain high accuracy while providing little value for
identifying higher-risk applications.

## Capacity-Based Review

A central part of the project is evaluating the model under limited
manual-review capacity.

    Review Capacity   Precision   Recall
  ----------------- ----------- --------
                 1%      45.92%    5.66%
                 2%      40.74%   10.01%
                 5%      33.13%   20.42%
                10%      26.95%   32.39%

At **5% review capacity**, approximately one-third of reviewed
applications belong to the positive class, compared with a population
prevalence of approximately 8.07%. This corresponds to roughly **4.1×
enrichment** over the base rate.

The decision threshold is therefore treated as a consequence of
available review capacity rather than as a universal risk boundary.

## Feature Engineering

The project includes engineered variables designed to represent current
behavior relative to historical or profile-level behavior. Examples
include:

-   Current credit relative to previous average credit
-   Current credit relative to previous maximum credit
-   Current goods price relative to previous averages
-   Current financing ratio
-   Change in financing ratio
-   Disagreement between external risk sources
-   New-customer indicator
-   Current credit relative to profile-level average credit
-   Capped historical credit z-score

These features provide explicit contextual relationships to the model.
No formal feature-ablation study was completed, so the repository does
not claim that every engineered feature independently improves
performance.

## Explainability

SHAP is used to investigate model behavior at both global and individual
levels.

The analysis includes:

-   Global feature importance
-   Beeswarm analysis
-   Local waterfall explanations
-   Dependence analysis
-   Feature interaction analysis

SHAP values explain how the model uses its inputs. They should **not**
be interpreted as evidence of causal relationships.

## Calibration

Predicted risk values were compared with observed positive rates across
probability ranges. Calibration was reasonably consistent across the
main populated ranges, while the highest-score ranges contained
relatively few observations and were therefore less stable.

The model obtained a Brier score of approximately **0.0681**, compared
with approximately **0.0742** for a prevalence-only baseline.

## Robustness

The model was evaluated across five random train/test splits.

-   Mean ROC-AUC: **0.7552**
-   ROC-AUC standard deviation: **0.0011**
-   Mean PR-AUC: **0.2418**
-   PR-AUC standard deviation: **0.0018**

A 1,000-sample bootstrap analysis on the primary held-out test set
produced approximately:

-   ROC-AUC 95% CI: **\[0.7461, 0.7601\]**
-   PR-AUC 95% CI: **\[0.2286, 0.2513\]**

These experiments support stability within the available historical
dataset, but they do not replace temporal validation on future data.

## Operational Pipeline

The final pipeline:

1.  Loads the saved XGBoost model.
2.  Loads the expected feature schema.
3.  Reads prepared application data.
4.  Applies validated preprocessing.
5.  Encodes categorical variables.
6.  Aligns the data to the expected 129-feature schema.
7.  Generates risk scores.
8.  Ranks applications by predicted risk.
9.  Selects the highest-risk applications according to review capacity.
10. Exports a ranked manual-review queue.

The saved model was reloaded and verified against the original model
output with a maximum probability difference of **0.0**.

## Limitations

The project is an experimental historical risk-prioritization system
rather than a production credit-decision engine.

Important limitations include:

-   No true out-of-time validation
-   Static public dataset
-   No production data-drift monitoring
-   No completed subgroup fairness audit
-   No formal feature-ablation study
-   Historical aggregates require a strict timestamp/leakage audit
    before production use
-   Current one-hot encoding could be replaced by a fitted encoder with
    explicit unknown-category handling
-   Hypothetical cost analysis does not represent real company costs

Potential future work includes temporal validation, drift monitoring,
periodic retraining, fairness analysis, feature ablation, stronger
categorical preprocessing, and automated raw-data ingestion.

## Dataset

The project is based on the public Home Credit dataset available on
Kaggle:

https://www.kaggle.com/datasets/mishra5001/credit-card/discussion/569675

The original dataset is **not included in this repository**. Download
the data from Kaggle and place it in your local data directory before
running the preprocessing pipeline.

## Repository Structure

A suggested structure is:

``` text
credit-risk-prioritization/
├── README.md
├── METHODOLOGY.md
├── requirements.txt
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── explain_model.py
│   └── final_pipeline.py
└── model/
    └── final_model_features.json
```

Large raw/intermediate datasets and generated review queues should
remain outside version control.

## Disclaimer

This project is for educational and portfolio purposes. Model scores
should not be used as the sole basis for real credit decisions. Any real
deployment would require additional validation, governance, fairness
testing, monitoring, security controls, and human oversight.
