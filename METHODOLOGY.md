# Methodology

## 1. Problem Definition

The project investigates how machine learning can prioritize higher-risk
credit applications when only a limited percentage of applications can
be manually reviewed.

The target is imbalanced, with the positive class representing
approximately 8.07% of the available applications. Because of this, the
objective is not simply to maximize classification accuracy. The model
should rank applications so that positive cases are concentrated toward
the top of the review queue.

The original Home Credit target is associated with repayment difficulty
/ credit risk. It is not treated as verified fraud in this repository.

## 2. Data Preparation

The modeling workflow combines selected current-application variables
with information derived from historical applications.

The preprocessing process includes:

1.  Loading the prepared application and historical information.
2.  Selecting model variables.
3.  Handling missing numerical and categorical values.
4.  Creating engineered features.
5.  One-hot encoding categorical variables.
6.  Aligning the resulting matrix to a fixed model schema.
7.  Verifying feature order and remaining missing values.

The final encoded dataset contains 129 predictive features.

### Missing Values

Different missing values are handled according to their meaning rather
than using one universal strategy.

Median imputation is used for selected numerical variables where a
robust central value is appropriate.

Some historical comparison features use zero when the historical
comparison is unavailable. In this context, zero is a chosen
representation of unavailable historical information rather than a
mathematically neutral value.

A separate new-customer indicator provides additional context for
applications with limited history.

## 3. Feature Engineering

The engineered features are intended to represent relationships that are
not directly available from individual raw variables.

The final feature set includes:

-   `CURRENT_TO_PREV_AVG_CREDIT_RATIO`
-   `CURRENT_TO_PREV_MAX_CREDIT_RATIO`
-   `CURRENT_TO_PREV_AVG_GOODS_RATIO`
-   `CURRENT_FINANCING_RATIO`
-   `FINANCING_RATIO_CHANGE`
-   `EXT_SOURCE_DISAGREEMENT`
-   `HIGH_AMOUNT_LOW_SCORE_FLAG`
-   `NEW_CUSTOMER_FLAG`
-   `CURRENT_TO_PROFILE_AVG_CREDIT_RATIO`
-   `CREDIT_Z_SCORE_CAPPED_50`

### Profile-Based Credit Ratio

A profile is constructed from variables describing income type,
occupation type, and organization type. The average credit amount for
similar profiles is calculated and the current requested credit is
compared with that value.

This feature provides context for whether an application is unusually
large relative to applicants with similar characteristics.

### Capped Credit Z-Score

A historical credit z-score is calculated using the applicant's previous
average and standard deviation of credit amounts.

Extreme values are clipped to the range `[-50, 50]` to prevent unusually
large values from dominating the representation.

## 4. Categorical Encoding

Categorical variables are converted using one-hot encoding.

The development workflow used:

``` python
pd.get_dummies(
    model_df,
    columns=categorical_features,
    drop_first=True,
    dtype=int
)
```

For inference, generated columns are aligned to the saved feature
schema. Missing expected columns are added with zero values and
unexpected columns are excluded.

For a production system, a fitted encoder such as
`OneHotEncoder(handle_unknown="ignore")` would provide stronger and more
explicit handling of unseen categories.

## 5. Model Training

The primary model is an XGBoost classifier.

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

XGBoost was selected because the dataset is structured/tabular and
contains nonlinear relationships and interactions that tree-based
boosting methods can model effectively.

The primary reported results come from held-out data. Results calculated
after fitting the final model on the complete historical dataset are
treated only as in-sample diagnostics and are not used as the main
evidence of generalization.

## 6. Evaluation

### ROC-AUC

ROC-AUC measures ranking performance across classification thresholds.

The primary held-out ROC-AUC is:

**0.7532**

This can be interpreted as the model having approximately a 75.3%
probability of ranking a randomly selected positive case above a
randomly selected negative case.

### Precision-Recall AUC

PR-AUC is particularly useful because the target is imbalanced.

The primary held-out PR-AUC is:

**0.2393**

The positive prevalence is approximately 0.0807, so the PR-AUC is
substantially above the prevalence baseline.

### Accuracy

Accuracy is not emphasized because a majority-class strategy could
achieve high accuracy while identifying very few positive cases.

## 7. Capacity-Based Evaluation

The model is evaluated as a ranking system under limited review
capacity.

For each capacity level:

1.  Generate risk scores for the held-out applications.
2.  Rank applications from highest to lowest predicted risk.
3.  Select the top portion corresponding to the available review
    capacity.
4.  Treat the lowest score inside that selected group as the
    capacity-implied threshold.
5.  Measure precision and recall within the selected queue.

Results:

    Capacity   Precision   Recall
  ---------- ----------- --------
          1%      45.92%    5.66%
          2%      40.74%   10.01%
          5%      33.13%   20.42%
         10%      26.95%   32.39%

At 5% capacity, the queue has approximately 33% positive prevalence
compared with approximately 8% in the overall population.

The threshold is therefore determined by operational capacity. It should
not be interpreted as a universal boundary between safe and risky
applications.

## 8. Calibration

Calibration analysis compares predicted risk values with observed
positive rates.

The model is reasonably calibrated across the main populated probability
ranges. Very high predicted-risk ranges contain fewer observations, so
observed rates in those ranges are less stable.

The Brier scores were approximately:

-   XGBoost: **0.0681**
-   Prevalence baseline: **0.0742**

This corresponds to an improvement of approximately 8.3% relative to the
prevalence-only baseline.

## 9. Explainability

SHAP is used to analyze how features contribute to model predictions.

The project uses:

-   Global beeswarm analysis
-   Local waterfall explanations
-   Dependence analysis
-   Interaction analysis

Important interactions included relationships among external-source
scores, disagreement between those scores, financing ratios, credit
amounts, and selected applicant characteristics.

Some engineered variables are mathematically derived from other model
inputs. Interactions involving those variables should therefore be
interpreted carefully because part of the relationship is structurally
expected.

SHAP describes model behavior and does not establish causality.

## 10. Anomaly Investigation

The `DAYS_EMPLOYED` variable contains the value `365243` in more than
55,000 observations. This value is not a realistic employment duration
and acts as a dataset sentinel/anomaly.

Two approaches were compared:

1.  Retaining the original value.
2.  Replacing the value with missing data and adding a separate anomaly
    indicator.

Performance was almost identical, with the original representation
performing marginally better. The original representation was therefore
retained while documenting the anomaly.

This experiment illustrates the importance of testing data-cleaning
assumptions rather than automatically changing unusual values.

## 11. Robustness Testing

### Repeated Splits

The model was evaluated across random seeds 7, 21, 42, 84, and 123.

Results:

-   Mean ROC-AUC: **0.7552**
-   ROC-AUC standard deviation: **0.0011**
-   Mean PR-AUC: **0.2418**
-   PR-AUC standard deviation: **0.0018**

### Bootstrap Confidence Intervals

A 1,000-sample bootstrap analysis was performed on the primary held-out
predictions.

Approximate 95% confidence intervals:

-   ROC-AUC: **\[0.7461, 0.7601\]**
-   PR-AUC: **\[0.2286, 0.2513\]**

These analyses indicate stability across random samples from the
available dataset. They do not establish future temporal performance.

## 12. Cost-Sensitivity Analysis

A hypothetical cost analysis was used to study the trade-off between
false positives and false negatives at different review capacities.

The cost multipliers are illustrative only and do not represent real
organizational costs.

This analysis demonstrates that the preferred review capacity depends on
the relative operational cost of unnecessary reviews versus missed
high-risk applications.

## 13. Saved Model and Inference Pipeline

After model development, the final XGBoost model is trained on all
available labeled applications and saved to disk together with its
expected feature schema.

Reloading the saved model produced a maximum probability difference of
**0.0** compared with the original in-memory model.

The inference pipeline validates:

-   Required input variables
-   Expected encoded feature count
-   Feature ordering
-   Unexpected encoded columns
-   Missing expected columns
-   Remaining missing values

After validation, the model generates risk scores, ranks applications,
selects the requested review capacity, and exports a review queue.

## 14. Leakage Considerations

Historical features must only use information that would have been
available at the time an application was scored.

The project does not claim that a complete timestamp-level leakage audit
has been performed for every historical aggregate. Such an audit would
be required before production deployment.

Encoding and schema alignment do not prevent temporal leakage; leakage
must be controlled during feature construction.

## 15. Limitations and Future Work

The largest limitation is the absence of true out-of-time validation.
Random train/test splits and bootstrap testing measure stability within
the available historical population but cannot guarantee performance on
future applications.

Future work should include:

-   Train-on-past / test-on-future temporal validation
-   Data and prediction drift monitoring
-   Periodic performance evaluation and retraining
-   Subgroup fairness and bias analysis
-   Formal feature-ablation experiments
-   More robust fitted categorical encoding
-   Automated raw-data ingestion and historical aggregation
-   Explicit timestamp validation for all historical features

The project should therefore be viewed as a complete experimental
risk-prioritization pipeline rather than a production-ready automated
credit-decision system.
