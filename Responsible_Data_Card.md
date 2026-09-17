# Responsible Data Card

## Dataset purpose
This dataset can support an exploratory customer-churn prediction exercise. A prediction could help a human reviewer decide which available records may merit review for a possible retention conversation. The data alone does not establish that a customer will churn or that any particular intervention will work.

The model must not make automatic decisions about a customer's access, price, service eligibility, account status, or treatment. It must not be used to infer protected characteristics or to make decisions about people beyond the stated churn-support use.

## Provenance and permission
The available files provide the CSV filename, column values, and the accompanying template. They do not provide the data creator, collection process, source system, consent information, licensing terms, retention rules, or permission details. These facts cannot be determined from the available data.

## Population and representation
The CSV contains 12 records. The available `plan_type` values are Basic, Pro, and Standard. No demographic, geographic, accessibility, or other protected-group fields are provided, so represented or missing demographic groups cannot be identified from the available data.

The sampling method and relationship of these rows to a wider customer population are not provided. Therefore, sampling bias and generalizability cannot be assessed. There are 5 churn records (41.7%) and 7 non-churn records (58.3%). This is a very small dataset, so percentages, subgroup comparisons, and model performance estimates are unstable.

## Features and target

| Feature name | Meaning | Used for prediction? | Potential leakage or proxy concerns |
|---|---|---:|---|
| `customer_id` | Identifier shown in the CSV, such as C001 | No | It is an identifier, not customer behavior. It could cause memorization or spurious relationships and should not be used as a feature. |
| `tenure_months` | Numeric tenure value in months | Yes | Its timing and definition are provided only by the column name. Confirm it is known before prediction and is not calculated using future information. |
| `support_tickets` | Numeric count of support tickets | Yes | Confirm the count is available at prediction time and does not include future tickets or post-churn activity. |
| `monthly_spend_inr` | Numeric monthly-spend value in INR as named by the column | Yes | It may reflect commercial or socioeconomic differences. Its timing and collection process are not provided. |
| `last_login_days` | Numeric days value since the last login, as named by the column | Yes | Confirm the value is measured before the prediction point and is not updated after the outcome. |
| `plan_type` | Categorical plan label: Basic, Pro, or Standard | Yes | Plan membership may act as a proxy for unobserved differences. Small group counts make comparisons unreliable. |
| `churned` | Binary target where 1 is churn and 0 is non-churn in this CSV | No | It is the outcome to predict. The exact business definition and observation period are not provided. |

The target column is `churned`. The script predicts the provided 0/1 label. `customer_id` is excluded because an identifier can let a model memorize rows without representing a reproducible churn signal.

Possible leakage cannot be fully determined because collection timing is not documented. The baseline excludes the target and identifier and prints a name-based leakage review, but a domain owner must confirm that all remaining features were available before the target outcome. No sensitive demographic fields are present, and sensitive proxies cannot be ruled out from the available feature meanings.

## Quality checks
- Missing values: none are present in the supplied 12 data rows.
- Duplicate rows: no duplicate rows are present.
- Data types: `customer_id` and `plan_type` are text; the other displayed fields are numeric, including the binary target.
- Unusual values: the observed ranges are small and non-negative, but valid business ranges and outlier rules are not provided. Outlier appropriateness cannot be determined from the available data.
- Class balance: 5 churn (41.7%) and 7 non-churn (58.3%).
- Train/test separation: the script uses a stratified random split with `random_state=42`; this is an evaluation split, not evidence of a time-based production separation.
- Other limitations: the dataset does not provide timestamps, a prediction horizon, a churn definition, collection provenance, or an independent validation set.

## Risks and safeguards

| Risk | Type | Potential impact | Safeguard |
|---|---|---|---|
| Bias from unobserved sampling | Bias | Results may not represent the wider customer population. | Document the sampling process before deployment and validate on broader, representative data. |
| Very small sample | Data quality | Metrics and subgroup rates can change substantially with one record. | Treat results as an educational baseline; collect more data and use a separate validation set. |
| Privacy | Privacy | Customer-linked records may expose information or enable unwanted profiling. | Confirm permission, minimize access, protect identifiers, and do not publish sensitive raw data without authorization. |
| Data leakage | Technical | Future information could make offline results falsely strong. | Define the prediction time and audit every feature's availability before that time. |
| Misuse | Governance | Predictions could be used for automatic or unrelated customer decisions. | Restrict use to decision support with documented human review. |
| False positives | Performance | A non-churning customer may receive an unnecessary retention contact or experience. | Review predictions, measure precision, and avoid automatic adverse action. |
| False negatives | Performance | A potentially churning customer may not receive an opportunity for review. | Measure recall, provide appropriate human processes, and do not treat the score as certainty. |
| Fairness | Responsible AI | Error rates may differ across groups that are not represented or are too small to compare. | Collect appropriate governance-approved evaluation data and compare groups only with sufficient sample sizes. |
| Model drift | Operations | Customer behavior or data definitions may change, reducing reliability. | Monitor feature distributions and outcomes after deployment and retrain or pause when needed. |
| Incorrect predictions | Performance | Any individual prediction may be wrong, affecting customer interactions. | Communicate uncertainty and require human review. |

## Intended evaluation
The majority-class classifier is the non-ML baseline. Logistic Regression with one-hot encoding for `plan_type` is the ML baseline. The script reports accuracy, precision, recall, F1-score, and a confusion matrix on a stratified test split.

Logistic Regression probabilities are used to print a Brier score, but calibration is unreliable with only a few test records. A larger independent validation set would be needed for meaningful calibration assessment. Fairness analysis is limited to descriptive error counts by `plan_type`; no protected-group fields are available, and group-level comparisons are unreliable with this sample size. Error analysis prints misclassified test rows and compares errors by `plan_type` when available.