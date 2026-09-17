# Customer Churn - Responsible ML Project

## Project overview
This RabTech Academy project frames a small customer-churn prediction problem and documents responsible-use considerations. It compares a simple majority-class non-ML baseline with a Logistic Regression baseline.

## RabTech Academy task
Task: **ML Problem Framing & Responsible Data Card**.

## Objective
Use the provided training records to explore whether a churn label can be predicted from the available customer-level fields, while being explicit about uncertainty, leakage, privacy, fairness, and small-sample limitations.

## Dataset description
`customer-churn-training.csv` contains 12 records and these columns: `customer_id`, `tenure_months`, `support_tickets`, `monthly_spend_inr`, `last_login_days`, `plan_type`, and `churned`. The target is the supplied binary `churned` column. The dataset has 5 churn records and 7 non-churn records. Provenance, permission, timing, churn definition, and sampling method are not provided.

## Problem framing
The prediction may support human review for a possible retention follow-up. It must not automatically determine account access, pricing, service eligibility, or customer treatment. The prediction window is not provided and must be defined before real use.

## Responsible AI considerations
The code excludes `customer_id` from model features, checks the target and obvious name-based leakage candidates, and reports a majority baseline alongside Logistic Regression. Because the dataset is very small and contains no demographic fields, fairness and calibration conclusions are unreliable.

## Project files
- `baseline_churn_model.py`: inspection, preprocessing, split, baselines, metrics, confusion matrix, calibration warning, and error analysis.
- `Responsible_Data_Card.md`: responsible data card following the supplied template.
- `ML_Problem_Framing_Memo.md`: decision and deployment framing memo.
- `Risk_Register.md`: risk register with mitigations and monitoring.
- `customer-churn-training.csv`: supplied training data.
- `responsible-data-card-template.md`: supplied template.
- `requirements.txt`: Python dependencies.
- `.gitignore`: common Python and generated-file exclusions.

## Technologies used
Python, pandas, scikit-learn, and matplotlib.

## Installation
From this project folder, create and activate a virtual environment if desired, then install dependencies:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install pandas scikit-learn matplotlib
```

## How to run the Python baseline
Open this folder in VS Code, select the Python interpreter for the environment, open the integrated terminal, and run:

```bash
python baseline_churn_model.py
```

The program prints dataset checks, class percentages, metrics, confusion matrices, a Brier score warning, and simple error analysis. It also opens a confusion-matrix plot in an interactive Python environment.

## Expected evaluation metrics
The script reports accuracy, precision, recall, F1-score, and a confusion matrix for both the majority-class baseline and Logistic Regression. Logistic Regression also reports a Brier score, but the estimate is not reliable with this small test set.

## Dataset limitations
With only 12 records, a test split contains very few observations. Metrics can change substantially when one record changes. There is no independent validation set, documented sampling method, timestamp, prediction horizon, exact churn definition, demographic coverage, provenance, consent, or license information. No production or fairness claims can be made from this exercise.

## Responsible use
Use the outputs for educational analysis and potential human-supported review only. Do not use them for automatic adverse decisions, profiling, or unrelated customer decisions. Confirm data permission and governance requirements before sharing or deploying anything.

## Future improvements
Collect a larger and representative dataset; document provenance, permission, timestamps, label definition, and action window; use a time-aware and independent validation design; define error costs with stakeholders; evaluate calibration and fairness with adequate governed data; and monitor drift and outcomes before considering deployment.