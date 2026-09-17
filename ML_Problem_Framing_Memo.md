# ML Problem Framing Memo

## 1. Decision
The decision is whether a customer-churn prediction should support human review for a possible customer-retention follow-up. The model should inform review, not decide the action automatically.

## 2. Prediction target
The model predicts the CSV's binary `churned` label: 1 for churn and 0 for non-churn, according to the values supplied. The available files do not define what operational event or time period those labels represent.

## 3. Unit of observation
One row represents one record identified by a `customer_id`. The files do not provide enough information to determine whether each row is a unique customer snapshot, the date of the snapshot, or the exact observation period.

## 4. Action window
In a real workflow, the prediction should be made at a defined point before the churn outcome, using only information available at that point. The dataset does not provide an action window, timestamps, or a prediction horizon. Any such timing is therefore an assumption that must be specified and validated before deployment.

## 5. Non-ML baseline
The majority-class baseline always predicts the most common class in the training data. It is useful because it provides a simple reference point: Logistic Regression should be compared with this baseline rather than treated as useful merely because it produces predictions.

## 6. Why ML may be useful
Logistic Regression may combine several available numeric and categorical signals into a consistent estimate for review. With this dataset, it is only an educational baseline; the data does not support a claim that ML improves retention outcomes or generalizes to other customers.

## 7. Costs of errors
A false positive predicts churn when the customer does not churn. It could lead to an unnecessary or poorly targeted retention contact and an unpleasant customer experience.

A false negative fails to identify a customer who churns. A potentially useful opportunity for human review may be missed. Monetary costs are not provided and are not assumed here.

## 8. Human review
Predictions should support, not automatically determine, customer-retention actions. A reviewer should consider the prediction as one uncertain signal, check that the data is current and appropriate, and follow applicable customer-service and privacy requirements.

## 9. Monitoring
After deployment, monitor prediction performance, false positives, and false negatives when confirmed outcomes become available. Also monitor data drift, changes in customer behavior, feature availability, and group-level error differences where enough appropriately governed data exists. `plan_type` error comparisons in this 12-record dataset are descriptive only.

## 10. Rollback conditions
Stop using the model if data leakage is discovered, required features are missing or redefined, performance is materially worse than the majority baseline, error patterns create unacceptable customer impact, monitoring cannot be performed, or the intended use changes without a new review. Reassess before resuming use.

## 11. Limitations
The dataset has 12 records, 5 churn labels, and 7 non-churn labels. A split leaves very few test records, so accuracy, precision, recall, F1-score, confusion-matrix counts, Brier score, and plan-level comparisons are unstable. The data provides no provenance, permission, timestamps, churn definition, prediction horizon, demographics, independent validation set, or documented sampling method. These limitations prevent reliable claims about production performance, fairness, calibration, or generalization.