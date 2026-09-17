# Risk Register

| Risk | Category | Likelihood | Impact | Mitigation | Monitoring |
|---|---|---|---|---|---|
| Data leakage | Technical | Cannot be determined from the files | Offline results may be misleading | Define prediction time and audit feature availability | Review feature timestamps and post-deployment performance |
| Small dataset | Data quality | High | Unstable metrics and poor generalization | Treat as educational; collect more records and validate independently | Track confidence, sample sizes, and outcome coverage |
| Sampling bias | Data quality | Cannot be determined from the files | Results may not represent the customer population | Document sampling and test on representative data | Compare incoming population with evaluation population |
| Class imbalance | Data quality | Present but modest in this CSV | Majority predictions can hide missed churn | Report precision, recall, F1, and the majority baseline | Monitor class rates and class-specific errors |
| Privacy | Governance | Cannot be determined from the files | Customer-linked information may be exposed | Confirm permission, restrict access, and minimize identifiers | Audit access and retention practices |
| False positives | Performance | Possible | Unnecessary retention contact or customer friction | Human review and precision monitoring | Confirm outcomes and review false-positive cases |
| False negatives | Performance | Possible | Missed opportunity for review | Recall monitoring and careful review workflow | Review missed churn cases |
| Feature misuse | Governance | Possible | Incorrect or unfair decisions from a feature | Use only documented, pre-outcome features | Audit feature definitions and usage |
| Model drift | Operations | Cannot be determined from the files | Performance can degrade as behavior changes | Monitor and retrain or pause when appropriate | Track feature distributions and confirmed outcomes |
| Fairness | Responsible AI | Cannot be assessed for protected groups from this CSV | Unequal errors may go undetected | Obtain governed evaluation data and use adequate group sizes | Compare group errors only when statistically and ethically appropriate |
| Incorrect predictions | Performance | Certain to occur in some cases | Customer actions may be inappropriate | Communicate uncertainty and require human review | Monitor error rates and qualitative complaints |
| Misuse of predictions | Governance | Possible | Scores may become automatic decisions outside scope | Document intended use and prohibit automated adverse action | Access controls, audits, and incident review |