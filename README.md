# Customer Churn Analysis & Prediction

End-to-end analysis and a from-scratch logistic regression model predicting which customers are likely to churn, based on tenure, contract type, tech support, product count, and support-call history.

**Tools:** Python, NumPy, Pandas, Matplotlib

---

## Note on Data

A live churn dataset wasn't reachable in the environment this was built in, so the notebook runs on a **synthetic-but-realistic telecom churn dataset** (3,000 customers, modeled after the structure of the well-known Telco Customer Churn dataset) generated with a deliberate, noisy relationship between features and churn — documented in the notebook's first cell. Every number, table, and chart in the notebook is real, executed output on that data. Pointing it at a real churn CSV only requires changing one `pd.read_csv(...)` line.

## Files

- `churn_analysis.ipynb` — full EDA + from-scratch logistic regression, with real executed outputs and plots
- `requirements.txt` — numpy, pandas, matplotlib

## Key Findings

- **Contract type is the strongest churn signal**: month-to-month customers churn at 4.5x the rate of two-year customers (27.2% vs 6.1%).
- Customers with tech support churn less (13.2% vs 22.8%); tenure is negatively correlated with churn (-0.23).
- A from-scratch logistic regression (NumPy, batch gradient descent — no sklearn) reaches **AUC = 0.74** on held-out data.
- At the default 0.5 threshold the model is too conservative (7.7% recall) — accuracy alone (79.7%) looks fine but barely beats the 78.3% naive baseline, which is misleading given the class imbalance. Re-tuning the decision threshold to 0.29 (maximizing F1) gives a far more usable model: 55% recall at 46% precision. This is reported directly in the notebook rather than only quoting whichever number looks best.
- `tenure_months`, contract length, and `tech_support` are the strongest predictors; payment method contributes only marginally.

## What I'd Add Next

- Swap in a real churn dataset (e.g. the IBM Telco Customer Churn dataset)
- Try a tree-based model (Random Forest / Gradient Boosting) to capture non-linear interactions, e.g. tenure × contract type
- Turn the best-F1 threshold into a cost-sensitive decision rule for an actual retention-offer setting, where missing a churner is usually costlier than a false alarm

## Contact

**Muhammad Ahmed Jawaid**
[LinkedIn](https://www.linkedin.com/in/m-ahmed-jawaid-416662253/) · ahmedjawaid513@outlook.com
