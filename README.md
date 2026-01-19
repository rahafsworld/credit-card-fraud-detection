# Credit Card Fraud Detection

Learning how to handle imbalanced datasets using the Kaggle credit card fraud dataset.

## The Problem

This dataset has 284,807 transactions but only 492 frauds (0.17%). Regular accuracy doesn't work here - a model that just predicts "not fraud" every time gets 99.83% accuracy but catches zero frauds.

## What I Tried

### Iteration 1-2: Random Forest with Class Weights
- Set class weights to 1:500 to force the model to care about fraud
- Got 92% precision and 75% recall
- Pretty good but wanted to try other methods

### Iteration 3: LightGBM with is_unbalance=True
- Used LightGBM's built-in balancing
- Got 80% recall but only 5% precision
- Way too many false alarms

### Iteration 4: Hybrid Resampling (SMOTE + Undersampling)
- SMOTE to generate synthetic fraud samples (0.17% → 10%)
- Then undersample normal transactions to get 50:50 balance
- Put it in a pipeline with LightGBM
- This worked way better

### Iteration 5-10: Threshold Tuning
At first the model at default 0.5 threshold gave 41% precision and 84% recall. I tested different thresholds:

| Threshold | Precision | Recall |
|-----------|-----------|--------|
| 0.50      | 41%       | 84%    |
| 0.70      | 58%       | 83%    |
| 0.82      | 70%       | 82%    |
| 0.90      | 78%       | 81%    |

I went with 0.82 because catching more fraud (82% vs 81%) is worth the precision drop for banking.

## Final Results

- **AUPRC: 0.73** (baseline is 0.0017 for random guessing)
- **Catches 116 out of 142 frauds** in the test set
- **70% precision** means 3 out of 4 alerts are real fraud

## What I Learned

1. Accuracy is useless for imbalanced data
2. SMOTE alone creates too much synthetic noise, so combining it with undersampling works better
3. The "best" threshold depends on whether you care more about precision or recall
4. F₂-score is useful when missing fraud costs more than false alarms

## Tech Stack

Python, scikit-learn, LightGBM, imbalanced-learn, pandas, matplotlib

## Files

- `FraudDetection.ipynb` - all the code and analysis

## Dataset

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud - 284K transactions, V1-V28 are PCA features, Time and Amount are the only non-transformed features.

## What I'd Try Next

- Anomaly detection methods like Isolation Forest
- Ensemble different models together
- Feature engineering on Time and Amount
- Test on real-time data if I had access
- Combine multivariable calculus concepts to find the best balance

## Acknowledgments

- Dataset provided by ULB Machine Learning Group via Kaggle

---

I'm still doing my Bachelors in Data Science (Hons.), and hopefully I can improve and get into Master's and PhD. studies. This is just a learning project and not an attempt for publishable work. Feedback welcome!
