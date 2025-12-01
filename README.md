# Loan Approval Prediction

Predicting loan approval decisions using machine learning on the Kaggle Playground Series S4E10 dataset.

## Overview

This project tackles binary classification to predict whether loan applications get approved or rejected. The main challenge is dealing with imbalanced data - about 86% of loans are approved, making it hard to catch the rejected ones.

**Dataset**: 58,637 loan applications  
**Goal**: Maximize recall for rejected loans while maintaining good overall accuracy

## Quick Start

```bash
pip install pandas numpy scikit-learn lightgbm catboost imbalanced-learn
jupyter notebook Project.ipynb
```

## Dataset

**11 Original Features:**
- Person info: age, income, employment length, home ownership
- Loan details: amount, interest rate, intent, grade
- Credit history: length, previous defaults

**7 Engineered Features:**
- loan_to_income_ratio
- risk_score (combines interest rate, amount, income)
- total_debt_burden
- income_per_age
- employment_stability
- And more...

## Preprocessing

1. Removed outliers (age > 85, income > $1M, employment > 50 years)
2. Handled 7,586 missing values in employment length
3. Scaled continuous features using StandardScaler
4. Encoded categorical variables (ordinal for loan_grade, label for others)

## Models Tested

| Model | Recall (Rejected) | Precision (Rejected) | Accuracy |
|-------|------------------|---------------------|----------|
| **LightGBM (best)** | **0.80** | **0.81** | **0.94** |
| CatBoost | 0.79 | 0.81 | 0.94 |
| Random Forest | 0.74 | 0.78 | 0.93 |
| Logistic Regression | 0.69 | 0.44 | 0.83 |

## Best Model: LightGBM

```python
LGBMClassifier(
    n_estimators=750,
    learning_rate=0.03,
    num_leaves=50,
    max_depth=10,
    class_weight={0: 4, 1: 1}  # 4x weight for rejected class
)
```

Achieved 80% recall on rejected loans (catches 8 out of 10 risky applications) with 94% overall accuracy.

## Key Findings

**Most Important Features:**
1. person_income (17%)
2. loan_int_rate (10%)
3. loan_to_income_ratio (8%) - engineered
4. total_debt_burden (7%) - engineered
5. loan_intent (7%)

**Class Imbalance Strategy:**  
Class weighting (4:1) worked better than SMOTE. It gave us the best balance between catching bad loans and not rejecting too many good ones.

**Feature Engineering:**  
The engineered features didn't boost performance much but ranked high in importance. They're useful for understanding what drives rejections.

## Results

5-fold cross-validation:
- Mean accuracy: 0.943 ± 0.003
- Mean F1-score: 0.967 ± 0.002

The model correctly identifies 80% of loans that should be rejected while maintaining 97% recall for approved loans.

## Files

- `Project.ipynb` - Main notebook with all experiments
- `train.csv` - Training data

## What I Learned

- Class weighting often beats SMOTE for tree-based models
- Sometimes simple features work just as well as complex engineered ones
- High feature importance doesn't always mean better predictions
- Finding the right precision-recall tradeoff matters more than raw accuracy

## Future Ideas

- Try ensemble methods (combine LightGBM + CatBoost)
- Test threshold adjustment for different business scenarios
- Add external data like credit bureau scores
- Experiment with cost-sensitive learning

## Requirements

```
pandas
numpy
scikit-learn
lightgbm
catboost
imbalanced-learn
seaborn
matplotlib
```

---

Built for data mining course project, Fall 2025