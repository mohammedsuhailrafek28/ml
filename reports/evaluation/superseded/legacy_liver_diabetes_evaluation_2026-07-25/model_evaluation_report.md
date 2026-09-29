# Persisted model evaluation report

Evaluation used saved production pipelines on recreated stratified 80/20 holdout test sets (`random_state=42`). No model was retrained.

## Liver model evaluation

Selected model: random_forest
Train/test: 466/117

| Metric | Value |
|---|---:|
| Accuracy | 0.7436 |
| Balanced Accuracy | 0.6630 |
| Precision | 0.7978 |
| Recall | 0.8554 |
| F1 | 0.8256 |
| Roc Auc | 0.7697 |
| Specificity | 0.4706 |
| Sensitivity | 0.8554 |
| Mcc | 0.3469 |
| Cohen Kappa | 0.3439 |
| Log Loss | 0.5031 |
| Brier Score | 0.1721 |

Confusion matrix: TN=16, FP=18, FN=12, TP=71

```
              precision    recall  f1-score   support

           0       0.57      0.47      0.52        34
           1       0.80      0.86      0.83        83

    accuracy                           0.74       117
   macro avg       0.68      0.66      0.67       117
weighted avg       0.73      0.74      0.74       117

```

| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.3 | 0.7248 | 0.9518 | 0.8229 |
| 0.4 | 0.7475 | 0.8916 | 0.8132 |
| 0.5 | 0.7912 | 0.8675 | 0.8276 |
| 0.6 | 0.8182 | 0.7590 | 0.7875 |
| 0.7 | 0.9216 | 0.5663 | 0.7015 |

## Diabetes model evaluation

Selected model: svm
Train/test: 614/154

| Metric | Value |
|---|---:|
| Accuracy | 0.7273 |
| Balanced Accuracy | 0.7261 |
| Precision | 0.5909 |
| Recall | 0.7222 |
| F1 | 0.6500 |
| Roc Auc | 0.8139 |
| Specificity | 0.7300 |
| Sensitivity | 0.7222 |
| Mcc | 0.4360 |
| Cohen Kappa | 0.4302 |
| Log Loss | 0.4994 |
| Brier Score | 0.1673 |

Confusion matrix: TN=73, FP=27, FN=15, TP=39

```
              precision    recall  f1-score   support

           0       0.83      0.73      0.78       100
           1       0.59      0.72      0.65        54

    accuracy                           0.73       154
   macro avg       0.71      0.73      0.71       154
weighted avg       0.75      0.73      0.73       154

```

| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.3 | 0.5844 | 0.8333 | 0.6870 |
| 0.4 | 0.6230 | 0.7037 | 0.6609 |
| 0.5 | 0.6596 | 0.5741 | 0.6139 |
| 0.6 | 0.6571 | 0.4259 | 0.5169 |
| 0.7 | 0.7273 | 0.2963 | 0.4211 |

## Limitations
Public datasets are relatively small and metrics are not clinical evidence. Results may not generalize across populations or clinical settings.