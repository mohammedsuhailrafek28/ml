# Final results summary

   Disease  Dataset Rows  Input Features    Selected Model  Accuracy  Precision   Recall       F1  ROC-AUC  CV F1 Mean  CV F1 Std              Evaluation Method       Leakage Controls             Main Limitation
     liver           583              11     random_forest  0.743590   0.797753 0.855422 0.825581 0.769667    0.769526        NaN Stratified holdout + 5-fold CV Pipeline preprocessing  Public dataset limitations
     heart           303              28     random_forest  0.885246   0.818182 0.964286 0.885246 0.952381    0.794993        NaN Stratified holdout + 5-fold CV Pipeline preprocessing  Public dataset limitations
  diabetes           768               8               svm  0.727273   0.590909 0.722222 0.650000 0.813889    0.668706   0.044800 Stratified holdout + 5-fold CV Pipeline preprocessing  Public dataset limitations
    kidney           400              38     random_forest  1.000000   1.000000 1.000000 1.000000 1.000000    0.992531   0.009985 Stratified holdout + 5-fold CV Pipeline preprocessing  Public dataset limitations
parkinsons           195              22 gradient_boosting  0.744186   0.738095 1.000000 0.849315 0.723118    0.892036   0.060016 GroupShuffleSplit + GroupKFold       Subject grouping Small grouped voice dataset