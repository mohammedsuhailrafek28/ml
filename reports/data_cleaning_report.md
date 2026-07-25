
## Diabetes

The Pima dataset contains 768 records and 8 numeric predictors. Zero is preserved for `pregnancies`, while zero placeholders in `glucose`, `blood_pressure`, `skin_thickness`, `insulin`, and `bmi` are converted to missing values before the leakage-safe pipeline. Numeric median imputation and scaling are fitted only on training folds. Extreme medical values are retained; no statistical outlier deletion is performed.
