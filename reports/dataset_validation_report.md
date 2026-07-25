# Dataset validation report

## Liver

- Source: Indian Liver Patient Dataset (ILPD), local verified mirror copied from `Documents/DS-project/datasets/raw/ilpd_raw.csv`.
- File: `datasets/raw/liver.csv`
- Shape: 583 data rows x 11 columns (header plus 583 records).
- Columns: Age, Gender, TB, DB, Alkphos, Sgpt, Sgot, TP, ALB, A/G Ratio, Selector.
- Target: Selector; verified ILPD mapping 1 = liver disease, 2 = no liver disease.
- Missing representation: blank values; the A/G Ratio field contains the known missing entry.
- Status: schema validated; ready for training once Python dependencies import successfully.

Other modules remain unavailable until their source datasets are acquired and schema-validated.

## Heart

- Source: UCI Cleveland Heart Disease (`processed.cleveland.data`).
- URL: https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data
- Local file: `datasets/raw/heart.csv`.
- Shape: 303 rows x 14 columns including target.
- Target mapping: `0` = no detected disease; `1–4` = disease presence, mapped to binary `0/1`.
- Status: schema validated and trained.

## Diabetes

- Dataset: Pima Indians Diabetes Database.
- Source: public UCI/Kaggle teaching-data mirror (the downloaded file is the canonical 768-row, 9-column schema).
- Local file: `datasets/raw/diabetes.csv`.
- Shape: 768 rows x 9 columns.
- Target: `target`; `1` = diabetes-positive outcome, `0` = negative outcome.
- Status: schema validated and trained.
