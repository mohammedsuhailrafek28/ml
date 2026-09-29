export const assessmentCopy = {
  liver: {
    title: 'Explore a liver-dataset model output',
    description: 'This educational module uses the UCI Indian Liver Patient Dataset (ILPD, 570 records after de-duplication) and a persisted scikit-learn logistic regression pipeline chosen by cross-validation. It produces an uncalibrated dataset-associated score and is not a diagnosis.',
  },
  diabetes: {
    title: 'Explore a diabetes-dataset model output',
    description: 'This educational module uses the Pima Indians Diabetes Database (768 records of Pima women aged 21+) and a persisted scikit-learn logistic regression pipeline chosen by cross-validation on a held-out development split. Impossible zero readings for glucose, blood pressure, skin fold, insulin and BMI are treated as not measured and imputed inside the pipeline. It produces an uncalibrated dataset-associated score and is not a diagnosis.',
  },
  heart: {
    title: 'Explore a heart-dataset model output',
    description: 'This educational module uses the UCI Cleveland Heart Disease database (303 patient records) and a persisted scikit-learn logistic regression pipeline chosen by cross-validation on a held-out development split. It produces an uncalibrated dataset-associated score and is not a diagnosis.',
  },
  kidney: {
    title: 'Explore a kidney-dataset model output',
    description: 'This educational module uses the UCI Chronic Kidney Disease dataset (400 records, 2015). A fold-safe feature-selection experiment reduced the 24 recorded fields to a compact 14-field set with no loss of cross-validated performance, and a persisted scikit-learn logistic regression pipeline was chosen on a held-out development split. Every field is optional; unavailable values are imputed by the pipeline. It produces an uncalibrated dataset-associated score and is not a diagnosis.',
    note: 'This dataset is close to separable on legitimate clinical markers, so held-out scores are very high. That reflects this small curated dataset, not clinical-grade CKD detection.',
  },
  parkinsons: {
    title: "Explore a Parkinson's-associated voice pattern",
    description: "This educational module uses the UCI Parkinson's voice dataset (195 sustained-vowel recordings from 32 people, 2008) and a persisted scikit-learn logistic regression pipeline. Training, tuning and evaluation are subject-aware: the same person never appears in both training and test data.",
    note: 'This module accepts pre-computed voice biomarkers produced by voice-analysis software such as Praat. Medical AI Suite does not record or process raw audio.',
    evidence: 'Experimental evidence: the 195 recordings include only 8 controls. The subject-disjoint holdout contains 7 people and produced ROC-AUC 0.586 with specificity 0.0 at the fixed 0.5 threshold. This limited result is not clinical validation.',
  },
} as const;
