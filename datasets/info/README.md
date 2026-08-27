# Dataset sources & attribution

All five datasets are public research/teaching datasets. Verify each upstream
licence before any redistribution. The suite makes **no clinical-validity claim**.

| Disease | Dataset | Source | Licence | Target | Local file |
| --- | --- | --- | --- | --- | --- |
| Liver | Indian Liver Patient Dataset (ILPD) | UCI ML Repository, dataset 225 | CC BY 4.0 | `Selector` (1 disease / 2 control) | `datasets/raw/liver.csv` |
| Heart | Heart Disease — Cleveland | UCI ML Repository, dataset 45 (Janosi, Steinbrunn, Pfisterer, Detrano, 1989) | CC BY 4.0 | `num` (0 / 1–4) | `datasets/raw/heart.csv` |
| Diabetes | Pima Indians Diabetes Database | National Institute of Diabetes and Digestive and Kidney Diseases (via UCI / OpenML 37) | Public domain | `Outcome` (0 / 1) | `datasets/raw/diabetes.csv` |
| Kidney | Chronic Kidney Disease | UCI ML Repository, dataset 336 (Rubini & Eswaran, 2015) | CC BY 4.0 | `class` (ckd / notckd) | `datasets/raw/kidney.csv` |
| Parkinson's | Parkinsons (Oxford voice) | UCI ML Repository, dataset 174 (Little, McSharry, Roberts, Costello, Moroz, 2008) | CC BY 4.0 | `status` (0 / 1), subject id in `name` | `datasets/raw/parkinsons.csv` |

SHA-256 checksums and row/column counts for each local CSV are recorded in
`reports/dataset_manifest.json` and re-verified inside every trainer and in the
disease test suites. `scripts/download_datasets.py` fetches public mirrors where
available and prints manual instructions otherwise.
