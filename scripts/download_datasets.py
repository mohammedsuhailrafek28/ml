"""Download canonical public teaching datasets into datasets/raw.

Sources are public mirrors of UCI datasets; verify upstream terms before redistribution.
"""
from pathlib import Path
from urllib.request import urlretrieve
from urllib.parse import quote

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'datasets'/'raw'; OUT.mkdir(parents=True,exist_ok=True)
SOURCES={
 'diabetes.csv':'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv',
 'heart.csv':'https://raw.githubusercontent.com/plotly/datasets/master/heart.csv',
 'parkinsons.csv':'https://raw.githubusercontent.com/plotly/datasets/master/parkinsons.data',
 'liver.csv':'https://raw.githubusercontent.com/plotly/datasets/master/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv',
}
for name,url in SOURCES.items():
    target=OUT/name
    if target.exists(): print(f'Exists: {target}'); continue
    try: urlretrieve(url,target); print(f'Downloaded: {name}')
    except Exception as exc: print(f'FAILED {name}: {exc}\nManual source: {url}')
print('Kidney dataset: download the UCI Chronic Kidney Disease CSV from https://archive.ics.uci.edu/ml/machine-learning-databases/00336/chronic_kidney_disease.arff, convert ARFF to CSV, and save as datasets/raw/kidney.csv.')
