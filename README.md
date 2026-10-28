# Horse Colic Survival Predictor (ML from scratch)

Predicts whether a horse with colic lives or dies, using logistic regression
written from scratch in NumPy (no scikit-learn).

**Dataset (real, public):** UCI Machine Learning Repository, *Horse Colic*
(McLeish & Cecile, Univ. of Guelph) https://archive.ics.uci.edu/dataset/47/horse+colic
300 horses, 28 columns, many missing values.

## Run
```
pip install -r requirements.txt
python download_data.py
python train.py
streamlit run app.py
```

## Preprocessing steps (all in `preprocess.py`)
| # | Step | Why |
|---|------|-----|
| 1 | Replace `?` with NaN | the file's marker for missing |
| 2 | Drop 1 row with unknown outcome | cannot learn without a label |
| 3 | Drop ID + post-diagnosis columns (hospital number, lesion codes, surgical lesion, cp_data) | ID is noise; the others would leak the answer |
| 4 | Fix `total_protein` units (values > 40 are g/L, divide by 10) | column mixes g/dL and g/L |
| 5 | Recode age (code 9 = young), capillary refill (invalid code 3 -> missing), surgery to 0/1 | inconsistent / invalid codes |
| 6 | Drop columns with > 50% missing (3 columns) | too little information to impute |
| 7 | Clip continuous columns to 1st-99th percentile | outliers (e.g. pulse 184) |
| 8 | Impute: median (numbers), most common value (categories) | fill remaining gaps |
| 9 | One-hot encode unordered categories (pain, mucous membranes, ...) | model needs numbers, labels have no order |
| 10 | Standardise continuous + ordinal columns | gradient descent converges better |

Steps 6-10 learn their numbers from the training set only, so there is no data leakage.

## Model and evaluation
- Logistic regression + gradient descent + L2 regularisation (`model.py`)
- L2 strength chosen by 5-fold cross-validation on the training set only
- Metrics: accuracy, precision, recall, F1, confusion matrix, ROC / AUC
- Over/underfitting: train vs cross-validation accuracy across L2 strengths (`plots/6_regularisation.png`)


# 🐴 Horse Colic Survival Predictor

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://horse-colic-predictor.streamlit.app)
