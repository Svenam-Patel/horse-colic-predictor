"""
preprocess.py
All data cleaning / preprocessing for the Horse Colic data, written by hand.

Two stages:
  1. clean()        -> fixes errors in the raw table (same rules for any data)
  2. Preprocessor   -> LEARNS numbers from the training set (medians, modes,
                       clip limits, means...) and then applies them to any data.
                       Learning only from training data avoids "data leakage".
"""
import numpy as np
import pandas as pd

COLUMNS = ["surgery", "age", "hospital_number", "rectal_temp", "pulse", "resp_rate",
           "temp_extremities", "peripheral_pulse", "mucous_membranes", "capillary_refill",
           "pain", "peristalsis", "abdominal_distension", "nasogastric_tube",
           "nasogastric_reflux", "nasogastric_reflux_ph", "rectal_exam_feces", "abdomen",
           "packed_cell_volume", "total_protein", "abdomo_appearance", "abdomo_total_protein",
           "outcome", "surgical_lesion", "lesion_1", "lesion_2", "lesion_3", "cp_data"]

# Not used as features: an ID, and facts only known AFTER diagnosis/surgery (would leak the answer)
NOT_FEATURES = ["hospital_number", "surgical_lesion", "lesion_1", "lesion_2", "lesion_3", "cp_data"]

CONTINUOUS = ["rectal_temp", "pulse", "resp_rate", "packed_cell_volume", "total_protein",
              "nasogastric_reflux_ph", "abdomo_total_protein"]       # real numbers
ORDINAL = ["abdominal_distension", "nasogastric_tube"]               # 1 < 2 < 3 (< 4): ordered levels
BINARY = ["surgery", "age", "capillary_refill"]                      # two options -> 0 / 1
NOMINAL = ["temp_extremities", "peripheral_pulse", "mucous_membranes", "pain", "peristalsis",
           "nasogastric_reflux", "rectal_exam_feces", "abdomen", "abdomo_appearance"]  # unordered labels
FEATURES = CONTINUOUS + ORDINAL + BINARY + NOMINAL

MAX_MISSING = 0.5   # drop a column if more than 50% of its values are missing


def load_raw(path="data/horse-colic.csv"):
    """Read the file. The dataset marks missing values with '?'."""
    return pd.read_csv(path, header=None, names=COLUMNS, na_values="?",
                       dtype={"lesion_1": str, "lesion_2": str, "lesion_3": str})


def clean(df):
    """Step-by-step fixes for errors found in the raw data."""
    df = df.copy()
    # Fix 1: total_protein mixes two units. Most values are g/dL (6-9) but ~1/3 are in g/L (46-89).
    df["total_protein"] = np.where(df["total_protein"] > 40, df["total_protein"] / 10, df["total_protein"])
    # Fix 2: recode two-option columns to 0/1 (anything not listed becomes missing)
    df["surgery"] = df["surgery"].map({1: 1, 2: 0})                 # 1 = had surgery
    df["age"] = df["age"].map({1: 0, 2: 1, 9: 1})                   # 1 = young; code 9 is a typo-variant of 2
    df["capillary_refill"] = df["capillary_refill"].map({1: 0, 2: 1})  # 1 = slow refill; invalid code 3 -> missing
    return df


def build_dataset(raw):
    """Drop rows with unknown outcome and make the target: 1 = horse lived, 0 = died / euthanised."""
    df = raw.dropna(subset=["outcome"]).reset_index(drop=True)
    y = (df["outcome"] == 1).astype(int).values
    return clean(df), y


class Preprocessor:
    def fit(self, df):
        # 1. drop columns that are mostly empty
        missing = df[FEATURES].isna().mean()
        self.dropped = missing[missing > MAX_MISSING].index.tolist()
        keep = lambda cols: [c for c in cols if c not in self.dropped]
        self.cont, self.ordinal = keep(CONTINUOUS), keep(ORDINAL)
        self.binary, self.nominal = keep(BINARY), keep(NOMINAL)
        self.used = self.cont + self.ordinal + self.binary + self.nominal

        # 2. outlier limits (1st and 99th percentile) for continuous columns
        self.lo = df[self.cont].quantile(0.01)
        self.hi = df[self.cont].quantile(0.99)

        # 3. fill-in values: median for numbers, most common value (mode) for categories
        self.fill = {c: float(df[c].median()) for c in self.cont + self.ordinal}
        self.fill.update({c: float(df[c].mode()[0]) for c in self.binary + self.nominal})

        # 4. list of categories for one-hot encoding
        self.cats = {c: sorted(df[c].dropna().unique()) for c in self.nominal}

        # 5. mean / std for scaling (computed after filling and clipping)
        scaled = self._fill_and_clip(df)[self.cont + self.ordinal]
        self.mean, self.std = scaled.mean(), scaled.std().replace(0, 1)

        self.feature_names = (self.cont + self.ordinal + self.binary +
                              [f"{c}={int(v)}" for c in self.nominal for v in self.cats[c]])
        return self

    def _fill_and_clip(self, df):
        d = df[self.used].copy()
        d = d.fillna(self.fill)                                              # missing values
        d[self.cont] = d[self.cont].clip(self.lo, self.hi, axis=1)           # outliers
        return d

    def transform(self, df):
        d = self._fill_and_clip(df)
        d[self.cont + self.ordinal] = (d[self.cont + self.ordinal] - self.mean) / self.std   # scaling
        numeric = d[self.cont + self.ordinal + self.binary].values.astype(float)
        onehot = [(d[c] == v).astype(float).values for c in self.nominal for v in self.cats[c]]  # encoding
        return np.column_stack([numeric] + onehot)
