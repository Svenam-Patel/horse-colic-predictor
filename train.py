"""
train.py   --   Predicting survival of horses with colic (UCI Horse Colic dataset)

Run order:  python download_data.py -> python train.py -> streamlit run app.py

Steps: audit data -> visualise -> preprocess -> choose regularisation with
       cross-validation -> train -> evaluate -> check over/underfitting -> save.
"""
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from model import (LogisticRegression, auc, confusion, k_fold_indices, roc_curve,
                   scores, split_indices)
from preprocess import FEATURES, Preprocessor, build_dataset, load_raw

os.makedirs("plots", exist_ok=True)
GREEN, RED, BLUE = "#4C9F70", "#D9534F", "#5B8DB8"

# ------------------------------------------------------------ 1. data audit
raw = load_raw()
print(f"Raw table: {raw.shape[0]} horses x {raw.shape[1]} columns")
print("Rows with unknown outcome (dropped):", int(raw["outcome"].isna().sum()))
print("Duplicate rows:", int(raw.duplicated().sum()))
df, y = build_dataset(raw)
print(f"Rows kept: {len(df)} | Lived: {y.sum()} ({y.mean():.1%}) | Died/euthanised: {(1 - y).sum()}")
print("Majority-class baseline accuracy:", round(max(y.mean(), 1 - y.mean()), 3), "\n")

missing_pct = (df[FEATURES].isna().mean() * 100).sort_values()
print("Missing values (%):\n", missing_pct.round(1).to_string(), "\n")

# ------------------------------------------------------------ 2. visualisation
plt.figure(figsize=(7, 6))
plt.barh(missing_pct.index, missing_pct.values, color=[RED if v > 50 else BLUE for v in missing_pct])
plt.axvline(50, color="k", ls="--"); plt.xlabel("% missing")
plt.title("Missing values per column (red = dropped)")
plt.tight_layout(); plt.savefig("plots/1_missing_values.png", dpi=130); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(9, 4))
ax[0].bar(["Lived", "Died / euthanised"], [y.sum(), (1 - y).sum()], color=[GREEN, RED])
ax[0].set_title("Class balance")
ax[1].hist(raw["total_protein"].dropna(), bins=30, color=RED, alpha=0.7, label="raw")
ax[1].hist(df["total_protein"].dropna(), bins=30, color=GREEN, alpha=0.7, label="after unit fix")
ax[1].set_title("Total protein: mixed units fixed"); ax[1].set_xlabel("g/dL"); ax[1].legend()
plt.tight_layout(); plt.savefig("plots/2_balance_and_unit_fix.png", dpi=130); plt.close()

fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
for a, col in zip(ax, ["pulse", "packed_cell_volume", "resp_rate", "rectal_temp"]):
    a.hist(df.loc[y == 1, col].dropna(), bins=20, alpha=0.6, color=GREEN, label="Lived")
    a.hist(df.loc[y == 0, col].dropna(), bins=20, alpha=0.6, color=RED, label="Died")
    a.set_title(col); a.legend()
plt.tight_layout(); plt.savefig("plots/3_distributions.png", dpi=130); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for a, col, title in zip(ax, ["pain", "abdominal_distension"], ["Pain level (1 none ... 5 severe)", "Abdominal distension (1 none ... 4 severe)"]):
    rate = df.assign(lived=y).groupby(col)["lived"].mean()
    a.bar(rate.index.astype(int).astype(str), rate.values, color=BLUE)
    a.set_ylim(0, 1); a.set_title(title); a.set_ylabel("Survival rate")
plt.tight_layout(); plt.savefig("plots/4_survival_by_category.png", dpi=130); plt.close()

num_cols = ["rectal_temp", "pulse", "resp_rate", "packed_cell_volume", "total_protein",
            "abdominal_distension", "nasogastric_tube", "surgery", "age", "capillary_refill"]
corr = df[num_cols].assign(lived=y).corr()
plt.figure(figsize=(7, 6))
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.xticks(range(len(corr)), corr.columns, rotation=45, ha="right"); plt.yticks(range(len(corr)), corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        plt.text(j, i, f"{corr.values[i, j]:.1f}", ha="center", va="center", fontsize=7)
plt.title("Correlation heatmap"); plt.colorbar()
plt.tight_layout(); plt.savefig("plots/5_correlation.png", dpi=130); plt.close()

# ------------------------------------------------------------ 3. split + preprocess
train_idx, test_idx = split_indices(len(df), test_size=0.2, seed=42)
train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

pre = Preprocessor().fit(train_df)                  # learns from TRAIN only
print("Dropped (>50% missing):", pre.dropped)
X_train, X_test = pre.transform(train_df), pre.transform(test_df)
print(f"Train: {X_train.shape} | Test: {X_test.shape} | features after one-hot encoding: {X_train.shape[1]}\n")

# ------------------------------------------------------------ 4. pick regularisation by cross-validation
# (the test set is NOT touched here)
folds = []
for tr, va in k_fold_indices(len(train_df), k=5):
    p = Preprocessor().fit(train_df.iloc[tr])
    folds.append((p.transform(train_df.iloc[tr]), y_train[tr], p.transform(train_df.iloc[va]), y_train[va]))

lams = [0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
tr_acc, cv_acc = [], []
print("=== OVER / UNDERFITTING: effect of L2 regularisation (5-fold CV) ===")
print(f"{'L2':>6s} {'train acc':>10s} {'CV acc':>8s}")
for lam in lams:
    a_tr, a_va = [], []
    for Xa, ya, Xb, yb in folds:
        m = LogisticRegression(lr=0.1, epochs=1000, l2=lam).fit(Xa, ya)
        a_tr.append(scores(ya, m.predict(Xa))["accuracy"])
        a_va.append(scores(yb, m.predict(Xb))["accuracy"])
    tr_acc.append(np.mean(a_tr)); cv_acc.append(np.mean(a_va))
    print(f"{lam:6.3f} {tr_acc[-1]:10.3f} {cv_acc[-1]:8.3f}")
best_lam = lams[int(np.argmax(cv_acc))]
print("Best L2 by cross-validation:", best_lam, "\n")

plt.figure(figsize=(6, 4))
xs = range(len(lams))
plt.plot(xs, tr_acc, "o-", label="Train accuracy")
plt.plot(xs, cv_acc, "o-", label="Cross-validation accuracy")
plt.xticks(xs, [str(l) for l in lams]); plt.xlabel("L2 strength (0 = none)")
plt.ylabel("Accuracy"); plt.title("Over / underfitting vs regularisation"); plt.legend()
plt.tight_layout(); plt.savefig("plots/6_regularisation.png", dpi=130); plt.close()

# ------------------------------------------------------------ 5. final model + evaluation
clf = LogisticRegression(lr=0.1, epochs=1000, l2=best_lam).fit(X_train, y_train, X_test, y_test)
plain = LogisticRegression(lr=0.1, epochs=1000, l2=0.0).fit(X_train, y_train)

plt.figure(figsize=(6, 4))
plt.plot(clf.train_loss, label="Train loss"); plt.plot(clf.val_loss, label="Test loss")
plt.xlabel("Epoch"); plt.ylabel("Log loss"); plt.title("Learning curve"); plt.legend()
plt.tight_layout(); plt.savefig("plots/7_loss_curve.png", dpi=130); plt.close()

proba, pred = clf.predict_proba(X_test), clf.predict(X_test)
tp, tn, fp, fn = confusion(y_test, pred)
test_scores = scores(y_test, pred)
fpr, tpr = roc_curve(y_test, proba)
test_auc = auc(fpr, tpr)

print("=== TEST RESULTS (final model) ===")
for k, v in test_scores.items():
    print(f"{k:10s}: {v:.3f}")
print(f"{'AUC':10s}: {test_auc:.3f}")
print(f"Confusion  : TP={tp} TN={tn} FP={fp} FN={fn}")
print(f"Train acc {scores(y_train, clf.predict(X_train))['accuracy']:.3f} | "
      f"No-regularisation test acc {scores(y_test, plain.predict(X_test))['accuracy']:.3f} | "
      f"Baseline {max(y_test.mean(), 1 - y_test.mean()):.3f}\n")

plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, label=f"Our model (AUC = {test_auc:.3f})"); plt.plot([0, 1], [0, 1], "k--", label="Random guess")
plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("ROC curve"); plt.legend()
plt.tight_layout(); plt.savefig("plots/8_roc.png", dpi=130); plt.close()

plt.figure(figsize=(4.8, 4))
plt.imshow(np.array([[tn, fp], [fn, tp]]), cmap="Blues")
for i, row in enumerate([[tn, fp], [fn, tp]]):
    for j, v in enumerate(row):
        plt.text(j, i, v, ha="center", va="center", fontsize=14)
plt.xticks([0, 1], ["Pred Died", "Pred Lived"]); plt.yticks([0, 1], ["Actual Died", "Actual Lived"])
plt.title("Confusion matrix"); plt.tight_layout(); plt.savefig("plots/9_confusion.png", dpi=130); plt.close()

order = np.argsort(-np.abs(clf.w))[:12]
plt.figure(figsize=(7, 5))
plt.barh([pre.feature_names[i] for i in order][::-1], clf.w[order][::-1],
         color=[GREEN if clf.w[i] > 0 else RED for i in order][::-1])
plt.title("Top 12 feature weights (green = helps survival, red = hurts)")
plt.tight_layout(); plt.savefig("plots/10_feature_weights.png", dpi=130); plt.close()
print("Top features:", ", ".join(f"{pre.feature_names[i]} ({clf.w[i]:+.2f})" for i in order[:8]))

# ------------------------------------------------------------ 6. save for the app
with open("model.pkl", "wb") as f:
    pickle.dump({"pre": pre, "w": clf.w, "b": clf.b}, f)
print("\nSaved model.pkl and plots/")
