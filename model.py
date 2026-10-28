"""
model.py
Written from scratch with NumPy only:
  - Logistic Regression trained with gradient descent (+ optional L2 regularisation)
  - evaluation metrics (accuracy, precision, recall, F1, ROC, AUC)
  - helpers to split data and make cross-validation folds
"""
import numpy as np


# ---------------------------------------------------------------- data splitting
def split_indices(n, test_size=0.2, seed=42):
    idx = np.random.default_rng(seed).permutation(n)
    cut = int(n * (1 - test_size))
    return idx[:cut], idx[cut:]


def k_fold_indices(n, k=5, seed=0):
    """Yield (train_idx, validation_idx) for k-fold cross-validation."""
    idx = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(idx, k)
    for i in range(k):
        yield np.concatenate([folds[j] for j in range(k) if j != i]), folds[i]


# ---------------------------------------------------------------- the model
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class LogisticRegression:
    def __init__(self, lr=0.1, epochs=1000, l2=0.0):
        self.lr = lr          # learning rate (step size)
        self.epochs = epochs  # number of gradient descent steps
        self.l2 = l2          # regularisation strength (0 = none)

    def loss(self, X, y):
        """Binary cross-entropy (log loss)."""
        p = np.clip(sigmoid(X @ self.w + self.b), 1e-12, 1 - 1e-12)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    def fit(self, X, y, X_val=None, y_val=None):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        self.train_loss, self.val_loss = [], []
        for _ in range(self.epochs):
            p = sigmoid(X @ self.w + self.b)             # 1. predict
            error = p - y                                # 2. how wrong are we?
            grad_w = X.T @ error / n + self.l2 * self.w  # 3. gradients (+ penalty on big weights)
            grad_b = error.mean()
            self.w -= self.lr * grad_w                   # 4. step downhill
            self.b -= self.lr * grad_b
            self.train_loss.append(self.loss(X, y))
            if X_val is not None:
                self.val_loss.append(self.loss(X_val, y_val))
        return self

    def predict_proba(self, X):
        return sigmoid(X @ self.w + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# ---------------------------------------------------------------- metrics
def confusion(y, pred):
    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    return tp, tn, fp, fn


def scores(y, pred):
    tp, tn, fp, fn = confusion(y, pred)
    acc = (tp + tn) / len(y)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def roc_curve(y, proba):
    order = np.argsort(-proba)           # highest probability first
    y = y[order]
    tpr = np.concatenate([[0], np.cumsum(y) / y.sum()])
    fpr = np.concatenate([[0], np.cumsum(1 - y) / (1 - y).sum()])
    return fpr, tpr


def auc(fpr, tpr):
    """Area under ROC curve (trapezoid rule)."""
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2))
