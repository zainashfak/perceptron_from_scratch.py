"""
Single-Layer Perceptron from Scratch (NumPy only)
Dataset : Pima Indians Diabetes (binary: 1 = diabetic, 0 = non-diabetic)
Source  : Kaggle -> https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
Mirror  : https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv

Install once:  pip install numpy pandas matplotlib
Run:           python perceptron_from_scratch.py
"""

import os
import numpy as np
import pandas as pd                 # used ONLY to read the CSV file
import matplotlib.pyplot as plt     # used ONLY for plotting

# ----------------------------------------------------------------------
# 1. LOAD DATASET
# ----------------------------------------------------------------------
URL = ("https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
       "pima-indians-diabetes.csv")
LOCAL_FILE = "diabetes.csv"   # if you downloaded it from Kaggle, put it next to this script
COLUMNS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
           "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]

if os.path.exists(LOCAL_FILE):
    df = pd.read_csv(LOCAL_FILE)          # Kaggle version has a header row
    df.columns = COLUMNS
else:
    df = pd.read_csv(URL, header=None, names=COLUMNS)   # GitHub mirror has no header

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nClass distribution:\n", df["Outcome"].value_counts())

# ----------------------------------------------------------------------
# 2. PREPROCESSING
# ----------------------------------------------------------------------
# 2a. In these columns a value of 0 is physiologically impossible -> missing
zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[zero_as_missing] = df[zero_as_missing].replace(0, np.nan)
print("\nMissing values before imputation:\n", df.isna().sum())

# 2b. Fill missing values with the column median
df = df.fillna(df.median())

X = df.drop(columns="Outcome").values.astype(float)
y = df["Outcome"].values.astype(int)

# 2c. Stratified 80/20 train-test split (implemented manually with NumPy)
rng = np.random.default_rng(42)
train_idx, test_idx = [], []
for cls in np.unique(y):
    idx = rng.permutation(np.where(y == cls)[0])
    split = int(0.8 * len(idx))
    train_idx.extend(idx[:split])
    test_idx.extend(idx[split:])
train_idx = rng.permutation(train_idx)
test_idx = rng.permutation(test_idx)

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

# 2d. Standardise features (mean/std from TRAIN set only -> no data leakage)
mean, std = X_train.mean(axis=0), X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

print(f"\nTrain size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")


# ----------------------------------------------------------------------
# 3. SINGLE-LAYER PERCEPTRON (from scratch)
# ----------------------------------------------------------------------
class SingleLayerPerceptron:
    def __init__(self, n_features, learning_rate=0.01, n_epochs=100, seed=42):
        self.lr = learning_rate
        self.n_epochs = n_epochs
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, 0.01, size=n_features)   # weights
        self.b = 0.0                                    # bias
        self.errors_per_epoch = []                      # misclassifications
        self.acc_per_epoch = []
        self.best_w, self.best_b, self.best_acc = self.w.copy(), self.b, 0.0

    @staticmethod
    def step(z):
        """Heaviside step activation."""
        return np.where(z >= 0, 1, 0)

    def predict(self, X):
        return self.step(X @ self.w + self.b)

    def fit(self, X, y):
        for epoch in range(self.n_epochs):
            errors = 0
            for xi, target in zip(X, y):
                pred = self.step(np.dot(xi, self.w) + self.b)
                update = self.lr * (target - pred)       # perceptron learning rule
                self.w += update * xi
                self.b += update
                errors += int(update != 0.0)
            self.errors_per_epoch.append(errors)
            acc = np.mean(self.predict(X) == y)
            self.acc_per_epoch.append(acc)
            # "Pocket" trick: data is not perfectly linearly separable, so the
            # plain perceptron keeps oscillating. Keep the best weights seen.
            if acc > self.best_acc:
                self.best_acc, self.best_w, self.best_b = acc, self.w.copy(), self.b
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1:3d}/{self.n_epochs} | "
                      f"misclassified: {errors:3d} | "
                      f"train accuracy: {acc:.4f}")
        self.w, self.b = self.best_w, self.best_b       # restore best weights
        print(f"Best training accuracy (pocket weights): {self.best_acc:.4f}")
        return self


# ----------------------------------------------------------------------
# 4. TRAIN
# ----------------------------------------------------------------------
print("\n--- Training ---")
model = SingleLayerPerceptron(n_features=X_train.shape[1],
                              learning_rate=0.01, n_epochs=100)
model.fit(X_train, y_train)


# ----------------------------------------------------------------------
# 5. EVALUATION METRICS (from scratch)
# ----------------------------------------------------------------------
def confusion_matrix(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tn, fp, fn, tp


def evaluate(y_true, y_pred, name):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    print(f"\n===== {name} RESULTS =====")
    print(f"Confusion Matrix  [[TN FP] [FN TP]]:\n[[{tn:3d} {fp:3d}]\n [{fn:3d} {tp:3d}]]")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    return np.array([[tn, fp], [fn, tp]]), accuracy, precision, recall, f1


evaluate(y_train, model.predict(X_train), "TRAIN")
cm, acc, prec, rec, f1 = evaluate(y_test, model.predict(X_test), "TEST")

# ----------------------------------------------------------------------
# 6. VISUALISE FINAL RESULTS
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

axes[0].plot(model.errors_per_epoch, color="tab:red")
axes[0].set_title("Misclassifications per Epoch")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Errors")

axes[1].plot(model.acc_per_epoch, color="tab:green")
axes[1].set_title("Training Accuracy per Epoch")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")

axes[2].imshow(cm, cmap="Blues")
axes[2].set_title("Test Confusion Matrix")
axes[2].set_xticks([0, 1]); axes[2].set_yticks([0, 1])
axes[2].set_xticklabels(["Pred 0", "Pred 1"])
axes[2].set_yticklabels(["Actual 0", "Actual 1"])
for i in range(2):
    for j in range(2):
        axes[2].text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black",
                     fontsize=14)

plt.tight_layout()
plt.savefig("perceptron_results.png", dpi=150)   # saved next to the script
print("\nPlot saved as perceptron_results.png")
plt.show()                                       # opens a window in VS Code