import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
import joblib

data = pd.read_csv(
    r"C:\Users\elena\Documents\GitHub\semper-tei_new\scripts\Q_rendition_automation\data\data_cleaned.csv"
)

data.head()
data.describe()

# split into input and output variables

X = data[
    [
        "num_lb",
        "num_quote",
        "num_rs",
        "num_add",
        "num_del",
        "num_hi",
        "num_handShift",
        "num_metamark",
        "num_anchor",
        "text_length",
        "num_ptr",
        "zone_area_px2",
    ]
]

y = data[["rendition"]]

y_raw = y["rendition"]  #
y_cat = y_raw.astype("category")  # convert to categorical
y_codes = y_cat.cat.codes.to_numpy()

# list class names
class_names = list(y_cat.cat.categories)
print("Classes:", class_names)

# split into train, and test sets

X_train, X_test, y_train_codes, y_test_codes = train_test_split(
    X, y_codes, stratify=y_codes, test_size=0.2, random_state=42
)

print(X_train.shape, X_test.shape, y_train_codes.shape, y_test_codes.shape)

# split train set into train and validation sets

X_train, X_val, y_train_codes, y_val_codes = train_test_split(
    X_train, y_train_codes, stratify=y_train_codes, random_state=42
)


print(X_train.shape, X_val.shape, y_train_codes.shape, y_val_codes.shape)


# confusion matrix function
def plot_confusion_matrix(y_true: any, y_pred: any):
    """
    Takes true and predicted labels (arrays) as input. Based on this, calculates and plots a confusion matrix.
    """
    labels = np.unique(y_true)
    fig = plt.figure(figsize=(len(labels), len(labels)))
    ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(
            y_true=y_true, y_pred=y_pred, labels=labels, normalize="all"
        ),
        display_labels=labels,
    ).plot(ax=fig.gca(), cmap="BuPu", xticks_rotation="vertical", include_values=True)
    plt.show()


# random forest

model2 = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=0)
model2.fit(X_train, y_train_codes)

# random forest accuracy on train set
y_train_codes_hat = model2.predict(X_train)

accuracy_score_train = accuracy_score(y_train_codes, y_train_codes_hat)
print(accuracy_score_train)

f1score_train_model2 = f1_score(
    y_true=y_train_codes, y_pred=y_train_codes_hat, average=None
)
model2.classes_

for cls_code, f1 in zip(model2.classes_, f1score_train_model2):
    class_name = class_names[cls_code]
    print(f"Class '{class_name}': F1 = {f1:.4f}")

# random forest accuracy on validation set
y_val_codes_hat = model2.predict(X_val)

accuracy_score_val_model2 = accuracy_score(y_val_codes, y_val_codes_hat)

print(accuracy_score_val_model2)

f1score_val_model2 = f1_score(y_true=y_val_codes, y_pred=y_val_codes_hat, average=None)

model2.classes_

for cls_code, f1 in zip(model2.classes_, f1score_val_model2):
    class_name = class_names[cls_code]
    print(f"Class '{class_name}': F1 = {f1:.4f}")

# roc-auc score
proba_train = model2.predict_proba(X_train)  # shape: (n_train, n_classes)
proba_val = model2.predict_proba(X_val)  # shape: (n_val, n_classes)


print("proba_train shape:", proba_train.shape)
print("Row sums (first 5):", proba_train.sum(axis=1)[:5])  # should be ~1.0

train_auc = roc_auc_score(
    y_train_codes, proba_train, multi_class="ovr", average="weighted"
)

val_auc = roc_auc_score(y_val_codes, proba_val, multi_class="ovr", average="weighted")

print("Train AUC:", train_auc)
print("Validation AUC:", val_auc)

# the scores are very good but are classes balanced?
pd.Series(y_train_codes).value_counts(normalize=True)

# plot learning curves

train_sizes, train_scores, val_scores = learning_curve(
    model2, X, y_codes, cv=5, scoring="accuracy"
)

plt.plot(train_sizes, np.mean(train_scores, axis=1), label="Training")
plt.plot(train_sizes, np.mean(val_scores, axis=1), label="Validation")
plt.legend()
plt.show()

# check out-of-the-bag error
print(model2.oob_score_)

max_estimators = 100

oob_errors = []

for i in range(1, max_estimators + 1):
    model2.set_params(n_estimators=i, oob_score=True)
    model2.fit(X_train, y_train_codes)

    pred = model2.predict(X_val)
    val_error = 1 - accuracy_score(y_val_codes, pred)

    oob_error = 1 - model2.oob_score_
    oob_errors.append((i, oob_error))

xs, ys = zip(*oob_errors)
plt.plot(xs, ys, label="RandomForest OOB error")
plt.xlim(1, max_estimators)
plt.xlabel("n_estimators")
plt.ylabel("OOB error rate")
plt.legend(loc="upper right")
plt.show()

# random forest with 10 estimators and balanced class weights

model3 = RandomForestClassifier(
    n_estimators=10, oob_score=True, random_state=0, class_weight="balanced"
)
model3.fit(X_train, y_train_codes)

# accuracy on train set

y_train_hat_model3 = model3.predict(X_train)

accuracy_score_train_model3 = accuracy_score(y_train_codes, y_train_hat_model3)
print(accuracy_score_train_model3)
f1score_train_model3 = f1_score(
    y_true=y_train_codes, y_pred=y_train_hat_model3, average=None
)

model3.classes_

for cls_code, f1 in zip(model3.classes_, f1score_train_model3):
    class_name = class_names[cls_code]
    print(f"Class '{class_name}': F1 = {f1:.4f}")

# accuracy on validation set

y_val_hat_model3 = model3.predict(X_val)

accuracy_score_val_model3 = accuracy_score(y_val_codes, y_val_hat_model3)

print(accuracy_score_val_model3)

f1score_val_model3 = f1_score(y_true=y_val_codes, y_pred=y_val_hat_model3, average=None)

model3.classes_

for cls_code, f1 in zip(model3.classes_, f1score_val_model3):
    class_name = class_names[cls_code]
    print(f"Class '{class_name}': F1 = {f1:.4f}")

# plot confusion matrix
plot_confusion_matrix(y_val_codes, y_val_hat_model3)

# plot learning curve model 3

train_sizes, train_scores, val_scores = learning_curve(
    model3, X, y_codes, cv=5, scoring="accuracy"
)

plt.plot(train_sizes, np.mean(train_scores, axis=1), label="Training")
plt.plot(train_sizes, np.mean(val_scores, axis=1), label="Validation")
plt.legend()
plt.show()

# compare OOB errors

print(model2.oob_score_)
print(model3.oob_score_)

# plot oob error for model 3
max_estimators = 100

oob_errors = []

for i in range(1, max_estimators + 1):
    model3.set_params(n_estimators=i, oob_score=True)
    model3.fit(X_train, y_train_codes)

    pred = model3.predict(X_val)
    val_error = 1 - accuracy_score(y_val_codes, pred)

    oob_error = 1 - model3.oob_score_
    oob_errors.append((i, oob_error))

xs, ys = zip(*oob_errors)
plt.plot(xs, ys, label="RandomForest OOB error")
plt.xlim(1, max_estimators)
plt.xlabel("n_estimators")
plt.ylabel("OOB error rate")
plt.legend(loc="upper right")
plt.show()

# try gridsearch for different n_stimators and max_samples
params = {
    "n_estimators": [10, 20, 100],
    "max_samples": [None, 0.9, 0.75, 0.5],
    "max_features": ["sqrt", None],
    "class_weight": ["balanced"],
}

search = GridSearchCV(
    RandomForestClassifier(bootstrap=True, random_state=0),
    params,
    cv=3,
    scoring="f1_macro",
    n_jobs=-1,
)

search.fit(X_train, y_train_codes)
print(search.best_params_)
print(search.best_score_)

# train a model with those params

model4 = RandomForestClassifier(
    n_estimators=100,
    oob_score=True,
    random_state=0,
    class_weight="balanced",
    max_samples=0.75,
    max_features=None,
)
model4.fit(X_train, y_train_codes)

# compare OOB scores:

print(model2.oob_score_)
print(model3.oob_score_)
print(model4.oob_score_)

# accuracy on train set

y_train_hat_model4 = model4.predict(X_train)

accuracy_score_train_model4 = accuracy_score(y_train_codes, y_train_hat_model4)
print(accuracy_score_train_model4)
f1score_train_model4 = f1_score(
    y_true=y_train_codes, y_pred=y_train_hat_model4, average=None
)

model4.classes_

for cls_code, f1 in zip(model4.classes_, f1score_train_model4):
    class_name = class_names[cls_code]
    print(f"Class '{class_name}': F1 = {f1:.4f}")

# accuracy on validation set

y_val_hat_model4 = model4.predict(X_val)

accuracy_score_val_model4 = accuracy_score(y_val_codes, y_val_hat_model4)
print(accuracy_score_val_model4)
f1score_val_model4 = f1_score(y_true=y_val_codes, y_pred=y_val_hat_model4, average=None)

model4.classes_

for cls_code, f1 in zip(model4.classes_, f1score_val_model4):
    class_name = class_names[cls_code]
    print(f"Class '{class_name}': F1 = {f1:.4f}")

# plot confusion matrix
plot_confusion_matrix(y_val_codes, y_val_hat_model4)

# plot learning curve model 3

train_sizes, train_scores, val_scores = learning_curve(
    model4, X, y_codes, cv=5, scoring="accuracy"
)

plt.plot(train_sizes, np.mean(train_scores, axis=1), label="Training")
plt.plot(train_sizes, np.mean(val_scores, axis=1), label="Validation")
plt.legend()
plt.show()

joblib.dump(model4, "rendition_rf_model.pkl")
np.save("rendition_class_names.npy", np.array(class_names))
