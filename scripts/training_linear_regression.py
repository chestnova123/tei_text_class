import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.metrics import roc_auc_score


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

X.head()
y.head()

# split into train, and test sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# split train set into train and validation sets

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, stratify=y_train, random_state=42
)


print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

# first attempt: logistic regression

model1 = Pipeline(
    [("std", StandardScaler()), ("clf", LogisticRegression(multi_class="multinomial"))]
)

model1.fit(X_train, y_train)

# accuracy on train set

y_train_hat = model1.predict(X_train)
accuracy_score_train = accuracy_score(y_train, y_train_hat)
f1score_train = f1_score(y_true=y_train, y_pred=y_train_hat, average=None)

model1.classes_

for cls, f1 in zip(model1.classes_, f1score_train):
    print(f"Class {cls}: F1 = {f1:.4f}")

print(accuracy_score_train)

# accuracy on validation set

y_val_hat = model1.predict(X_val)
accuracy_score_val = accuracy_score(y_val, y_val_hat)

print(accuracy_score_val)

f1score_val = f1_score(y_true=y_val, y_pred=y_val_hat, average=None)

for cls, f1 in zip(model1.classes_, f1score_val):
    print(f"Class {cls}: F1 = {f1:.4f}")

# plot confusion matrix on train set


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


plot_confusion_matrix(y_train, y_train_hat)
plot_confusion_matrix(y_val, y_val_hat)
