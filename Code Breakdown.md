# 🧠 Fraud Detection - Code Breakdown

This document explains the purpose and flow of each part of the code in the `FraudDetection.ipynb` notebook. The model uses various machine learning algorithms to detect fraudulent financial transactions.

---

## 📦 1. Importing Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score as ras
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
```

Loads libraries for data manipulation, visualization, model building, and evaluation.

## 📥 2. Loading the Dataset
```python
data = pd.read_csv('new_data.csv')
data.head()
```

Loads the transaction dataset from a CSV file.
Displays the first few rows to get an overview.

## 🧾 3. Exploring the Dataset
```python
data.info()
data.describe()
```

.info(): Displays data types and null values.
.describe(): Shows statistics like mean, min, max, and standard deviation.

## 📊 4. Visualizing Transaction Types
```python
sns.countplot(x='type', data=data)
```

Plots the frequency of each transaction type (e.g., CASH_OUT, TRANSFER).

## ⚠️ 5. Fraud Label Distribution
```python
data['isFraud'].value_counts()
```

Shows the count of fraud vs. non-fraud transactions.

## 📉 6. Correlation Heatmap
```python
plt.figure(figsize=(12, 6))
sns.heatmap(data.apply(lambda x: pd.factorize(x)[0]).corr(),
            cmap='BrBG', fmt='.2f', linewidths=2, annot=True)
```

Factorizes categorical data and visualizes the correlation between all features.

## 🛠️ 7. Feature Engineering
```python
type_new = pd.get_dummies(data['type'], drop_first=True)
data_new = pd.concat([data, type_new], axis=1)
```

One-hot encodes the type column and appends it to the original dataset.

## 🧮 8. Defining Features and Target
```python
X = data_new.drop(['isFraud', 'type', 'nameOrig', 'nameDest'], axis=1)
y = data_new['isFraud']
```

Drops unnecessary or non-numeric columns.
Sets X as features and y as the fraud label.

## 🧪 9. Train-Test Split
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

Splits the data into 70% training and 30% testing.

## 🤖 10. Model Training & Evaluation
```python
models = [
    LogisticRegression(max_iter=1000),
    XGBClassifier(),
    RandomForestClassifier(n_estimators=7, criterion='entropy', random_state=7)
]

for model in models:
    model.fit(X_train, y_train)
    print(f'{model} :')
    train_preds = model.predict_proba(X_train)[:, 1]
    print('Training Accuracy : ', ras(y_train, train_preds))
    y_preds = model.predict_proba(X_test)[:, 1]
    print('Validation Accuracy : ', ras(y_test, y_preds))
    print()
```

Trains three models: Logistic Regression, XGBoost, and Random Forest.
Evaluates performance using ROC AUC score.

## 📌 11. Confusion Matrix
```python
cm = ConfusionMatrixDisplay.from_estimator(models[1], X_test, y_test)
cm.plot(cmap='Blues')
plt.show()
```

Displays confusion matrix for the XGBoost model.

## 📈 12. ROC Curve
```python
fpr, tpr, _ = roc_curve(y_test, y_preds)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_preds):.2f}')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
```

Plots the ROC curve using predictions from the last evaluated model (usually Random Forest).
