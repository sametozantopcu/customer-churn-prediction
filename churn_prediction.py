"""
Customer Churn Prediction Pipeline
==================================
An end-to-end machine learning pipeline developed to predict customer churn
risk. Logistic Regression and KNN models were trained, and their
performances along with the effects of class imbalance were examined.

Main Libraries Used:
- pandas, scikit-learn, matplotlib

Setup and Usage:
1. Install dependencies:
   $ pip install -r requirements.txt

2. Run the script:
   $ python churn_prediction.py
"""

# 1. DATA LOADING AND EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
# Loading the customer dataset, checking its dimensions, and examining
# the distribution of the target variable, customer churn.

import pandas as pd

df = pd.read_csv("musteri_veri_seti.csv")

print("First 5 Rows:\n", df.head())
print(df.info())
print(f"\nDataset Shape: {df.shape}")
print(f"\nTarget Variable Distribution: {df["churn"].value_counts()}")

# 2. DATA PREPROCESSING AND MISSING VALUE HANDLING
# ==============================================================================
# Detecting missing (null) values in the dataset and filling them using
# statistical methods appropriate to the variable type (median for
# numerical variables, mode for categorical variables).

print("\nMissing Values:\n", df.isnull().sum())
eksik_sayisal_sutunlar = ["yas", "destek_talebi_sayisi"]

for sutun in eksik_sayisal_sutunlar:
    medydan_degeri = df[sutun].median()
    df[sutun] = df[sutun].fillna(medydan_degeri)

eksik_kategorik_sutunlar = ["sehir", "uyelik_tipi"]

for sutun in eksik_kategorik_sutunlar:
    mod_degeri = df[sutun].mode()[0]
    df[sutun] = df[sutun].fillna(mod_degeri)

print(f"\nAfter filling missing values: \n{df.head()}\n")
print(f"After filling missing values: \n{df.isnull().sum()}\n")

sayisal_sutunlar = ["yas", "gelir", "abonelik_suresi", "destek_talebi_sayisi"]
aykiri_deger_maskesi = pd.Series(False, index=df.index)

for sutun in sayisal_sutunlar:
    q1 = df[sutun].quantile(0.25)
    q3 = df[sutun].quantile(0.75)
    iqr = q3 - q1
    alt_sinir = q1 - 1.5 * iqr
    ust_sinir = q3 + 1.5 * iqr
    sutun_maskesi = ((df[sutun] < alt_sinir) | (df[sutun] > ust_sinir))
    aykiri_deger_maskesi = aykiri_deger_maskesi | sutun_maskesi
    print(f"{sutun} - outlier count: {sutun_maskesi.sum()}")

print(f"\nTotal outlier rows: {aykiri_deger_maskesi.sum()}")

df = df.loc[~aykiri_deger_maskesi].reset_index(drop=True)

print(f"Updated dataset after removing outliers: {df.shape}")

# 3. FEATURE ENGINEERING
# ==============================================================================
# Creating new, domain-knowledge-driven derived features to improve the
# model's predictive power and its ability to capture interactions
# between variables.

df["destek_orani"] = df["destek_talebi_sayisi"] / (df["abonelik_suresi"] + 1)  
df["gelir_yas_orani"] = df["gelir"] / df["yas"]
df["destek_abonelik_farki"] = df["destek_talebi_sayisi"] - (df["abonelik_suresi"] / 10)

korelasyonlar = df.corr(numeric_only=True)["churn"].sort_values(ascending=False)
print(f"\nCorrelations: \n{korelasyonlar}")

secilen_oznitelikler = korelasyonlar[abs(korelasyonlar) > 0.50].index.tolist()
secilen_oznitelikler.remove("churn")

print(secilen_oznitelikler)
df = df.drop(columns=["gelir_yas_orani"])
print("\nFirst 5 rows after adding new features:")
print(df.head())

# 4. CATEGORICAL ENCODING
# ==============================================================================
# Converting categorical features (sehir, uyelik_tipi) into numerical
# vectors using One-Hot Encoding so that the algorithms can process them,
# and dropping the first category to avoid the dummy variable trap.

y = df["churn"].values
X = df.drop(columns= ["churn"])
X = pd.get_dummies(X, columns= ["sehir", "uyelik_tipi"], drop_first= True, dtype= int)

print("\nFirst 5 rows after categorical encoding")
print(X.head())

# 5. TRAIN / VALIDATION / TEST SPLIT
# ==============================================================================
# Splitting the data into 70% Train, 15% Validation, and 15% Test to
# prevent overfitting and measure generalization performance. 'stratify'
# is used to preserve the class distribution of the target variable.

from sklearn.model_selection import train_test_split 

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size= 0.3, random_state= 42, stratify= y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size= 0.5, random_state= 42, stratify=y_temp )

print(f"\nX_train: {X_train.shape}")
print(f"X_val: {X_val.shape}")
print(f"X_test: {X_test.shape}")

# 6. FEATURE SCALING
# ==============================================================================
# Applying StandardScaler to prevent bias caused by numerical variables
# being on different scales and to improve model sensitivity. To prevent
# Data Leakage, the scaler is fit only on the Train data.

from sklearn.preprocessing import StandardScaler

sayisal_sutunlar = ["yas", "gelir", "abonelik_suresi", "destek_talebi_sayisi", "destek_orani", "destek_abonelik_farki"]

standard_scaler = StandardScaler()

X_train_standard = X_train.copy()
X_val_standard = X_val.copy()
X_test_standard = X_test.copy()

X_train_standard[sayisal_sutunlar] = (
    standard_scaler.fit_transform(
        X_train[sayisal_sutunlar]
    )
)

X_val_standard[sayisal_sutunlar] = (
    standard_scaler.transform(
        X_val[sayisal_sutunlar]
    )
)

X_test_standard[sayisal_sutunlar] = (
    standard_scaler.transform(
        X_test[sayisal_sutunlar]
    )
)

print(f"\nX_train_standard: \n{X_train_standard.head()}")

# 7. MODEL TRAINING AND ARCHITECTURE SELECTION
# ==============================================================================
# Training a linear-based (Logistic Regression) and a distance-based
# (K-Nearest Neighbors) classification model on the scaled training data
# in order to compare different mathematical approaches on this dataset.

#Logistic Regression Model
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(C = 1, max_iter = 100, random_state = 42)
log_reg.fit(X_train_standard, y_train)

val_acc = log_reg.score(X_val_standard, y_val)
print(f"\nLogistic Regression Validation Accuracy: {val_acc}")

#KNN Model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

k_accuracy = []
k_values = []
for k in range(2, 15):
    knn_k = KNeighborsClassifier(n_neighbors=k)
    knn_k.fit(X_train_standard, y_train)
    y_pred = knn_k.predict(X_val_standard)
    k_accuracy.append(accuracy_score(y_val, y_pred))
    k_values.append(k)

plt.plot(k_values, k_accuracy, marker = "o")
plt.xlabel("k_values")
plt.ylabel("k_accuracy")
plt.title("Validation Accuracy for Different k Values")
plt.show()

# Final KNN model with the best k value (k=5)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_standard, y_train)

y_pred_val = knn.predict(X_val_standard)

accuracy = accuracy_score(y_val, y_pred_val)
print(f"\nKNN Validation Accuracy: {accuracy}")

conf_matrix = confusion_matrix(y_val, y_pred_val)
print(f"\nConfusion Matrix Validation: \n{conf_matrix}")

# 8. MODEL SELECTION AND VALIDATION EVALUATION
# ==============================================================================
# Comparing the Validation set accuracy scores in order to check for
# overfitting and determine which architecture achieves the highest
# generalization performance.

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score

print("\n================FINAL TEST SET EVALUATION ================")
print(f"""
MODEL COMPARISON AND SELECTION:
Based on the validation tests conducted;
- Logistic Regression model: %{val_acc*100:.1f} accuracy
- KNN (k=5) model: %{accuracy*100:.1f} accuracy
were obtained. Logistic Regression, due to its linearity assumption, captured
the patterns in this dataset better and showed higher generalization
performance. Therefore, the Logistic Regression model was selected for
the final evaluation on the test set.
\n""")

y_pred_log_test = log_reg.predict(X_test_standard)
log_test_acc = accuracy_score(y_test, y_pred_log_test)
print(f"Logistic Regression Test Accuracy: {log_test_acc:.4f}")

conf_matrix_test = confusion_matrix(y_test, y_pred_log_test)
print("--- Logistic Regression Confusion Matrix (Test) ---")
print(conf_matrix_test)

cls_report = classification_report(y_test, y_pred_log_test)
print("\n--- Logistic Regression Classification Report (Test) ---")
print(cls_report)

recall_churn1 = recall_score(y_test, y_pred_log_test, pos_label=1)
churn_dagilimi = pd.Series(y).value_counts()
kalan_sayisi = churn_dagilimi[0]
ayrilan_sayisi = churn_dagilimi[1]

print("\n================ INTERPRETATION OF RESULTS ================")
print(f"""
In this project, Logistic Regression and KNN models were trained and compared.
During validation, Logistic Regression (%{val_acc*100:.1f}) outperformed KNN (%{accuracy*100:.1f}),
and was therefore selected as the final model.

However, examining the Classification Report of the selected Logistic Regression
model on the test set revealed an important finding:
While the model predicted "retained" customers (Churn=0) almost perfectly, it
struggled to identify "churned" customers (Churn=1), with a Recall of only
%{recall_churn1*100:.0f}.

This is not caused by a weakness in the algorithm, but by the CLASS IMBALANCE
IN THE DATASET ({kalan_sayisi} Retained vs. {ayrilan_sayisi} Churned). In minimizing the loss
function, the model focused on learning the majority class (0) and overlooked
the minority class (1). In future work, this imbalance could be addressed by
collecting more 'Churn=1' (churned customer) samples, or by using synthetic
data generation techniques to help the model better learn and capture
(Recall) the minority class.
""")