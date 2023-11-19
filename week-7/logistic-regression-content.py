# Pima Indian Diabetes

#1.EDA
# 2. Data Preprocessing
# 3. Model & Prediction
# 4. Model Evaluation
# 5. Model Validation : Holdout
# 6. Model Validation : 10-fold Cross Validation
# 7. Prediction for a New Observation

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,roc_auc_score,
                             confusion_matrix,
                             classification_report,
                             RocCurveDisplay)
from sklearn.model_selection import train_test_split,cross_validate

pd.set_option('display.max_columns',None)
pd.set_option('display.expand_frame_repr',False)


# 1. EDA

df = pd.read_csv("/home/msel/Desktop/MIUUL_13/miuul-13/miuul-13/week-7/machine_learning/datasets/diabetes.csv")
df.head()
df.shape

# target analizi
df["Outcome"].value_counts()
sns.countplot(x = "Outcome",data=df)
#plt.show()

# feature analizi
df.describe().T
# boxplot - histogram
df["BloodPressure"].hist(bins = 20)

df.groupby("Outcome").agg({k:'mean' for k in df.columns.difference(["Outcome"])})
df.hist(
    legend=True,layout=(4,3),figsize=(20,10) ,sharex=False,
    sharey=True,bins=20)

# fig, axs = plt.subplots(3,3)
# for index,column in enumerate(df.columns.tolist()):
#     print(axs[index])
#     axs.reshape(-1)[index].boxplot(df[column])
# plt.show()

#################
# 2. Data Preprocessing

df.isnull().sum()

def IQR(df , col , low=0.05 , up = 0.95):
    q1 = df[col].quantile(low)
    q3 = df[col].quantile(up)
    IQR_value = q3-q1
    low_limit = q1 - 1.5*IQR_value
    up_limit = q3 + 1.5*IQR_value
    df.loc[ df[col] < low_limit  , col] = low_limit
    df.loc[df[col] > up_limit , col] = up_limit

for col in df.columns:
    IQR(df,col)

df.describe().T

for col in df.columns:
    df[col] = RobustScaler().fit_transform(df[[col]])

df.head()

# 3. Model & Prediction
y = df["Outcome"]
X = df.drop(["Outcome"],axis = 1)

log_model = LogisticRegression().fit(X,y)
log_model.intercept_
log_model.coef_

y_pred = log_model.predict(X)
y_pred[0:10]
y[0:10]

# Model Evaluation
acc = accuracy_score(y,y_pred)
cm = confusion_matrix(y,y_pred)
sns.heatmap(cm,annot=True,fmt=".0f")

print( classification_report(y,y_pred))
# 1 class ına bakılır
# Accuracy : 0.78
# Precision: 0.74
# Recall: 0.58
# F1-score : 0.65

# ROC AUC
y_prob = log_model.predict_proba(X)[:,1]
roc_auc_score(y,y_prob)
# ROC : 0.83

# BURAYA KADAR AYNI DATAYLA HEM TRAIN HEM TEST
# HEM MODEL ACCURACY HESAPLADIK
#######################
# Model Validation : Holdout

X_train , X_test , y_train , y_test = train_test_split(X,
                                                       y,
                                                       test_size=0.20,random_state=17)
log_model = LogisticRegression().fit(X_train,y_train)

y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:,1]
print(classification_report(y_test,y_pred))

# model görmedigi veriye dokundugunda score lar degismis
#
RocCurveDisplay.from_estimator(log_model, X_test, y_test, name="Model ROC Curve")
roc_auc_score(y_test,y_prob)
# farklı test setleri icin farklı scorelar geliyor
# burada Cross validation yaparız

# Model Validation: 10-Fold Cross Validation

log_model = LogisticRegression().fit(X,y)
cv_results = cross_validate(log_model,X,y,cv=5,scoring=[
    'accuracy','precision','recall','f1','roc_auc'])

cv_results['test_accuracy']
cv_results['test_accuracy'].mean()
{k:cv_results[k].mean() for k in cv_results }

# Prediction for a New Observation
