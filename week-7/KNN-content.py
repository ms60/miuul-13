# KNN

# 1.EDA
# 2. Data preprocessing - Feature Engineering
# 3. Modeling & prediction
# 4. Model Evaluation
# 5. Hyperparameter Optimization
# 6. Final Model

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.preprocessing import RobustScaler,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,roc_auc_score,
                             confusion_matrix,
                             classification_report,
                             RocCurveDisplay)
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.neighbors import KNeighborsClassifier

pd.set_option('display.max_columns',None)
pd.set_option('display.expand_frame_repr',False)

# 1. EDA
df = pd.read_csv("/home/msel/Desktop/MIUUL_13/miuul-13/miuul-13/week-7/machine_learning/datasets/diabetes.csv")
df.head()
df.shape

# 2. Data PreProcessing
y = df["Outcome"]
X = df.drop(["Outcome"],axis = 1)
X_scaled = StandardScaler().fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)
X.head()

# 3. Modeling & Prediction

knn_model = KNeighborsClassifier().fit(X,y)
random_user = X.sample(1,random_state=45)
knn_model.predict(random_user)

# 4. Model Evaluation
y_pred = knn_model.predict(X)
y_prob = knn_model.predict_proba(X)[:,1]

print( classification_report(y,y_pred) )
# acc 0.83
# f1 0.74
roc_auc_score(y,y_prob)
# roc 0.90

# Model Validation

cv_results = cross_validate(knn_model,X,y,cv=5,scoring=[
    'accuracy','f1','roc_auc'])

{k:cv_results[k].mean() for k in cv_results}

# Başarı skorları nasıl artırılabilir?

# 1 veri boyutu artırılabilir
# 2 veri ön işleme
# 3 feature engineering new features
# 4 hiperparametre optimizasyonu
knn_model.get_params()


# Hyperparameter Optimization

knn_model = KNeighborsClassifier()
knn_model.get_params()
knn_params = {"n_neighbors":range(2,50)}

from sklearn.model_selection import GridSearchCV

knn_gs_best = GridSearchCV(knn_model,knn_params,cv=5,
                         n_jobs=-1,
             verbose=1).fit(X,y)
# hyper parametre icin de Cross Validation kullanıyoruz

knn_gs_best.best_params_

# Final Model

knn_final = knn_model.set_params(
    **knn_gs_best.best_params_).fit(X,y)
cv_results = cross_validate(knn_final,X,y,cv=5,scoring=[
    "accuracy","f1","roc_auc"])

{k:cv_results[k].mean() for k in cv_results}