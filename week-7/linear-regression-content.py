# Linear Regression

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#pd.set_option()

from sklearn.linear_model import  LinearRegression
from sklearn.metrics import (
    mean_squared_error,mean_absolute_error)
from sklearn.model_selection import (
    train_test_split,cross_val_score)

df = pd.read_csv("/home/msel/Desktop/MIUUL_13/miuul-13/miuul-13/week-7/machine_learning/datasets/advertising.csv")
df.head()

##

X = df[["TV"]]
y = df[["sales"]]

#model
reg_model = LinearRegression().fit(X,y)
# y_ = b + w*x
reg_model.intercept_[0] # b
reg_model.coef_[0][0] # w

# Prediction
# 150 birimlik TV harcaması olsa sales ?
(reg_model.intercept_[0] +
 reg_model.coef_[0][0]*150)
# 14.16
g = sns.regplot(x=X,y=y,scatter_kws={
    'color':'b','s':9},ci=False,
                color='r')
plt.show()

# Prediction Accuracy
# Tahmin Basarisi
y_pred = reg_model.predict(X)
#MSE
mean_squared_error(y,y_pred) #10
y.mean() # 14.02
y.std() # 5.21
# hata cok gibi

#RMSE
np.sqrt(mean_squared_error(y,y_pred))
#3.24

#MAE
mean_absolute_error(y,y_pred) # 2.54

# R^2
# bagimsiz degiskenlerin , bagimli
# degiskeni aciklama yüzdesi
reg_model.score(X,y) # 0.61


# Multiple Linear Regression
# divide train-test data

X_train , X_test , y_train , y_test =\
    train_test_split(X,y,
                     test_size=0.2,
                     random_state=1)


#model
multi_reg_model = LinearRegression()
multi_reg_model.fit(X_train,y_train)

#sabit # b - bias
multi_reg_model.intercept_
#coefficients # w- weights
multi_reg_model.coef_

#Tahmin basarisi
y_pred = multi_reg_model.predict(X_test)

#RMSE
np.sqrt(mean_squared_error(y_test,y_pred))

#R^2
multi_reg_model.score(X_test,y_test)

# 10 Katlı CV RMSE

np.mean(np.sqrt( -1 *
                 cross_val_score(
                     multi_reg_model,
                     X,y,cv=10,
                     scoring='neg_mean_squared_error') ))


#