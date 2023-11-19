import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


experience = np.array([5,7,3,3,2,7,3,10,6,4,8,1,1,9,1])
salary = np.array([600,900,550,500,400,950,540,1200,900,
                   550,1100,460,400,1000,380])


df = pd.DataFrame({"Experience":experience,"Salary":salary})
df.head()

# 1 b = 275 , w = 90
b = 275
w = 90

# 2


df["Predicted_Salary"] = b + w * df["Experience"]


# se: square error
se = (df["Predicted_Salary"] - df["Salary"])**2
#sse: sum of square error
sse = se.sum()
# mse : mean sum of square error
mse = sse / df.shape[0]
print(mse)

# rmse: root mean square error
rmse = np.sqrt(mse)
print(rmse)

# ae :  absolute error
ae = abs(df["Predicted_Salary"] - df["Salary"])
# mae : mean absolute error
mae = ae.sum() / df.shape[0]
print(mae)
#--------
print("--")
print("MSE:",mse)
print("RMSE:",rmse)
print("MAE:",mae)

#

model = LinearRegression()
model.intercept_ = b
model.coef_ = np.array([w])
df["Predicted_Salary_Model"] = model.predict(df[[
    "Experience"]])

from sklearn.metrics import (
    mean_squared_error,mean_absolute_error)

mse_model = mean_squared_error(df[
                                  "Predicted_Salary_Model"] ,
                    df["Salary"])
rmse_model = np.sqrt(mse_model)

mae_model = mean_absolute_error(df[
                                  "Predicted_Salary_Model"] ,
                    df["Salary"])

print("---")
print("MSE_model:",mse_model)
print("RMSE_model:",rmse_model)
print("MAE_model:",mae_model)