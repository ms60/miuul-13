import pandas as pd
import numpy as np

churn = np.array([1,1,1,1,1,1,0,0,0,0])
churn_prob = np.array([0.7,0.8,0.65,0.9,0.45,0.5,0.55,
                       0.35,0.4,0.25])

df = pd.DataFrame({"Churn":churn,"Churn_Prob":churn_prob})
df
#1
df["Chrun_Predicted"] = (df["Churn_Prob"] > 0.5).astype(int)
df

TP = df[(df["Churn"] == 1) & (df["Chrun_Predicted"]==1)
]["Churn"].shape[0]

FP = df[(df["Churn"] == 0) & (df["Chrun_Predicted"]==1)
]["Churn"].shape[0]

FN = df[(df["Churn"] == 1) & (df["Chrun_Predicted"]==0)
]["Churn"].shape[0]

TN = df[(df["Churn"] == 0) & (df["Chrun_Predicted"]==0)
]["Churn"].shape[0]

df_confusion_matrix = pd.DataFrame({
    "Chrun_Predicted_1":[TP,FP],
    "Chrun_Predicted_0":[FN,TN],
})

df_confusion_matrix.index = ["Chrun_Real_1","Churn_Real_0"]

df_confusion_matrix

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP )
recall = TP / (TP + FN)
f1_score = 2 * (precision*recall)/(precision+recall)
print({"accuracy":accuracy,
       "precision":precision,
       "recall":recall,
       "f1_score":f1_score})

##
import seaborn as sns
from sklearn.metrics import (accuracy_score,roc_auc_score,
                             confusion_matrix,
                             classification_report
                            )
acc = accuracy_score(df["Churn"],df["Chrun_Predicted"])
cm = confusion_matrix(df["Churn"],df["Chrun_Predicted"])
sns.heatmap(cm,annot=True,fmt=".0f")

print( classification_report(df["Churn"],df["Chrun_Predicted"]))
##

print("------------------------------")
#2
df2 = pd.DataFrame({"Fraud_Predicted_1":[5,90],
                    "Non_Fraud_Predicted_0":[5,900]})
df2.index = ["Fraud_1","Non_Fraud_0"]
df2

TP2 = 5
FP2 = 90
FN2 = 5
TN2 = 900

accuracy2 =( TP2 + TN2) / (TP2 + TN2 + FN2 + FP2)
precision2 = TP2 / (TP2 + FP2)
recall2 = TP2 / (TP2 + FN2)
f1_score2 = 2 * (precision2*recall2)/(precision2+recall2)

print({"accuracy2":accuracy2,
       "precision2":precision2,
       "recall2":recall2,
       "f1_score2":f1_score2})

# yorum
# precision,recall ve f1 score gibi metrikler
# modelin aslında iyi bir model olmadığını ve
# accuracy nin tek başına değerlendirilmemesi
# gerektiğini gösteriyor