# CUSTOMER LIFETIME VALUE (CLTV)

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
#read data
df_ = pd.read_excel("/home/husamdata/Desktop/MIUUL_13/miuul-13/miuul-13/week-3/crmAnalytics/datasets/online_retail_II.xlsx", sheet_name="Year 2009-2010")
df = df_.copy()
# check data
df.head()
df.isnull().sum()

#data cleaning
df = df[~df["Invoice"].str.contains("C",na=False)]
df.describe().T

df = df[df["Quantity"] > 0]
df.dropna(inplace=True)

df.describe().T

# preparing CLTV parameters
df["Total_Price"] = df["Price"]* df["Quantity"]

cltv_c = df.groupby("Customer ID").agg({"Invoice":lambda x: x.nunique(),"Quantity":lambda x: x.sum(),"Total_Price":lambda x: x.sum()})

cltv_c.head()
cltv_c.columns = ["total_transaction" , "total_unit","total_price"]
cltv_c.head()

# 2 Average Order Value

cltv_c["average_order_value"] = cltv_c["total_price"] / cltv_c["total_transaction"]

# 3 Purchase Frequency   total_transaction / total_number_of_customers
len(cltv_c.index)
cltv_c.shape

cltv_c["purchase_frequency"] = cltv_c["total_transaction"] / len(cltv_c.index)

# 4 Repeat Rate & Churn Rate

repeat_rate = cltv_c[cltv_c["total_transaction"] > 1].shape[0] / cltv_c.shape[0]

churn_rate = 1- repeat_rate

#5 Profit Margin (profit_margin = total_price * 0.10)
cltv_c["profit_margin"] = cltv_c["total_price"] * 0.1

# 6 Customer Value( customer_value = average_order_value * purchase_frequency)
cltv_c.head()
cltv_c["customer_value"] = cltv_c["average_order_value"] * cltv_c["purchase_frequency"]

# 7 Customer Lifetime Value (CLTV = customer_value / churn_rate) * profit_margin

cltv_c["cltv"] = (cltv_c["customer_value"] / churn_rate) * cltv_c["profit_margin"]

cltv_c.sort_values(by="cltv",ascending=False).head()

cltv_c["segment"] = pd.qcut(cltv_c["cltv"] , 10 , range(1,11))
cltv_c.head()

cltv_c.groupby("segment").agg(["count","sum","mean"])