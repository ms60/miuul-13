# 1 Data Preperation

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# read data
df_ = pd.read_excel("/home/husamdata/Desktop/MIUUL_13/miuul-13/miuul-13/week-3/crmAnalytics/datasets/online_retail_II.xlsx",sheet_name="Year 2010-2011")
df = df_.copy()
df.describe().T
df.head()
df.isnull().sum()

# data pre processing

df.dropna(inplace=True)
df.describe().T

df = df[~df["Invoice"].str.contains("C",na=False)]
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]



replace_with_thresholds(df , "Quantity")
replace_with_thresholds(df , "Price")

df.describe().T

df["TotalPrice"] = df["Quantity"] * df["Price"]

today_date = dt.datetime(2011,12,11) # analysis date

## prepare lifetime data
# recency : analysis date - last purchase date
# T : age of customer , weekly , analysis date - first purchase date
# frequency : >2 transaction
# monetary : total purchase / total transaction

cltv_df = df.groupby("Customer ID").agg({"InvoiceDate":[lambda x : (x.max() - x.min()).days , lambda x : (today_date - x.min()).days  ]})
