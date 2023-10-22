import pandas as pd
from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)
pd.set_option("display.width",500)
pd.set_option("display.expand_frame_repr",False)
pd.set_option("display.float_format",lambda x: '%.5f' % x)

df = pd.read_csv("/home/msel/Desktop/MIUUL_13/miuul-13/miuul-13/week-4/measurement_problems/datasets/course_reviews.csv")
df.head()

## egitim verisi
# Puan: 4.8
# toplam Puan : 4611

# rating dagilimi
df.shape
df["Rating"].value_counts()
100.0 * df["Rating"].value_counts() / df["Rating"].shape[0]


df["Questions Asked"].value_counts()

df.groupby("Questions Asked").agg({"Questions Asked":["mean","count"] , "Rating":["mean","count"]})

####

df["Rating"].mean()

# sadece bir puan ortalaması almak yerine
# başka şeyler de yapmak lazım
# ne yaparsak trendi ortalamaya dahil ederiz

###################################
# Time-Based Weighted Average
#Puan zamanlarına göre ağırlıklı ortalama

#change Timestamp type to date

df["Timestamp"] = pd.to_datetime(df["Timestamp"])
from datetime import timedelta

#analysis_date = df["Timestamp"].max() + timedelta(days=5)
analysis_date = pd.to_datetime("2021-02-10 0:0:0")

df["days"] = (analysis_date - df["Timestamp"]).dt.days

df.loc[df["days"] <= 30 ,"Rating"].mean()
df.loc[(df["days"] > 30) & (df["days"] <=90) , "Rating" ].mean()
df.loc[(df["days"] > 90) & (df["days"] <=180) , "Rating" ].mean()
df.loc[df["days"] > 180 ,"Rating"].mean()

# bunlara agırlık vererek meanleri hesaplayalım

df.loc[df["days"] <= 30 ,"Rating"].mean() * 0.28 \
+df.loc[(df["days"] > 30) & (df["days"] <=90) , "Rating" ].mean() * 0.26 \
+df.loc[(df["days"] > 90) & (df["days"] <=180) , "Rating" ].mean() * 0.24 \
+df.loc[df["days"] > 180 ,"Rating"].mean() * 0.22

# virgülden sonraki basamaklar çok önemli ve hassas

def time_based_weighted_average():
    pass


###################################
# User-Based Weighted Average
# her kullanıcının oyu aynı mı kabul edilmeli
# bazı kullanıcılar %10 unu izlemiş
# bununla >%75 izleyenin oyu aynı mı

df["Rating"].mean()

df.groupby("Progress").agg({"Rating":"mean"})
# %1 izleyen dusuk puan vermis

df.loc[df["Progress"] <= 10 ,"Rating"].mean() * 0.22 \
+df.loc[(df["Progress"] > 10) & (df["Progress"] <=45) , "Rating" ].mean() * 0.24 \
+df.loc[(df["Progress"] > 45) & (df["Progress"] <=75) , "Rating" ].mean() * 0.26 \
+df.loc[df["Progress"] > 75 ,"Rating"].mean() * 0.28

def user_based_weighted_average():
    pass

###########################
# Weighted Rating

def course_weighted_rating(df , time_w = 50 , user_w = 50):
    # time_w * time_weighted_average + user_w * user_weighted_average
    pass
