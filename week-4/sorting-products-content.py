import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import math
import scipy.stats as st
pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)
pd.set_option("display.expand_frame_repr",False)

df = pd.read_csv("/home/msel/Desktop/MIUUL_13/miuul-13/miuul-13/week-4/measurement_problems/datasets/product_sorting.csv")
df.shape
df.head(10)

# sorting by ratings

df.sort_values("rating",ascending=False).head(20)
# satın alma sayısı ve yorum sayıları ezilmis
# social proof - bu faktorleride dikkate almak lazım

df.sort_values("commment_count",ascending=False).head(20)

# Sorting by Rating , Comment  , Purchase
# bu ucunu carpsak rating cok kucuk ezilir
# burada normalize - ölceklendirme yapmak lazım
# burada mesela rating 0-5.0 arası , diger degiskenleride buna cekelim


## single brackets returns series
## double brackets return dataframe
MinMaxScaler(feature_range=(1,5)).fit(df["purchase_count"].to_frame()).transform(df[["purchase_count"]])

df["purchase_count_scaled"] = MinMaxScaler(feature_range=(1,5))\
    .fit(df[["purchase_count"]])\
    .transform(df[["purchase_count"]])

df["comment_count_scaled"] = MinMaxScaler(feature_range=(1,5))\
    .fit(df[["commment_count"]])\
    .transform(df[["commment_count"]])


df.head(20)

df["sorting_values_scaled"]=df["comment_count_scaled"] * 0.32 + \
df["purchase_count_scaled"] * 0.26 + \
df["rating"] * 0.42

df.sort_values("sorting_values_scaled",ascending=False).head(20)

def weighted_sorting_score(df , w1 , w2,w3):
    pass

#-----------------------
# Bayesian Average Rating Score

# sorting products with 5 star rated
# sorting products according to dist. of 5 star rating


# puanların dagılım bilgisini kullanarak
#   olasılıksal ortalama hesaplayacagız


def bayesian_average_rating(n , confidence = 0.95):
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score

df["bar_score"] = df.apply(lambda col: bayesian_average_rating(col[["1_point",
                                                                    "2_point",
                                                                    "3_point",
                                                                    "4_point",
                                                                    "5_point"]]) , axis = 1)

df.sort_values("bar_score",ascending=False).head(20)

# summary until now
# Rating Products
# -Average
# -Time-Based Weighted Average
# -User-Based Weighted Average
# -Weighted Rating
# -Bayesian Average Rating Score

# Sorting Products
# -Sorting by Rating
# -Sorting by Comment count or Purchase count
# -Sorting by Rating,Comment and Purchase Weighted
# -Sorting by Average Bayesian Rating Score
# -Hybrid Sorting : BAR Score + other factors


def hybrid_sorting_score(df , bar_w = 60 , wss_w = 40):
    # bar_w: bayesian average score
    # wss_w: weighted rating,comment_count,purchase_count
    pass
