import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import math
import scipy.stats as st
pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)
pd.set_option("display.expand_frame_repr",False)

df = pd.read_csv("/home/msel/Desktop/MIUUL_13/miuul-13/miuul-13/week-4/measurement_problems/datasets/movies_metadata.csv")
df = df[["title","vote_average","vote_count"]]
df.head()

# sorting by Vote average

df.sort_values("vote_average",ascending=False).head(20)
# vote_count 1 olanlar var

df["vote_count"].describe([0.1,0.25,0.5,0.7,0.8,0.9,0.95,0.99]).T

df[df["vote_count"] > 400].sort_values("vote_average" , ascending=False).head(20)
# begenmedik

df["vote_count_score"] = MinMaxScaler(feature_range=(1,10))\
    .fit(df[["vote_count"]])\
    .transform(df[["vote_count"]])

df["average_count_score"] = df["vote_average"] * df["vote_count_score"]

df.sort_values("average_count_score",ascending=False).head(20)

# imdb nin kendi custom sıralama yontemi
# weighted_rating = (v/(v+M)*r) + (M/(v+M)*C)
# r = vote average
# v = vote_count
# M = minimum votes required to be listed in top 250
# C = the mean vote across the whole report

M = 2500
C = df["vote_average"].mean()

def weighted_rating(r,v,M,C):
    return (v/(v+M)*r) + (M/(v+M)*C)

df["weighted_rating"] = weighted_rating(df["vote_average"],
                                        df["vote_count"],M,C)

df.sort_values("weighted_rating" , ascending=False).head(10)

# Bayesian Average Rating Score
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

