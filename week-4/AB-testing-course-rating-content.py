import pandas as pd
import numpy as np
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp , shapiro,levene,ttest_ind,mannwhitneyu,\
    pearsonr,spearmanr,kendalltau,f_oneway,kruskal
from statsmodels.stats.proportion import proportions_ztest

# kursu izleyenlerin verdiği puanla
# kursu izlemeyenlerin verdiği puan arasinda
# önemli bir fark var mı?

df = pd.read_csv("/home/msel/Desktop/MIUUL_13/miuul-13/miuul-13/week-4/measurement_problems/datasets/course_reviews.csv")
df.head()

df.loc[df["Progress"] > 75 , "Rating"].mean() # 4.86
df.loc[df["Progress"] < 25 , "Rating"].mean() # 4.72

#1 hipotezi kur
# h0: there are no rating difference
# between >75 and <25
#2 Varsayimlari belirle
#shapiro h0: test datas are not different
# from normal distribution
test_stat,pvalue = shapiro(df.loc[df["Progress"] > 75 , "Rating"])
test_stat,pvalue
#pvalue < 0 , so >75 is not normal dist.

test_stat,pvalue = shapiro(df.loc[df["Progress"] < 25 , "Rating"])
test_stat,pvalue
# pvalue < 0 so <25 is not normal dist.

# 3. hipotezi uygula
# non-parametric mannwhitneyu

test_stat,pvalue = mannwhitneyu(df.loc[df["Progress"] > 75 , "Rating"],
                                df.loc[df["Progress"] < 25 , "Rating"])
test_stat,pvalue

# p value < 0 , so we reject null hypotesis
# there are statistically significant
# difference rating between >75 progressers
# and <25 progressers