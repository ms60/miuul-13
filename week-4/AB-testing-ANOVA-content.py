import pandas as pd
import numpy as np
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp , shapiro,levene,ttest_ind,mannwhitneyu,\
    pearsonr,spearmanr,kendalltau,f_oneway,kruskal
from statsmodels.stats.proportion import proportions_ztest


df = sns.load_dataset("tips")
df.head()

# haftanın günleri arasında ödenen tipte
# istatistiksel olarak fark var mı
df.groupby("day").agg({"total_bill": "mean"})

# 1. hipotezi kur
# h0 : bütün günler arasında fark yoktur

# 2. varsayimlari kontrol
# normallik varsayımı
# h0: dataların normal dagılımdan farklı yoktur

for group in list(df["day"].unique()):
    pvalue = shapiro(df.loc[df["day"]==group,"total_bill"])[1]
    print(pvalue)

# p values < 0 so reject shapiro h0
# datas are not normal

# varyans kontrolü
# h0: varyanslar homojendir
test_stat,pvalue = levene(df.loc[df["day"]=="Sun","total_bill"],
                          df.loc[df["day"]=="Sat","total_bill"],
                          df.loc[df["day"]=="Thur","total_bill"],
                          df.loc[df["day"]=="Fri","total_bill"])
test_stat,pvalue
# pvalue > 0.05 , so variances are homogenous

# 3 hipotez testi
# non parametric test

df.groupby("day").agg({"total_bill": ["mean","median"]})

# parametric anova test
# f_oneway(df.loc[df["day"]=="Sun","total_bill"],
#                           df.loc[df["day"]=="Sat","total_bill"],
#                           df.loc[df["day"]=="Thur","total_bill"],
#                           df.loc[df["day"]=="Fri","total_bill"])

# non-parametric anova test
kruskal(df.loc[df["day"]=="Sun","total_bill"],
                          df.loc[df["day"]=="Sat","total_bill"],
                          df.loc[df["day"]=="Thur","total_bill"],
                          df.loc[df["day"]=="Fri","total_bill"])
# p < 0.05 so we reject h0 hypotesis


#### another comparison
from statsmodels.stats.multicomp import MultiComparison
comparison= MultiComparison(df["total_bill"],df["day"])
tukey = comparison.tukeyhsd(0.05)
print(tukey.summary())