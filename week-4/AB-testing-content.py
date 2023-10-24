# AB testing
# bagimsiz iki örneklem T testi

# 1. Hipotezi kur
# 2 .Varsayım kontrolü
#       a.Normallik varsayıı
#        b.Varyans homojenliği
# 3.Hipotezi uygula
#   a.varsayimlar saglaniyorsa t testi (parametrik test)
#   b. varsayımlar saglanmıyorsa mannwhitneyu testi (non-parametrik test)

# Not:
# - normallik saglanmıyorsa direk 2 numara
#    varyans homojenligi saglanmıyorsa 1 numaraya gidilir
# - normallik incelemesi öncesi aykırı degerlere bakmak ve
#    düzeltmek faydalı olabilir

# 4 . p-value degerine göre yorumla


# uygulama 1
# sigara icenlerle icmeyenlerin ödedikleri
# hesap ortalamaları arasında fark var mı?

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

df.groupby("smoker").agg({"total_bill":"mean"})
#1 hipotezi kur
# H0 : M1=M2 fark yoktur

#2 varsayim kontrolü

# check if data distributed normal
test_stat , pvalue = shapiro(df.loc[df["smoker"]=='Yes' , "total_bill"])
(test_stat,pvalue)
# h0: shapiro der ki verinin dagılımıyla normal dagılım arasında fark yoktur
# pvalue = 0.0002 < 0.05 -> h0 reject, so not normal distributed

test_stat , pvalue = shapiro(df.loc[df["smoker"]=='No' , "total_bill"])
(test_stat,pvalue)
# pvalue < 0.05 so not normal distributed

# check if variance is homogenous
# levene says h0 : variances are homogenous
test_stat , pvalue = levene(df.loc[df["smoker"]=="Yes","total_bill"],
                            df.loc[df["smoker"]=="No","total_bill"])

(test_stat,pvalue)
# p value < 0.05 so variances are not homogenous


# 3. Hipotezin uygulanması

test_stat , pvalue = mannwhitneyu(df.loc[df["smoker"]=="Yes","total_bill"],
                             df.loc[df["smoker"]=="No","total_bill"])

(test_stat,pvalue)

# p value = 0.34 > 0.05 so we cant reject h0 hypotesis
# so we cant say distributions are different



# test_stat , pvalue = ttest_ind(df.loc[df["smoker"]=="Yes","total_bill"],
#                             df.loc[df["smoker"]=="No","total_bill"],
#                                equal_var = False)
# (test_stat,pvalue)

##############

df = sns.load_dataset("titanic")
df.head()

df.groupby("sex").agg({"age":"mean"})
# kadın-erkek yolcuların yaslarında bi farklılık var mı?

#1 hipotezi kur
# h0 : kadin erkek yas arasında fark yoktur

#2 varsayımlari incele
# saphiro h0: dagilim normalden farkı yoktur
test_stat , pvalue = shapiro(df.loc[df["sex"]=="female","age"].dropna())
test_stat , pvalue
# pvalue = 0.007 < 0.05 , so we reject saphiro h0
# woman ages are not normal

# saphiro h0: dagilim normalden farkı yoktur
test_stat , pvalue = shapiro(df.loc[df["sex"]=="male","age"].dropna())
test_stat , pvalue
# pvalue = 0,0000... < 0.05 , so we reject saphiro h0
# men ages are not normal

# non-parametric test kullanacagiz

# variance homogenous
# levene h0: distrubtions are homogenous
test_stat , pvalue = levene(df.loc[df["sex"]=="female","age"].dropna(),
                            df.loc[df["sex"]=="male","age"].dropna())
test_stat,pvalue
# pvalue = 0.97 > 0.05 so we fail to reject levene h0

#3 hipotezi uygulanon parametric mannwhiteneyu test
#h0 neydi : kadin erkek arasinda fark yoktur
test_stat,pvalue = mannwhitneyu(df.loc[df["sex"]=="female","age"].dropna(),
                                df.loc[df["sex"]=="male","age"].dropna())
test_stat,pvalue
# pvalue = 0.02 < 0.05 so we reject h0 null hypotesis
# there iss statistically significant difference
# between ages of men and women