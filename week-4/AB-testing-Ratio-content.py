import pandas as pd
import numpy as np
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp , shapiro,levene,ttest_ind,mannwhitneyu,\
    pearsonr,spearmanr,kendalltau,f_oneway,kruskal
from statsmodels.stats.proportion import proportions_ztest

# İki Örneklem Oran testi
# kullanici arayüzü sadelestirme yapildi
# sade olanda 1000 kisi carpti , 300 üye oldu
# sade olmayanda 1100 kisi carpti , 250 üye oldu

# sadelestirme islemi statistically significant?

signup_counts = np.array([300,250])
hit_counts = np.array([1000,1100])

proportions_ztest(count=signup_counts,nobs=hit_counts)
#p-value < 0.05 , so we reject h0 null hypotesis

#####################################
# kadin ve erkeklerin hayatta kalma oranları
# arasinda fark var mı
df = sns.load_dataset("titanic")
df.head()

df.loc[df["sex"]=="female","survived"].mean()
df.loc[df["sex"]=="male","survived"].mean()

female_succ_count = df.loc[df["sex"]=="female","survived"].sum()
male_succ_count = df.loc[df["sex"]=="male","survived"].sum()
df[df["sex"]=="female"].shape[0]
df[df["sex"]=="male"].shape[0]

test_stat , pvalue = proportions_ztest(count=[female_succ_count,male_succ_count],
                                       nobs=[df[df["sex"]=="female"].shape[0] , df[df["sex"]=="male"].shape[0] ])


test_stat,pvalue
