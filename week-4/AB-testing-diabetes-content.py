import pandas as pd
import numpy as np
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp , shapiro,levene,ttest_ind,mannwhitneyu,\
    pearsonr,spearmanr,kendalltau,f_oneway,kruskal
from statsmodels.stats.proportion import proportions_ztest

df = pd.read_csv("/home/msel/Desktop/MIUUL_13/miuul-13/miuul-13/week-4/measurement_problems/datasets/diabetes.csv")
df.head()

df.groupby("Outcome").agg({"Age":"mean"})
#1 hipotezi kur
# h0: there is not significantly difference
# between age of diabetes and non-diabetes person

# 2 Varsayimlari incele
# shapiro h0: datasets are not different from normal dist.
test_stat,pvalue = shapiro(df.loc[df["Outcome"]==0,"Age"])
test_stat,pvalue
# p value <0.05 so non-diabetics age are not normally distributed

test_stat,pvalue = shapiro(df.loc[df["Outcome"]==1,"Age"])
test_stat,pvalue
# p value <0.05 so diabetics age are not normally distributed

# so we need to use non-parametric mannwhitneyu

test_stat,pvalue = mannwhitneyu(df.loc[df["Outcome"]==0,"Age"],
                                df.loc[df["Outcome"]==1,"Age"])
test_stat,pvalue
# pvalue < 0.05 so we reject null hyptoesis
# so age is important for diabetes