# Basic Statistics
# # Sampling
import pandas as pd
import numpy as np
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp , shapiro,levene,ttest_ind,mannwhitneyu,\
    pearsonr,spearmanr,kendalltau,f_oneway,kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)
pd.set_option("display.width",500)
pd.set_option("display.expand_frame_repr",False)
pd.set_option("display.float_format",lambda x: '%.5f' % x)


# Sampling
# sampling from population

population = np.random.randint(0,80,10000)
population.mean()
np.random.seed(115)
sampling = np.random.choice(a=population , size=100)
sampling.mean()

np.random.seed(10)
sampling1 = np.random.choice(a=population,size=100)
sampling2 = np.random.choice(a=population,size=100)
sampling3 = np.random.choice(a=population,size=100)
sampling4 = np.random.choice(a=population,size=100)
sampling5 = np.random.choice(a=population,size=100)
sampling6 = np.random.choice(a=population,size=100)
sampling7 = np.random.choice(a=population,size=100)
sampling8 = np.random.choice(a=population,size=100)
sampling9 = np.random.choice(a=population,size=100)
sampling10 = np.random.choice(a=population,size=100)

####
# Descriptive Statistics
df = sns.load_dataset("tips")
df.describe().T

###
# Confidence Intervals
df.head()
sms.DescrStatsW(df["total_bill"]).tconfint_mean()
# total_bill %95 güven ile bu 2 aralıgın arasındadır

sms.DescrStatsW(df["tip"]).tconfint_mean()
# tips %95 güven ile bu araliktadir

df = sns.load_dataset("titanic")
sms.DescrStatsW(df["age"].dropna()).tconfint_mean()
sms.DescrStatsW(df["fare"].dropna()).tconfint_mean()

#####
# Correlation
df = sns.load_dataset("tips")
df.head()
# correlation between tips-totall_bill
# total_bill = bill + tip
df["bill"] = df["total_bill"] - df["tip"]

df.plot.scatter("tip","bill")

df["tip"].corr(df["bill"])
