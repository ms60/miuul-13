import pandas as pd
import seaborn as sns
df = sns.load_dataset("titanic")

print(df.head())
sex_embarked_values_survived = df.pivot_table(values="survived",index=["sex","deck"],columns="embarked",aggfunc=["mean","std"])
print(sex_embarked_values_survived)
print(sex_embarked_values_survived.index)
sex_embarked_values_survived_gby = df.groupby(["sex","embarked"]).agg({"survived":["mean","std"]})
print(sex_embarked_values_survived_gby)
print(sex_embarked_values_survived_gby.index)