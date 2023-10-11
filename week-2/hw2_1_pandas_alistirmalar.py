# Pandas Alıştırmalar
##################################################

import numpy as np
import seaborn as sns
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

#########################################
# Görev 1: Seaborn kütüphanesi içerisinden Titanic veri setini tanımlayınız.
#########################################
df = sns.load_dataset("titanic")

#########################################
# Görev 2: Yukarıda tanımlanan Titanic veri setindeki kadın ve erkek yolcuların sayısını bulunuz.
#########################################

print(df["sex"].value_counts())


#########################################
# Görev 3: Her bir sutuna ait unique değerlerin sayısını bulunuz.
#########################################

print(df.nunique())

#########################################
# Görev 4: pclass değişkeninin unique değerleri bulunuz.
#########################################
print(df["pclass"].nunique())



#########################################
# Görev 5:  pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz.
#########################################
print(df[["pclass","parch"]].nunique())


#########################################
# Görev 6: embarked değişkeninin tipini kontrol ediniz. Tipini category olarak değiştiriniz. Tekrar tipini kontrol ediniz.
#########################################

print(df["embarked"].dtype)                         
df["embarked"] = df["embarked"].astype("category")
print(df["embarked"].dtype)
#########################################
# Görev 7: embarked değeri C olanların tüm bilgelerini gösteriniz.
#########################################

print( df[df["embarked"] == "C"].head() )


#########################################
# Görev 8: embarked değeri S olmayanların tüm bilgelerini gösteriniz.
#########################################
print(df[df["embarked"] != "S"].head())


#########################################
# Görev 9: Yaşı 30 dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.
#########################################

print(df[(df["sex"] == "female") & (df["age"] < 30)].head())

#########################################
# Görev 10: Fare'i 500'den büyük veya yaşı 70 den büyük yolcuların bilgilerini gösteriniz.
#########################################
print( df[ (df["fare"] > 500) | (df["age"] > 70 ) ] )


#########################################
# Görev 11: Her bir değişkendeki boş değerlerin toplamını bulunuz.
#########################################

print(df.isnull())
print(df.isnull().sum())

#########################################
# Görev 12: who değişkenini dataframe'den düşürün.
#########################################
print(df.columns)
df.drop("who" ,axis=1,inplace=True)
print(df.columns)
#########################################
# Görev 13: deck değikenindeki boş değerleri deck değişkenin en çok tekrar eden değeri (mode) ile doldurunuz.
#########################################
print( df["deck"].isnull() )
print("--")
print(df["deck"].mode()  ) # most frequent value
print(df["deck"].value_counts().idxmax()) #most frequent value
print(df["deck"].isnull().sum())
#first solution select via dataframe loc
df.loc[df["deck"].isnull() , "deck"] = df["deck"].value_counts().idxmax()
#df.loc[df["deck"].isnull() ]["deck"] = df["deck"].value_counts().idxmax()
# this doesnt work due to __getitem__ __setitem__ issue ,
# always use loc to get references
#second solution select and assign via series

#df.fillna()
df["deck"][df["deck"].isnull()] = df["deck"].value_counts().idxmax()
print(df["deck"].isnull().sum())


#########################################
# Görev 14: age değikenindeki boş değerleri age değişkenin medyanı ile doldurun.
#########################################
print(df["age"].isnull().sum())
df.loc[ df["age"].isnull() , "age"] = df["age"].median()
print(df["age"].isnull().sum())
#########################################
# Görev 15: survived değişkeninin Pclass ve Cinsiyet değişkenleri kırılımınında sum, count, mean değerlerini bulunuz.
#########################################
print( df.pivot_table("survived" , "pclass","sex",aggfunc=["count","sum","mean"]) )



#########################################
# Görev 16:  30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 vericek bir fonksiyon yazınız.
# Yazdığınız fonksiyonu kullanarak titanik veri setinde age_flag adında bir değişken oluşturunuz oluşturunuz. (apply ve lambda yapılarını kullanınız)
#########################################
df["age_flag"] = df["age"].apply(lambda x: 1 if x < 30 else 0)
print(df[["age","age_flag"]])

#########################################
# Görev 17: Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız.
#########################################
tips = sns.load_dataset("tips")
print(tips.head())


#########################################
# Görev 18: Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill  değerlerinin toplamını, min, max ve ortalamasını bulunuz.
#########################################
print(tips.pivot_table("total_bill","time",aggfunc=["min","max","mean"]) )
print( tips.groupby("time").agg({"total_bill": ["min", "max", "mean"]}) )

#########################################
# Görev 19: Günlere ve time göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.
#########################################
print( tips.groupby(["day","time",]).agg({"total_bill": ["min", "max", "mean"]}) )
print(tips.pivot_table("total_bill","time" , "day",aggfunc=["min","max","mean"]) )

#########################################
# Görev 20:Lunch zamanına ve kadın müşterilere ait total_bill ve tip  değerlerinin day'e göre toplamını, min, max ve ortalamasını bulunuz.
#########################################
print(tips[(tips["sex"]=="Female") & (tips["time"]=="Dinner")].pivot_table(["total_bill","tip"],"day",aggfunc=["count","sum","min","max","mean"])  )


#########################################
# Görev 21: size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir?
#########################################
print(tips.dtypes)
print(tips.loc[(tips["size"] < 3) & (tips["total_bill"] > 10) ,"total_bill" ].mean()  )

#########################################
# Görev 22: total_bill_tip_sum adında yeni bir değişken oluşturun. Her bir müşterinin ödediği totalbill ve tip in toplamını versin.
#########################################
tips["total_bill_tip_sum"] = tips["total_bill"] + tips["tip"]
print(tips.head())


#########################################
# Görev 23: total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve ilk 30 kişiyi yeni bir dataframe'e atayınız.
#########################################
new_tips = tips.sort_values( by="total_bill_tip_sum" ,ascending=False).head(30)
print(new_tips)

