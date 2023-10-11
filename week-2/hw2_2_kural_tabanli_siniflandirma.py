
#############################################
# Kural Tabanlı Sınıflandırma ile Potansiyel Müşteri Getirisi Hesaplama
#############################################

#############################################
# İş Problemi
#############################################
# Bir oyun şirketi müşterilerinin bazı özelliklerini kullanarak seviye tabanlı (level based) yeni müşteri tanımları (persona)
# oluşturmak ve bu yeni müşteri tanımlarına göre segmentler oluşturup bu segmentlere göre yeni gelebilecek müşterilerin şirkete
# ortalama ne kadar kazandırabileceğini tahmin etmek istemektedir.

# Örneğin: Türkiye’den IOS kullanıcısı olan 25 yaşındaki bir erkek kullanıcının ortalama ne kadar kazandırabileceği belirlenmek isteniyor.


#############################################
# Veri Seti Hikayesi
#############################################
# Persona.csv veri seti uluslararası bir oyun şirketinin sattığı ürünlerin fiyatlarını ve bu ürünleri satın alan kullanıcıların bazı
# demografik bilgilerini barındırmaktadır. Veri seti her satış işleminde oluşan kayıtlardan meydana gelmektedir. Bunun anlamı tablo
# tekilleştirilmemiştir. Diğer bir ifade ile belirli demografik özelliklere sahip bir kullanıcı birden fazla alışveriş yapmış olabilir.

# Price: Müşterinin harcama tutarı
# Source: Müşterinin bağlandığı cihaz türü
# Sex: Müşterinin cinsiyeti
# Country: Müşterinin ülkesi
# Age: Müşterinin yaşı

################# Uygulama Öncesi #####################

#    PRICE   SOURCE   SEX COUNTRY  AGE
# 0     39  android  male     bra   17
# 1     39  android  male     bra   17
# 2     49  android  male     bra   17
# 3     29  android  male     tur   17
# 4     49  android  male     tur   17

################# Uygulama Sonrası #####################

#       customers_level_based        PRICE SEGMENT
# 0   BRA_ANDROID_FEMALE_0_18  1139.800000       A
# 1  BRA_ANDROID_FEMALE_19_23  1070.600000       A
# 2  BRA_ANDROID_FEMALE_24_30   508.142857       A
# 3  BRA_ANDROID_FEMALE_31_40   233.166667       C
# 4  BRA_ANDROID_FEMALE_41_66   236.666667       C


#############################################
# PROJE GÖREVLERİ
#############################################

#############################################
# GÖREV 1: Aşağıdaki soruları yanıtlayınız.
#############################################
import numpy as np
import seaborn as sns
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Soru 1: persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.
persona = pd.read_csv("persona.csv")
print(persona.head())
print(persona.info())
print(persona.describe())
print(persona.isnull().sum())
print(persona.dtypes)

# Soru 2: Kaç unique SOURCE vardır? Frekansları nedir?
print(persona["SOURCE"].nunique())
print(persona["SOURCE"].value_counts())

# Soru 3: Kaç unique PRICE vardır?
print(persona["PRICE"].nunique())

# Soru 4: Hangi PRICE'dan kaçar tane satış gerçekleşmiş?
print(persona["PRICE"].value_counts())

# Soru 5: Hangi ülkeden kaçar tane satış olmuş?

print( persona["COUNTRY"].value_counts() )


# Soru 6: Ülkelere göre satışlardan toplam ne kadar kazanılmış?

print( persona.pivot_table("PRICE" ,"COUNTRY"  , aggfunc="sum") )
print( persona.groupby("COUNTRY").agg({"PRICE":"sum"}) )

# Soru 7: SOURCE türlerine göre göre satış sayıları nedir?
print( persona["SOURCE"].value_counts() )

# Soru 8: Ülkelere göre PRICE ortalamaları nedir?
print( persona.groupby("COUNTRY").agg({"PRICE":"mean"}) )

# Soru 9: SOURCE'lara göre PRICE ortalamaları nedir?
print( persona.groupby("SOURCE").agg({"PRICE":"mean"}) )

# Soru 10: COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?
print( persona.pivot_table("PRICE","COUNTRY","SOURCE","mean") )

#############################################
# GÖREV 2: COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir?
#############################################
#print( persona.groupby("PRICE").agg({["COUNTRY", "SOURCE", "SEX", "AGE"]: "mean"}) )

print(persona[["COUNTRY","SOURCE","SEX","AGE"]].head())
print(persona.groupby(["COUNTRY","SOURCE","SEX","AGE"] ).agg({"PRICE": "mean"}).head() )
print(persona.pivot_table("PRICE",["COUNTRY","SOURCE","SEX","AGE"] , aggfunc="mean" ).head()  )

#############################################
# GÖREV 3: Çıktıyı PRICE'a göre sıralayınız.
#############################################
# Önceki sorudaki çıktıyı daha iyi görebilmek için sort_values metodunu azalan olacak şekilde PRICE'a uygulayınız.
# Çıktıyı agg_df olarak kaydediniz.

agg_df = persona.pivot_table("PRICE",["COUNTRY","SOURCE","SEX","AGE"] , aggfunc="mean").sort_values(by = "PRICE",ascending=False)
print(agg_df.head())

#############################################
# GÖREV 4: Indekste yer alan isimleri değişken ismine çeviriniz.
#############################################
# Üçüncü sorunun çıktısında yer alan PRICE dışındaki tüm değişkenler index isimleridir.
# Bu isimleri değişken isimlerine çeviriniz.
# İpucu: reset_index()
# agg_df.reset_index(inplace=True)
print(agg_df.index)
agg_df.reset_index(inplace=True)
print(agg_df.head())


#############################################
# GÖREV 5: AGE değişkenini kategorik değişkene çeviriniz ve agg_df'e ekleyiniz.
#############################################
# Age sayısal değişkenini kategorik değişkene çeviriniz.
# Aralıkları ikna edici olacağını düşündüğünüz şekilde oluşturunuz.
# Örneğin: '0_18', '19_23', '24_30', '31_40', '41_70'

agg_df["AGE_CAT"]=pd.cut(agg_df["AGE"] , [0,18,23,30,40,70] , labels=['0_18', '19_23', '24_30', '31_40', '41_70'])
print(agg_df.head())



#############################################
# GÖREV 6: Yeni level based müşterileri tanımlayınız ve veri setine değişken olarak ekleyiniz.
#############################################
# customers_level_based adında bir değişken tanımlayınız ve veri setine bu değişkeni ekleyiniz.
# Dikkat!
# list comp ile customers_level_based değerleri oluşturulduktan sonra bu değerlerin tekilleştirilmesi gerekmektedir.
# Örneğin birden fazla şu ifadeden olabilir: USA_ANDROID_MALE_0_18
# Bunları groupby'a alıp price ortalamalarını almak gerekmektedir.


agg_df["customers_level_based"] = agg_df.select_dtypes(exclude=["int64","float64"]).apply(lambda x :"_".join(x).upper()  , axis=1)
print(agg_df.head())
customers_level_based = agg_df[["customers_level_based","PRICE"]].groupby("customers_level_based").agg({"PRICE": "mean"})
print(customers_level_based.head())
#############################################
# GÖREV 7: Yeni müşterileri (USA_ANDROID_MALE_0_18) segmentlere ayırınız.
#############################################
# PRICE'a göre segmentlere ayırınız,
# segmentleri "SEGMENT" isimlendirmesi ile agg_df'e ekleyiniz,
# segmentleri betimleyiniz,
agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"],4,labels=["D","C","B","A"])
print(agg_df.head())
print(agg_df[["SEGMENT","PRICE"]].groupby("SEGMENT").mean())

#############################################
# GÖREV 8: Yeni gelen müşterileri sınıflandırınız ne kadar gelir getirebileceğini tahmin ediniz.
#############################################
# 33 yaşında ANDROID kullanan bir Türk kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?

print(agg_df[(agg_df["SOURCE"]=="android") & (agg_df["SEX"]=="female") & (agg_df["AGE_CAT"] == "31_40") & (agg_df["COUNTRY"]=="tur")] )
print(agg_df[agg_df["customers_level_based"] == "TUR_ANDROID_FEMALE_31_40"])
# 35 yaşında IOS kullanan bir Fransız kadını hangi segmente ve ortalama ne kadar gelir kazandırması beklenir?
print(agg_df[agg_df["customers_level_based"] == "FRA_IOS_FEMALE_31_40"])

##

print(agg_df[["customers_level_based","PRICE","SEGMENT"]].groupby("customers_level_based").mean("PRICE"))