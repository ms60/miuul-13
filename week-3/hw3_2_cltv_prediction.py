##############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
##############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################
# FLO satış ve pazarlama faaliyetleri için roadmap belirlemek istemektedir.
# Şirketin orta uzun vadeli plan yapabilmesi için var olan müşterilerin gelecekte şirkete sağlayacakları potansiyel değerin tahmin edilmesi gerekmektedir.


###############################################################
# Veri Seti Hikayesi
###############################################################

# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline alışveriş yapan) olarak yapan müşterilerin geçmiş alışveriş davranışlarından
# elde edilen bilgilerden oluşmaktadır.

# master_id: Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi


###############################################################
# GÖREVLER
###############################################################
# GÖREV 1: Veriyi Hazırlama
# 1. flo_data_20K.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.

import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler


# 2. Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.
# Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return round(low_limit), round(up_limit)


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


df = pd.read_csv("/home/husamdata/Desktop/MIUUL_13/miuul-13/miuul-13/week-3/FLOCLTVPrediction/flo_data_20k.csv")
df_ = df.copy()
df.head()
df.dtypes
df.describe().T

# 3. "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online" değişkenlerinin
# aykırı değerleri varsa baskılayanız.


replace_with_thresholds(df,"order_num_total_ever_online")
replace_with_thresholds(df,"order_num_total_ever_offline")
replace_with_thresholds(df,"customer_value_total_ever_offline")
replace_with_thresholds(df,"customer_value_total_ever_online")

df.describe().T


# 4. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Herbir müşterinin toplam
# alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.

df["total_order"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_purchase"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
df.head()


# 5. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.

#df.loc[: , [col for col in df.columns if "date" in col.lower()]].apply(lambda x: pd.to_datetime(x) , axis = 0)
#df.dtypes

for col in df.columns:
    if "date" in col:
        df[col] = pd.to_datetime(df[col])
df.dtypes
# GÖREV 2: CLTV Veri Yapısının Oluşturulması
# 1.Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.
from datetime import timedelta
analysis_date = df["last_order_date"].max() + timedelta(days=2)

# 2.customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturunuz.
# Monetary değeri satın alma başına ortalama değer olarak, recency ve tenure değerleri ise haftalık cinsten ifade edilecek.

df["monetary_cltv_avg"] = df["total_purchase"] / df["total_order"]
df["recency_cltv_weekly"] = (df["last_order_date"] - df["first_order_date"]).apply(lambda x: x.days / 7)
df["T_weekly"] = (analysis_date -  df["first_order_date"] ).apply(lambda x : x.days / 7)
df["frequency"] = df["total_order"]
cltv = df[["master_id" , "recency_cltv_weekly","T_weekly","frequency","monetary_cltv_avg"]]

cltv = cltv[cltv["frequency"] > 1]
cltv.set_index("master_id",inplace = True)

cltv.head()
cltv.describe().T



# GÖREV 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması, CLTV'nin hesaplanması
# 1. BG/NBD modelini fit ediniz.

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv['frequency'],cltv['recency_cltv_weekly'],cltv['T_weekly'])


# a. 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.
exp_sales_3_month = bgf.predict(4*3,cltv['frequency'],cltv['recency_cltv_weekly'],cltv['T_weekly']).sort_values(ascending=False)
exp_sales_3_month.head()
#exp_sales_3_month.to_csv("exp_sales_3_month.csv")

# b. 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.
exp_sales_6_month = bgf.predict(4*6,cltv['frequency'],cltv['recency_cltv_weekly'],cltv['T_weekly']).sort_values(ascending=False)
exp_sales_6_month.head()



# 2. Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv dataframe'ine ekleyiniz.
ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv['frequency'], cltv['monetary_cltv_avg'])
cltv["exp_average_value"] = ggf.conditional_expected_average_profit(cltv['frequency'],cltv['monetary_cltv_avg'])

# 3. 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.
cltv["cltv"] = ggf.customer_lifetime_value(bgf,cltv['frequency'],cltv['recency_cltv_weekly'],cltv['T_weekly'],cltv['monetary_cltv_avg'],time=6, freq="W",  discount_rate=0.01)

# b. Cltv değeri en yüksek 20 kişiyi gözlemleyiniz.
cltv.sort_values("cltv",ascending=False).head(20)

# GÖREV 4: CLTV'ye Göre Segmentlerin Oluşturulması
# 1. 6 aylık tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz. cltv_segment ismi ile dataframe'e ekleyiniz.
cltv["cltv_segment"] = pd.qcut(cltv["cltv"],q=4,labels=["D","C","B","A"])
cltv.head()
# 2. 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz

# BONUS: Tüm süreci fonksiyonlaştırınız.




