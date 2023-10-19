###############################################################
# RFM ile Müşteri Segmentasyonu (Customer Segmentation with RFM)
###############################################################

# 1. İş Problemi (Business Problem)
# 2. Veriyi Anlama (Data Understanding)
# 3. Veri Hazırlama (Data Preparation)
# 4. RFM Metriklerinin Hesaplanması (Calculating RFM Metrics)
# 5. RFM Skorlarının Hesaplanması (Calculating RFM Scores)
# 6. RFM Segmentlerinin Oluşturulması ve Analiz Edilmesi (Creating & Analysing RFM Segments)
# 7. Tüm Sürecin Fonksiyonlaştırılması

###############################################################
# 1. İş Problemi (Business Problem)
###############################################################

# Bir e-ticaret şirketi müşterilerini segmentlere ayırıp bu segmentlere göre
# pazarlama stratejileri belirlemek istiyor.

# Veri Seti Hikayesi
# https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

# Online Retail II isimli veri seti İngiltere merkezli online bir satış mağazasının
# 01/12/2009 - 09/12/2011 tarihleri arasındaki satışlarını içeriyor.

# Değişkenler
#
# InvoiceNo: Fatura numarası. Her işleme yani faturaya ait eşsiz numara. C ile başlıyorsa iptal edilen işlem.
# StockCode: Ürün kodu. Her bir ürün için eşsiz numara.
# Description: Ürün ismi
# Quantity: Ürün adedi. Faturalardaki ürünlerden kaçar tane satıldığını ifade etmektedir.
# InvoiceDate: Fatura tarihi ve zamanı.
# UnitPrice: Ürün fiyatı (Sterlin cinsinden)
# CustomerID: Eşsiz müşteri numarası
# Country: Ülke ismi. Müşterinin yaşadığı ülke.


###############################################################

import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option('display.max_rows', None)
pd.set_option("display.float_format" , lambda x: '%.3f' % x)

df_ = pd.read_excel("/home/husamdata/Desktop/MIUUL_13/miuul-13/miuul-13/week-3/crmAnalytics/datasets/online_retail_II.xlsx" , sheet_name="Year 2009-2010")
df = df_.copy() # deep copy
print(df.head())
print(df.shape)
df.isnull().sum()

#unique urun sayisi
df["Description"].isnull().sum()
df.loc[~df["Description"].isnull() , "Description"].nunique(dropna=False)
#or
df["StockCode"].isnull().sum()
df["StockCode"].nunique()

df["Description"].value_counts().head()
df.groupby("Description").agg({"Quantity":"sum"})

df["Invoice"].nunique()

df["Total_Price"] = df["Quantity"] * df["Price"]
df.groupby("Invoice").agg({"Total_Price":"sum"}).head()

## 3 . VERI HAZIRLAMA
df.shape
df.isnull().sum()
df.dropna(inplace=True)
df.shape
df.describe().T

# remove returned products
df = df[~df["Invoice"].str.contains("C",na=False)]

# 4 . Calculating RFM Metrics
# Recency , Frequency , Monetary
df.head()
df.shape
import datetime
df["InvoiceDate"].max()
today_date = datetime.datetime(2010,12,11) # 2 days later of max
today_date

rfm = df.groupby("Customer ID").agg({"InvoiceDate":lambda x : (today_date - x.max()).days,"Invoice": lambda x : x.nunique(),"Total_Price": lambda x : x.sum()})
rfm.head()
rfm.columns = ["Recency","Frequency","Monetary"]

# 5 . Calculating RFM Scores
rfm["Recency_Score"] = pd.qcut(rfm["Recency"],5,labels=[5,4,3,2,1])
rfm["Monetary_Score"] = pd.qcut(rfm["Monetary"] , 5 , labels=[1,2,3,4,5])
# rfm["Frequency_Score"] = pd.qcut(rfm["Frequency"],5,labels=[1,2,3,4,5])
# value error
rfm["Frequency"].rank(method="first")
rfm["Frequency_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"),5,labels=[1,2,3,4,5])

rfm.head()

rfm["RFM_SCORE"]  = rfm["Recency_Score"].astype(str) + rfm["Frequency_Score"].astype(str)
rfm.head()

rfm[rfm["RFM_SCORE"] == "55"] # champions
rfm[rfm["RFM_SCORE"] == "11"] # hibernating

# 6. RFM Segmentation and Analysis

# regex
seg_map = {
    r'[1-2][1-2]':'hibernating',
    r'[1-2][3-4]': 'at_risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]':'about_to_sleep',
    r'33':'need_attention',
    r'[3-4][4-5]':'loyal_customers',
    r'41':'promising',
    r'51':'new_customers',
    r'[4-5][2-3]':'potantial_loyalists',
    r'5[4-5]':'champions'
}

rfm["SEGMENT"] = rfm["RFM_SCORE"].replace(seg_map , regex=True)
rfm[["SEGMENT","Recency","Frequency","Monetary"]].groupby("SEGMENT").agg(["mean","count"])

# 7. function for all process
