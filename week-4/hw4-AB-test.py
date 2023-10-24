#####################################################
# AB Testi ile BiddingYöntemlerinin Dönüşümünün Karşılaştırılması
#####################################################
import pandas as pd
import numpy as np
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp , shapiro,levene,ttest_ind,mannwhitneyu,\
    pearsonr,spearmanr,kendalltau,f_oneway,kruskal
from statsmodels.stats.proportion import proportions_ztest


#####################################################
# İş Problemi
#####################################################

# Facebook kısa süre önce mevcut "maximumbidding" adı verilen teklif verme türüne alternatif
# olarak yeni bir teklif türü olan "average bidding"’i tanıttı. Müşterilerimizden biri olan bombabomba.com,
# bu yeni özelliği test etmeye karar verdi veaveragebidding'in maximumbidding'den daha fazla dönüşüm
# getirip getirmediğini anlamak için bir A/B testi yapmak istiyor.A/B testi 1 aydır devam ediyor ve
# bombabomba.com şimdi sizden bu A/B testinin sonuçlarını analiz etmenizi bekliyor.Bombabomba.com için
# nihai başarı ölçütü Purchase'dır. Bu nedenle, istatistiksel testler için Purchasemetriğine odaklanılmalıdır.




#####################################################
# Veri Seti Hikayesi
#####################################################

# Bir firmanın web site bilgilerini içeren bu veri setinde kullanıcıların gördükleri ve tıkladıkları
# reklam sayıları gibi bilgilerin yanı sıra buradan gelen kazanç bilgileri yer almaktadır.Kontrol ve Test
# grubu olmak üzere iki ayrı veri seti vardır. Bu veri setleriab_testing.xlsxexcel’ininayrı sayfalarında yer
# almaktadır. Kontrol grubuna Maximum Bidding, test grubuna AverageBiddinguygulanmıştır.

# impression: Reklam görüntüleme sayısı
# Click: Görüntülenen reklama tıklama sayısı
# Purchase: Tıklanan reklamlar sonrası satın alınan ürün sayısı
# Earning: Satın alınan ürünler sonrası elde edilen kazanç



#####################################################
# Proje Görevleri
#####################################################

######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)
######################################################

# 1. Hipotezleri Kur
# 2. Varsayım Kontrolü
#   - 1. Normallik Varsayımı (shapiro)
#   - 2. Varyans Homojenliği (levene)
# 3. Hipotezin Uygulanması
#   - 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi
#   - 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi
# 4. p-value değerine göre sonuçları yorumla
# Not:
# - Normallik sağlanmıyorsa direkt 2 numara. Varyans homojenliği sağlanmıyorsa 1 numaraya arguman girilir.
# - Normallik incelemesi öncesi aykırı değer incelemesi ve düzeltmesi yapmak faydalı olabilir.




#####################################################
# Görev 1:  Veriyi Hazırlama ve Analiz Etme
#####################################################

# Adım 1:  ab_testing_data.xlsx adlı kontrol ve test grubu verilerinden oluşan veri setini okutunuz. Kontrol ve test grubu verilerini ayrı değişkenlere atayınız.

control_group = pd.read_excel("/home/msel/Desktop/MIUUL_13/miuul-13/miuul-13/week-4/ABTesti/ab_testing.xlsx",sheet_name="Control Group")
test_group = pd.read_excel("/home/msel/Desktop/MIUUL_13/miuul-13/miuul-13/week-4/ABTesti/ab_testing.xlsx",sheet_name="Test Group")
# Adım 2: Kontrol ve test grubu verilerini analiz ediniz.

control_group.head()
control_group.describe().T
control_group.isnull().sum()

test_group.head()
test_group.describe().T
test_group.isnull().sum()

# Adım 3: Analiz işleminden sonra concat metodunu kullanarak kontrol ve test grubu verilerini birleştiriniz.

control_test_group = pd.concat([control_group,test_group])
control_test_group.head()




#####################################################
# Görev 2:  A/B Testinin Hipotezinin Tanımlanması
#####################################################

# Adım 1: Hipotezi tanımlayınız.
# h0: there is no difference between earnings
# of control and test groups

# Adım 2: Kontrol ve test grubu için purchase(kazanç) ortalamalarını analiz ediniz
control_group["Purchase"].mean()
test_group["Purchase"].mean()


#####################################################
# GÖREV 3: Hipotez Testinin Gerçekleştirilmesi
#####################################################

######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)
######################################################


# Adım 1: Hipotez testi yapılmadan önce varsayım kontrollerini yapınız.Bunlar Normallik Varsayımı ve Varyans Homojenliğidir.

# Kontrol ve test grubunun normallik varsayımına uyup uymadığını Purchase değişkeni üzerinden ayrı ayrı test ediniz
# shapiro h0: data is not different than normal dist.
test_stats,pvalue = shapiro(test_group["Purchase"])
test_stats,pvalue
# pvalue = 0.15 > 0.05 so we fail to reject shapiro h0
# so we can say normal dist.

test_stats,pvalue = shapiro(control_group["Purchase"])
test_stats,pvalue
# pvalue = 0.589 > 0.05
# so we can say normal dist.

# varyans kontrolu
# levene h0: variances are homogenous

test_stats,pvalue = levene(control_group["Purchase"],test_group["Purchase"])
test_stats,pvalue
#p value = 0.1 > 0.05 so we fail to reject h0
# variances are homogenous



# Adım 2: Normallik Varsayımı ve Varyans Homojenliği sonuçlarına göre uygun testi seçiniz

# normal distributed and homogenous ,
# so we use parametric test
test_stat , pvalue = ttest_ind(control_group["Purchase"],
                             test_group["Purchase"],
                                equal_var = True)

# Adım 3: Test sonucunda elde edilen p_value değerini göz önünde bulundurarak kontrol ve test grubu satın alma
# ortalamaları arasında istatistiki olarak anlamlı bir fark olup olmadığını yorumlayınız.

test_stat , pvalue
# p value = 0.34 > 0.05 so we can't reject h0 hypotesis
# there is no statistically significant different
# between two datasets

##############################################################
# GÖREV 4 : Sonuçların Analizi
##############################################################

# Adım 1: Hangi testi kullandınız, sebeplerini belirtiniz.




# Adım 2: Elde ettiğiniz test sonuçlarına göre müşteriye tavsiyede bulununuz.

# SORULAR
# 1 neden concat
# 2 ürün bazında oran testi daha mantıklı degil miydi
#       40 ürünün ortalama verilerine bakarak degisiklik yok dedik
#       ama ürün bazında purchase/impressions oran testi yapmak her bir ürüne etkisine bakıp
#       öyle ortalama almak ?
# 3 test_stat , pvalue deki test_stat ne?
# 4 AB testinde dagilimlar biri normal
#    digeri normal degilse hangi testi sececegiz?
#    dagilim degismis yani farklılık var mı diyecegiz?
