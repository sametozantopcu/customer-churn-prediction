"""
Customer Churn Prediction Pipeline
==================================
Müşteri terk (churn) riskini tahmin etmek amacıyla geliştirilmiş uçtan uca 
makine öğrenmesi pipeline'dır. Logistic Regression ve KNN modelleri 
eğitilerek performansları ve sınıf dengesizliği etkileri incelenmiştir.

Kullanılan Ana Kütüphaneler:
- pandas, numpy, scikit-learn, matplotlib

Kurulum ve Çalıştırma:
1. Bağımlılıkları yükleyin:
   $ pip install -r requirements.txt

2. Script'i çalıştırın:
   $ python odev.py
"""

# 1. VERİ YÜKLEME VE KEŞİFSEL VERİ ANALİZİ (EDA)
# ==============================================================================
# Müşteri veri setinin yüklenmesi, boyutlarının kontrol edilmesi ve 
# hedef değişken olan müşteri kayıp (churn) dağılımının incelenmesi.

import pandas as pd

df = pd.read_csv("musteri_veri_seti.csv")

print("İlk 5 Satır:\n", df.head())
print(df.info())
print(f"\nVeri Seti Boyutu: {df.shape}")
print(f"\nHedef Değişken Dağılımı: {df["churn"].value_counts()}")

# 2. VERİ ÖN İŞLEME VE EKSİK VERİ YÖNETİMİ
# ==============================================================================
# Veri setindeki eksik (null) değerlerin tespiti ve değişken tipine uygun 
# istatistiksel yöntemlerle (sayısal değişkenler için medyan, kategorikler 
# için mod) doldurulması.

print("\nEksik Değerler:\n", df.isnull().sum())
eksik_sayisal_sutunlar = ["yas", "destek_talebi_sayisi"]

for sutun in eksik_sayisal_sutunlar:
    medydan_degeri = df[sutun].median()
    df[sutun] = df[sutun].fillna(medydan_degeri)

eksik_kategorik_sutunlar = ["sehir", "uyelik_tipi"]

for sutun in eksik_kategorik_sutunlar:
    mod_degeri = df[sutun].mode()[0]
    df[sutun] = df[sutun].fillna(mod_degeri)

print(f"\nEksik veriler doldurulduktan sonra: \n{df.head()}\n")
print(f"Eksik veriler doldurulduktan sonra: \n{df.isnull().sum()}\n")

sayisal_sutunlar = ["yas", "gelir", "abonelik_suresi", "destek_talebi_sayisi"]
aykiri_deger_maskesi = pd.Series(False, index=df.index)

for sutun in sayisal_sutunlar:
    q1 = df[sutun].quantile(0.25)
    q3 = df[sutun].quantile(0.75)
    iqr = q3 - q1
    alt_sinir = q1 - 1.5 * iqr
    ust_sinir = q3 + 1.5 * iqr
    sutun_maskesi = ((df[sutun] < alt_sinir) | (df[sutun] > ust_sinir))
    aykiri_deger_maskesi = aykiri_deger_maskesi | sutun_maskesi
    print(f"{sutun} - aykırı değer sayısı: {sutun_maskesi.sum()}")

print(f"\nToplam aykırı değer satırı: {aykiri_deger_maskesi.sum()}")

df = df.loc[~aykiri_deger_maskesi].reset_index(drop=True)

print(f"Aykırı değerler silindikten sonra güncel veri seti: {df.shape}")

# 3. ÖZNİTELİK MÜHENDİSLİĞİ (FEATURE ENGINEERING)
# ==============================================================================
# Modelin tahmin gücünü ve değişkenler arası etkileşimi yakalama yeteneğini 
# artırmak amacıyla alan bilgisine dayalı yeni türetilmiş özniteliklerin oluşturulması.

df["destek_orani"] = df["destek_talebi_sayisi"] / (df["abonelik_suresi"] + 1)  
df["gelir_yas_orani"] = df["gelir"] / df["yas"]
df["destek_abonelik_farki"] = df["destek_talebi_sayisi"] - (df["abonelik_suresi"] / 10)

korelasyonlar = df.corr(numeric_only=True)["churn"].sort_values(ascending=False)
print(f"\nKorelasyonlar: \n{korelasyonlar}")

secilen_oznitelikler = korelasyonlar[abs(korelasyonlar) > 0.50].index.tolist()
secilen_oznitelikler.remove("churn")

print(secilen_oznitelikler)
df = df.drop(columns=["gelir_yas_orani"])
print("\nYeni öznitelikler eklendikten sonra ilk 5 satır:")
print(df.head())

# 4. KATEGORİK DEĞİŞKEN DÖNÜŞÜMÜ (CATEGORICAL ENCODING)
# ==============================================================================
# Algoritmaların işleyebilmesi amacıyla kategorik özniteliklerin (sehir, uyelik_tipi) 
# One-Hot Encoding yöntemiyle sayısal vektörlere dönüştürülmesi ve kukla değişken 
# tuzağından kaçınmak için ilk kategorinin düşürülmesi.

y = df["churn"].values
X = df.drop(columns= ["churn"])
X = pd.get_dummies(X, columns= ["sehir", "uyelik_tipi"], drop_first= True, dtype= int)

print("\nKategorik dönüşüm sonrası ilk 5 satır")
print(X.head())

# 5. VERİ KÜMESİNİN BÖLÜNMESİ (TRAIN / VALIDATION / TEST SPLIT)
# ==============================================================================
# Aşırı öğrenmeyi (overfitting) önlemek ve genelleme performansını ölçmek amacıyla
# verinin %70 Train, %15 Validation ve %15 Test olarak ayrılması.
# Hedef değişkendeki sınıf dengesizliğini korumak için 'stratify' kullanılmıştır.

from sklearn.model_selection import train_test_split 

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size= 0.3, random_state= 42, stratify= y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size= 0.5, random_state= 42, stratify=y_temp )

print(f"\nX_train: {X_train.shape}")
print(f"X_val: {X_val.shape}")
print(f"X_test: {X_test.shape}")

# 6. ÖZNİTELİK ÖLÇEKLEME (FEATURE SCALING)
# ==============================================================================
# Sayısal değişkenlerin farklı ölçeklerde olmasından kaynaklı bias'ı önlemek ve
# model hassasiyetini artırmak için StandardScaler uygulanması.
# Data Leakage (veri sızıntısı) engellemek adına scaler sadece Train verisiyle fit edilmiştir.

from sklearn.preprocessing import StandardScaler

sayisal_sutunlar = ["yas", "gelir", "abonelik_suresi", "destek_talebi_sayisi", "destek_orani", "destek_abonelik_farki"]

standard_scaler = StandardScaler()

X_train_standard = X_train.copy()
X_val_standard = X_val.copy()
X_test_standard = X_test.copy()

X_train_standard[sayisal_sutunlar] = (
    standard_scaler.fit_transform(
        X_train[sayisal_sutunlar]
    )
)

X_val_standard[sayisal_sutunlar] = (
    standard_scaler.transform(
        X_val[sayisal_sutunlar]
    )
)

X_test_standard[sayisal_sutunlar] = (
    standard_scaler.transform(
        X_test[sayisal_sutunlar]
    )
)

print(f"\nX_train_standard: \n{X_train_standard.head()}")

# 7. MODEL EĞİTİMİ VE MİMARİ SEÇİMİ (MODEL TRAINING)
# ==============================================================================
# Veri seti üzerinde farklı matematiksel yaklaşımları kıyaslamak amacıyla
# Doğrusal Tabanlı (Logistic Regression) ve Uzaklık Tabanlı (K-Nearest Neighbors)
# sınıflandırma modellerinin ölçeklenmiş eğitim verisiyle eğitilmesi.

#Logistic Regression Modeli
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(C = 1, max_iter = 100, random_state = 42)
log_reg.fit(X_train_standard, y_train)

val_acc = log_reg.score(X_val_standard, y_val)
print(f"\nLogistic Regression Validation Accuracy: {val_acc}")

#KNN Modeli
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

k_accuracy = []
k_values = []
for k in range(2, 15):
    knn_k = KNeighborsClassifier(n_neighbors=k)
    knn_k.fit(X_train_standard, y_train)
    y_pred = knn_k.predict(X_val_standard)
    k_accuracy.append(accuracy_score(y_val, y_pred))
    k_values.append(k)

plt.plot(k_values, k_accuracy, marker = "o")
plt.xlabel("k_values")
plt.ylabel("k_accuracy")
plt.title("Farklı k Değerleri için Validation Accuracy")
plt.show()

# En iyi k değeriyle final KNN modeli (k=5)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_standard, y_train)

y_pred_val = knn.predict(X_val_standard)

accuracy = accuracy_score(y_val, y_pred_val)
print(f"\nKNN Validation Accuracy: {accuracy}")

conf_matrix = confusion_matrix(y_val, y_pred_val)
print(f"\nConfision Matrix Validation: \n{conf_matrix}")

# 8. MODEL SEÇİMİ VE VALİDASYON DEĞERLENDİRMESİ
# ==============================================================================
# Modellerin aşırı öğrenme (overfitting) durumunu kontrol etmek ve en yüksek 
# genelleme performansına sahip mimariyi belirlemek amacıyla Validation kümesi 
# başarım oranlarının (Accuracy) karşılaştırılması.

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("\n================TEST KÜMESİ NİHAİ DEĞERLENDİRMESİ ================")
print("""
MODELLERİN KARŞILAŞTIRILMASI VE SEÇİMİ:
Yapılan validasyon testleri sonucunda;
- Logistic Regression modeli: %90.4 doğruluk (accuracy)
- KNN (k=5) modeli: %84.9 doğruluk (accuracy)
elde etmiştir. Logistic Regression, doğrusallık varsayımıyla bu veri setindeki
örüntüleri daha iyi yakalamış ve daha yüksek bir genelleme başarısı göstermiştir.
Bu nedenle, test verisi üzerinde nihai değerlendirme yapmak üzere
Logistic Regression modeli seçilmiştir.
\n""")

y_pred_log_test = log_reg.predict(X_test_standard)
log_test_acc = accuracy_score(y_test, y_pred_log_test)
print(f"1. Logistic Regression Test Accuracy: {log_test_acc:.4f}")

conf_matrix_test = confusion_matrix(y_test, y_pred_log_test)
print("--- Logistic Regression Confusion Matrix (Test) ---")
print(conf_matrix_test)

cls_report = classification_report(y_test, y_pred_log_test)
print("\n--- Logistic Regression Sınıflandırma Raporu (Test) ---")
print(cls_report)


print("\n================ SONUÇLARIN YORUMLANMASI ================")
print("""
Bu projede Logistic Regression ve KNN modelleri eğitilmiş ve karşılaştırılmıştır.
Validasyon aşamasında Lojistik Regresyon (%90.4) KNN'den (%84.9) daha başarılı 
olduğu için nihai model olarak seçilmiştir.

Ancak, seçilen Lojistik Regresyon modelinin test verisi üzerindeki Sınıflandırma
Raporu incelendiğinde önemli bir durum tespit edilmiştir:
Model "kalan" müşterileri (Churn=0) mükemmel tahmin ederken, "ayrılan" 
müşterileri (Churn=1) bulmada zorlanmış ve Recall (Duyarlılık) değeri %52'de kalmıştır.

Bunun temel sebebi algoritmanın yetersizliği değil, VERİ SETİNDEKİ SINIF DENGESİZLİĞİDİR
(334 Kalan vs. 166 Ayrılan). Model, kayıp/hata fonksiyonunu minimize etmek için 
çoğunluk sınıfı olan 0'ları öğrenmeye odaklanmış ve 1'leri gözden kaçırmıştır. 
Gelecek çalışmalarda bu modeller özelinde dengesizliği gidermek adına, veri setine daha fazla 'Churn=1' 
(ayrılan müşteri) örneği toplanarak eklenebilir veya azınlık sınıfı için yapay (sentetik) 
veri üretme teknikleri kullanılarak modelin bu sınıfı öğrenmesi ve yakalama başarısı 
(Recall) artırılabilir.
""")