# Customer Churn Prediction Pipeline 📊

## Projenin Amacı:

Bu projenin temel amacı, müşteri davranışlarına ait verileri analiz ederek şirketi terk etme (churn) riski taşıyan müşterileri makine öğrenmesi algoritmalarıyla önceden tahmin etmektir. 

Proje kapsamında veri ön işleme, eksik veri yönetimi, öznitelik mühendisliği (feature engineering) ve ölçekleme adımlarından oluşan uçtan uca bir makine öğrenmesi boru hattı (pipeline) kurulmuştur. Tahminleme için **Lojistik Regresyon** ve **K-Nearest Neighbors (KNN)** algoritmaları eğitilmiş ve performansları karşılaştırılmıştır.

## Repository İçeriği:

- `churn_prediction.py`: Veri işleme, model eğitimi ve değerlendirme adımlarını içeren ana Python betiği.
- `musteri_veri_seti.csv`: Modellerin eğitildiği ve test edildiği müşteri veri seti.
- `requirements.txt`: Projenin bağımlılıklarını ve kütüphane sürümlerini içeren dosya.
- `README.md`: Proje dokümantasyonu.

## Nasıl Çalıştırılır?

Projeyi kendi yerel ortamınızda (local) çalıştırmak için aşağıdaki adımları izleyebilirsiniz:

1. **Repoyu Klonlayın veya İndirin:**
   Proje dosyalarını bilgisayarınıza indirin ve terminal üzerinden dosya dizinine (klasöre) gidin.

2. **Gerekli Kütüphaneleri Yükleyin:**
   Modelin çalışması için gereken bağımlılıkları tek seferde kurmak için şu komutu çalıştırın:
   ```bash
   pip install -r requirements.txt
   ```
3. **Projeyi Çalıştırın**
   Ortam hazırlandıktan sonra ana script'i çalıştırarak tüm pipeline'ı başlatabilirsiniz:
   ```bash
   python churn_prediction.py
   ```

## Sonuç ve Değerlendirme Özeti

* **Model Seçimi:** Yapılan validasyon testleri sonucunda; Lojistik Regresyon modeli (%90.4), KNN modeline (%84.9) göre daha yüksek genelleme başarısı göstermiş ve nihai test modeli olarak seçilmiştir.
* **Sınıf Dengesizliği Etkisi:** Seçilen Lojistik Regresyon modelinin test verisi üzerindeki metrikleri incelendiğinde, sistemde kalmaya devam eden müşterileri (Churn=0) tahmin etmede mükemmel bir performans sergilediği görülmüştür.
* **Çıkarım:** Buna karşın, ayrılan müşterileri (Churn=1) yakalama oranı (Recall) %52 seviyesinde kalmıştır. Bu durumun algoritmanın yetersizliğinden değil, veri setindeki belirgin sınıf dengesizliğinden (334 Kalan vs. 166 Ayrılan) kaynaklandığı tespit edilmiştir.
* **Gelecek Çalışmalar:** Bir sonraki aşamada sentetik veri üretme (SMOTE) veya ağırlıklandırılmış sınıflandırma (Class Weights) teknikleri kullanılarak azınlık sınıfına ait Recall (Duyarlılık) değerinin artırılması hedeflenmektedir.

---
**Geliştirici:** Samet Ozan Topcu