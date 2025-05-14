# Credit Risk API

Bu proje, kredi riski tahmini yapmak için geliştirilmiş bir Python Flask API'sidir. Model, eğitim verisiyle eğitilmiş ve sonrasında API üzerinden tahminler almak için kullanılmaktadır.

## Kurulum

1. **Depoyu klonlayın veya dosyaları indirin.**
2. **Gerekli kütüphaneleri yükleyin:**
   ```bash
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Kullanım

### API'yi Başlatmak

```bash
python train_model.py
python app.py
```

### Tahmin Almak

`/predict` endpoint'ine POST isteği göndererek kredi riski tahmini alabilirsiniz.

#### Örnek JSON Gönderimi

```json
{
  "person_age": 30,
  "person_income": 20000,
  "person_emp_length": 0,
  "loan_amnt": 10000,
  "loan_int_rate": 0.25,
  "loan_percent_income": 0.4,
  "loan_grade": "D",
  "loan_intent": "PERSONAL",
  "person_home_ownership": "RENT",
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 10
}
```

#### Yanıt

```json
{
  "prediction": 0
}
```

`prediction` değeri:

- `0`: Temerrütsüz geri ödeyebilir
- `1`: Temerrüte düşme olasılığı var

## Model Eğitimi

Modeli yeniden eğitmek için:

```bash
python train_model.py
```

Bu işlem sonunda `model.pkl` ve `encoder.pkl` dosyaları güncellenir.

## Dosya Açıklamaları

- `app.py`: Flask API uygulaması
- `train_model.py`: Model eğitim ve kaydetme scripti
- `requirements.txt`: Gerekli Python paketleri
- `model.pkl`: Eğitilmiş model dosyası

## Notlar

- API'nin çalışabilmesi için `model.pkl` ve `encoder.pkl` dosyalarının mevcut olması gerekir.
- Girdi formatı eğitimde kullanılan veriyle uyumlu olmalıdır.

---

Herhangi bir sorunla karşılaşırsanız, lütfen Python ortamınızı ve bağımlılıkları kontrol edin.
