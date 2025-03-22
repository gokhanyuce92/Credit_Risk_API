import pandas as pd
from sklearn import preprocessing

# Veri setini yükle
df = pd.read_csv('./credit_risk_dataset.csv')

print(df[(df["person_age"] < 0) | (df["person_age"] > 100)])
print(df[(df["person_emp_length"] >= 50) & (df["person_emp_length"].notnull())]["person_emp_length"])

# Yaş aralığına göre veri temizleme (0'dan küçük veya 100'den büyük değerleri temizleme)
def clean_age_range(df, column, min_age=0, max_age=100):
    return df[(df[column] >= min_age) & (df[column] <= max_age)]
# person_emp_length için 50 ve üzeri kayıtları çıkaran fonksiyon
def clean_high_and_null_emp_length(df, column, threshold=50):
    return df[(df[column] < threshold) & (df[column].notnull())]

df = clean_age_range(df, 'person_age')
df = clean_high_and_null_emp_length(df, 'person_emp_length')

label_encoder=preprocessing.LabelEncoder()
# sayısal olmayan değerleri sayısal değerlere dönüştürme
df['loan_grade_num']=label_encoder.fit_transform(df['loan_grade'])
print("\n'loan_grade' kolonundaki kategorik değerlerin sayısal karşılıkları:")
for category, value in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)):
    print(f"{category}: {value}")
    
df['loan_intent_num']=label_encoder.fit_transform(df['loan_intent'])
print("\n'loan_intent_num' kolonundaki kategorik değerlerin sayısal karşılıkları:")
for category, value in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)):
    print(f"{category}: {value}")

df['person_home_ownership_num']=label_encoder.fit_transform(df['person_home_ownership'])
print("\n'person_home_ownership' kolonundaki kategorik değerlerin sayısal karşılıkları:")
for category, value in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)):
    print(f"{category}: {value}")

df['cb_person_default_on_file_num']=label_encoder.fit_transform(df['cb_person_default_on_file'])
print("\n'cb_person_default_on_file' kolonundaki kategorik değerlerin sayısal karşılıkları:")
for category, value in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)):
    print(f"{category}: {value}")

# Veri setinin genel bilgilerini görüntüle
print("Veri setinin genel bilgileri1:")
print(df.info())

# Yaşı 100'den büyük 0'dan küçük olan kayıtları tespit etme
outliers_age = df[(df['person_age'] > 100) | (df['person_age'] < 0)]
print("\nYaşı 100'den büyük 0'dan küçük olan kayıtlar:")
print(outliers_age)

# Veri setinin genel bilgilerini görüntüle
print("Veri setinin genel bilgileri:")
print(df.info())

# Eksik değerlerin sayısını görüntüle
print("\nEksik değerlerin sayısı:")
print(df.isnull().sum())

# Sayısal sütunların istatistiksel özetini görüntüle
print("\nSayısal sütunların istatistiksel özeti:")
print(df.describe().T) 

# Veri setinin kaç adet feature içerdiğini görüntüle
num_features = df.shape[1]
print(f"\nVeri seti {num_features} adet feature içeriyor.")