import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

df= pd.read_csv('./credit_risk_dataset.csv')

# OneHotEncoder kullanımı
encoder = OneHotEncoder(sparse_output=False)
encoded_data = encoder.fit_transform(df[['loan_grade', 'loan_intent', 'person_home_ownership', 'cb_person_default_on_file']])

# Yeni sütun isimlerini almak için
encoded_columns = encoder.get_feature_names_out(['loan_grade', 'loan_intent', 'person_home_ownership', 'cb_person_default_on_file'])

# Yeni DataFrame oluşturma
encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns)

# Orijinal DataFrame ile birleştirme
df = df.join(encoded_df)
df = df.drop(['loan_grade', 'loan_intent', 'person_home_ownership', 'cb_person_default_on_file'], axis=1)

print(df.columns)

# Hedef değişkenin sınıf dağılımını inceleme
print("Orijinal veri setindeki sınıf dağılımı:")
print(df['loan_status'].value_counts())

# Özellikler ve hedef değişken
X = df.drop('loan_status', axis=1)
y = df['loan_status']

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Undersampling ile dengeleme
undersampler = RandomUnderSampler(random_state=42)
X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)

# Dengeleme sonrası sınıf dağılımını inceleme
print("Random Undersampling sonrası eğitim setindeki sınıf dağılımı:")
print(pd.Series(y_train_resampled).value_counts())