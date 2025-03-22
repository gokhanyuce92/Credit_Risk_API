import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

df= pd.read_csv('./credit_risk_dataset.csv')

# person_age: Kişinin yaşı
# person_income: Kişinin geliri
# person_home_ownership: Kişinin ev sahipliği durumu
# person_emp_length: Kişinin işte çalışma süresi
# loan_intent: Kredi amacı

# loan_grade: Kredi notu
# (A: Mükemmel - Bu kredi notuna sahip kişilerin kredi geçmişi çok iyidir 
# ve kredi yükümlülüklerini yerine getirme olasılıkları çok yüksektir. 
# B: Çok İyi - Bu kredi notuna sahip kişilerin kredi geçmişi iyidir ve 
# kredi yükümlülüklerini yerine getirme olasılıkları yüksektir. 
# C: İyi - Bu kredi notuna sahip kişilerin kredi geçmişi ortalamadır ve 
# kredi yükümlülüklerini yerine getirme olasılıkları orta düzeydedir. 
# D: Zayıf - Bu kredi notuna sahip kişilerin kredi geçmişi zayıftır ve 
# kredi yükümlülüklerini yerine getirme olasılıkları düşüktür.)

# loan_amnt: Kredi miktarı
# loan_int_rate: Kredi faiz oranı
# loan_status: Kredi durumu (0 temerrütsüz geri ödeyebilir, 1 temerrüte düşme olasılığı var)
# loan_percent_income: Kredi miktarının gelire oranı
# cb_person_default_on_file: Kişinin kredi geçmişinde temerrüt durumu
# cb_person_cred_hist_length: Kişinin kredi geçmişi uzunluğu

# data preprocessing
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

# Null değerleri ortalama ile doldurma
df = df.fillna(df.mean())

# Korelasyon matrisi oluşturma
corr_matrix = df.corr()

# Korelasyon matrisini yazdırma
print(corr_matrix)

# Korelasyon matrisini görselleştirme
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Korelasyon Matrisi')
plt.show()

print(df['loan_status'].value_counts())

X=df.drop(['loan_status'],axis=1)
y=df['loan_status']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20, random_state=42)

# SMOTE ile dengeleme
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Modeli eğit
# rf_clf = RandomForestClassifier(min_samples_leaf=1,
#                                 min_samples_split=2,
#                                 n_estimators=200,
#                                 random_state=42).fit(X_train_resampled, y_train_resampled)
# y_train_pred = rf_clf.predict(X_train_resampled)

rf_clf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator = rf_clf, param_grid = param_grid, cv = 5, n_jobs = -1, verbose = 2)
grid_search.fit(X_train, y_train)

print(f"Best parameters found: {grid_search.best_params_}")

# En iyi parametrelerle modeli yeniden eğit
# best_rf_clf = grid_search.best_estimator_
# best_rf_clf.fit(X_train, y_train)

# Modeli test et
# y_pred = rf_clf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))

# # Çapraz doğrulama ile performans değerlendirme
scores = cross_val_score(rf_clf, X, y, cv=5, scoring='f1')
print("Cross-Validation F1 Scores:", scores)
print("Mean F1 Score:", scores.mean())

# Modeli kaydet
with open('model.pkl', 'wb') as f:
    pickle.dump(rf_clf, f)

# Encoder'ı kaydetme
with open('encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

print("Model trained and saved as model.pkl")    