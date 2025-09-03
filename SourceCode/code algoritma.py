import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix
import numpy as np

# ==============================================================================
# BAGIAN 1: MEMBACA DAN MEMPERSIAPKAN DATA
# ==============================================================================

# Membaca data dari file CSV"
try:
    df = pd.read_csv("Dataset.csv", sep=';')
except Exception as e:
    print(f"Error membaca file dengan separator ';': {e}")
    print("Mencoba membaca dengan separator ','...")
    df = pd.read_csv("Dataset.csv")

print("Data berhasil dimuat. Memulai persiapan data untuk Machine Learning...")

# Mendefinisikan variabel target yang akan diprediksi
TARGET = 'Depression'

# Membuat pemetaan untuk kolom CGPA
cgpa_map = {
    '0 - 1.99': 1.0,
    '2.00 - 2.49': 2.25,
    '2.50 - 2.99': 2.75,
    '3.00 - 3.49': 3.25,
    '3.50 - 4.00': 3.75
}

# Mengonversi semua kolom boolean/string boolean menjadi format numerik (1/0)
for col in ['Gender', 'MaritalStatus', 'Depression', 'Anxiety', 'PanicAttack', 'SeekTreatment']:
    if col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(lambda x: 1 if x.strip().lower() in ['female', 'true'] else 0)
        elif df[col].dtype == 'bool':
            df[col] = df[col].astype(int)

# Menambahkan kolom CGPA_Numeric ke dataframe
if 'CGPA' in df.columns:
    df['CGPA_Numeric'] = df['CGPA'].map(cgpa_map)

# Mendefinisikan fitur (X) dan target (y)
# Fitur yang tidak relevan atau menyebabkan kebocoran data akan dibuang
features_to_drop = ['Timestamp', TARGET, 'Anxiety', 'PanicAttack', 'CGPA', 'CGPA_Grade']
X = df.drop(columns=features_to_drop, errors='ignore')
y = df[TARGET]

# Mengidentifikasi kolom kategorikal dan numerik untuk preprocessing
categorical_features = ['Course', 'StudyYear']
numerical_features = ['Age', 'CGPA_Numeric']

# Membuat preprocessor untuk mentransformasi kolom
# OneHotEncoder untuk data kategorikal, StandardScaler untuk data numerik
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough' # Biarkan kolom lain (spt Gender, MaritalStatus) apa adanya
)

# Memisahkan data menjadi data latih dan data uji (80% latih, 20% uji)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nUkuran data latih (X_train): {X_train.shape}")
print(f"Ukuran data uji (X_test): {X_test.shape}")


# ==============================================================================
# BAGIAN 2: ALGORITMA 1 - LINEAR REGRESSION (UNTUK KLASIFIKASI)
# ==============================================================================
print("\n--- Melatih Model: Linear Regression ---")

# Membuat pipeline: Preprocessing -> Model
lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('regressor', LinearRegression())])

# Melatih model
lr_pipeline.fit(X_train, y_train)

# Membuat prediksi (hasilnya adalah nilai kontinu)
y_pred_lr_raw = lr_pipeline.predict(X_test)

# Mengonversi hasil prediksi menjadi kelas biner (0 atau 1) dengan threshold 0.5
y_pred_lr = (y_pred_lr_raw > 0.5).astype(int)

# Evaluasi model
mse = mean_squared_error(y_test, y_pred_lr_raw)
accuracy_lr = accuracy_score(y_test, y_pred_lr)

print(f"\nMean Squared Error (MSE) dari prediksi mentah: {mse:.4f}")
print(f"Akurasi setelah thresholding (> 0.5): {accuracy_lr:.4f}")
print("\nLaporan Klasifikasi (Linear Regression):")
print(classification_report(y_test, y_pred_lr, zero_division=0))


# ==============================================================================
# BAGIAN 3: ALGORITMA 2 - STOCHASTIC GRADIENT DESCENT (SGD)
# ==============================================================================
print("\n--- Melatih Model: Stochastic Gradient Descent (SGD) Classifier ---")

# Membuat pipeline: Preprocessing -> Model
# loss='log_loss' membuat SGDClassifier bekerja seperti Regresi Logistik
sgd_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', SGDClassifier(loss='log_loss', random_state=42, max_iter=1000, tol=1e-3))])

# Melatih model
sgd_pipeline.fit(X_train, y_train)

# Membuat prediksi (hasilnya sudah dalam bentuk kelas 0 atau 1)
y_pred_sgd = sgd_pipeline.predict(X_test)

# Evaluasi model
accuracy_sgd = accuracy_score(y_test, y_pred_sgd)
print(f"\nAkurasi: {accuracy_sgd:.4f}")

print("\nLaporan Klasifikasi (SGD Classifier):")
print(classification_report(y_test, y_pred_sgd, zero_division=0))

print("\nConfusion Matrix (SGD Classifier):")
cm = confusion_matrix(y_test, y_pred_sgd)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Depression', 'Depression'], yticklabels=['No Depression', 'Depression'])
plt.title('Confusion Matrix - SGD Classifier')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
