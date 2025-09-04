import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Muat Dataset ---
try:
    # Membaca file CSV dengan pemisah (delimiter) titik koma (;)
    df = pd.read_csv('Dataset.csv', delimiter=';')
    print("Dataset berhasil dimuat.")
    print("\n5 baris pertama dari data asli:")
    print(df.head())
except FileNotFoundError:
    print("Error: File 'Dataset.csv' tidak ditemukan. Pastikan file berada di direktori yang sama dengan skrip ini.")
    exit()

# --- 2. Pra-pemrosesan Data (Preprocessing) ---
# Membuat salinan dataframe agar data asli tidak berubah
df_encoded = df.copy()
encoders = {} # Dictionary untuk menyimpan encoder setiap kolom

# Mengubah setiap kolom yang bukan angka menjadi angka (encoding)
for column in df_encoded.select_dtypes(include=['object', 'bool']).columns:
    # 'Timestamp' tidak akan digunakan sebagai fitur, jadi kita lewati
    if column == 'Timestamp':
        continue
    le = LabelEncoder()
    df_encoded[column] = le.fit_transform(df_encoded[column])
    encoders[column] = le # Simpan encoder untuk referensi nanti

print("\n5 baris pertama dari data setelah di-encode:")
print(df_encoded.head())

# --- 3. Tentukan Fitur (X) dan Target (y) ---
# Kolom target yang ingin diprediksi adalah 'Depression'
# Kolom 'Timestamp' dan 'CGPA_Grade' dihapus dari fitur
# 'CGPA_Grade' dihapus karena informasinya sudah terwakili oleh kolom 'CGPA'
X = df_encoded.drop(columns=['Timestamp', 'Depression', 'CGPA_Grade'])
y = df_encoded['Depression']

print(f"\nKolom Target (y): 'Depression'")
print(f"Kolom Fitur (X): {list(X.columns)}")

# --- 4. Bagi Dataset menjadi Data Latih dan Data Uji ---
# 80% data untuk melatih model, 20% untuk menguji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nJumlah data latih: {len(X_train)} baris")
print(f"Jumlah data uji: {len(X_test)} baris")


# --- 5. Latih Model Naive Bayes ---
# Menggunakan CategoricalNB karena semua fitur kita bersifat kategori
model = CategoricalNB()

# Melatih model menggunakan data latih
model.fit(X_train, y_train)
print("\nModel Naive Bayes berhasil dilatih.")


# --- 6. Evaluasi Performa Model ---
# Membuat prediksi pada data uji
y_pred = model.predict(X_test)

# Mengambil label asli dari encoder untuk laporan yang lebih mudah dibaca
# Contoh: 0 -> FALSE, 1 -> TRUE
try:
    target_names_labels = encoders['Depression'].inverse_transform(np.unique(y_test))
    target_names_display = [f"Tidak Depresi ({label})" for label in target_names_labels]
except KeyError:
    # Jika kolom target sudah numerik dari awal
    target_names_display = ['Kelas 0', 'Kelas 1']


# Menghitung akurasi
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAKURASI MODEL: {accuracy:.2%}")

# Menampilkan laporan klasifikasi (presisi, recall, f1-score)
print("\nLaporan Klasifikasi:")
print(classification_report(y_test, y_pred, target_names=target_names_display))

# Menampilkan Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Visualisasi Confusion Matrix menggunakan heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=[f"Prediksi {name}" for name in target_names_labels],
            yticklabels=[f"Aktual {name}" for name in target_names_labels])
plt.title('Confusion Matrix Hasil Prediksi')
plt.ylabel('Label Aktual')
plt.xlabel('Label Prediksi')
plt.show()
