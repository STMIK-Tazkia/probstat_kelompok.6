```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Membaca data dari file CSV
df = pd.read_csv("mental_health_students.csv")

# Tampilkan 5 data teratas
print("Data awal:\n", df.head())

# Cek jumlah data dan kolom
print("\nJumlah baris dan kolom:", df.shape)

# Hitung jumlah kasus mental health
mental_issues = ['Depression', 'Anxiety', 'PanicAttack']
for issue in mental_issues:
    count = df[issue].value_counts()
    print(f"\nJumlah mahasiswa dengan {issue}:")
    print(count)

# Distribusi CGPA Grade berdasarkan Depression
plt.figure(figsize=(10, 6))
sns.countplot(x='CGPA_Grade', hue='Depression', data=df)
plt.title("Distribusi CGPA Grade Berdasarkan Kondisi Depression")
plt.xlabel("Nilai CGPA")
plt.ylabel("Jumlah Mahasiswa")
plt.legend(title='Depression')
plt.show()

# Distribusi Mahasiswa yang Mencari Bantuan (SeekTreatment)
plt.figure(figsize=(8, 5))
sns.countplot(x='SeekTreatment', hue='Anxiety', data=df)
plt.title("Kondisi Anxiety dan Upaya Mencari Pengobatan")
plt.xlabel("Mencari Pengobatan?")
plt.ylabel("Jumlah Mahasiswa")
plt.legend(title='Anxiety')
plt.show()

# Korelasi antara mental health dengan CGPA
mental_cols = ['Depression', 'Anxiety', 'PanicAttack', 'SeekTreatment']
cgpa_map = {
    '0 - 1.99': 1.99,
    '2.00 - 2.49': 2.25,
    '2.50 - 2.99': 2.75,
    '3.00 - 3.49': 3.25,
    '3.50 - 4.00': 3.75
}
df['CGPA_Numeric'] = df['CGPA'].map(cgpa_map)

correlation = df[mental_cols + ['CGPA_Numeric']].corr()
print("\nKorelasi antara Mental Health dan CGPA:\n", correlation)

# Heatmap korelasi
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title("Korelasi Mental Health dengan CGPA")
plt.show()
```
