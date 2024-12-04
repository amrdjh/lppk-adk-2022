import pandas as pd

file_path = "C:\\Users\\Amir\\Documents\\Kuantitatif\\REVISI DATA CODING_LPPK_KELAS B.xlsx"
rawdata_df = pd.read_excel(file_path, sheet_name='Rawdata')
listvariables_df = pd.read_excel(file_path, sheet_name='ListVariables')

cleaned_data = rawdata_df.iloc[2:].reset_index(drop=True)
cleaned_data.columns = rawdata_df.iloc[1]  # nama kolom dari baris ke-2
cleaned_data = cleaned_data.loc[:, ~cleaned_data.columns.isna()]  # Hapus kolom kosong

print("============================")
print(cleaned_data.isnull().sum()) # Cek missing value
print("============================")
print(cleaned_data.info())  # Tipe data setiap kolom
print("============================")
print(cleaned_data.describe())  # Statistik dasar

stres_columns = [col for col in cleaned_data.columns if col.startswith('A.')]
stres_items = cleaned_data[stres_columns].astype(float)

dukungan_columns = [col for col in cleaned_data.columns if col.startswith('B.')]
dukungan_items = cleaned_data[dukungan_columns].astype(float)

konflik_columns = [col for col in cleaned_data.columns if col.startswith('C.')]
konflik_items = cleaned_data[konflik_columns].astype(float)

from pingouin import cronbach_alpha

print("==== Stres Akademik ====")
cleaned_data['Stres_Akademik'] = cleaned_data[stres_columns].astype(float).mean(axis=1)
alpha, _ = cronbach_alpha(stres_items)
print(f"Cronbach's Alpha: {alpha}")
print("==== Dukungan Sosial ====")
cleaned_data['Dukungan_Sosial'] = cleaned_data[dukungan_columns].astype(float).mean(axis=1)
alpha, _ = cronbach_alpha(dukungan_items)
print(f"Cronbach's Alpha: {alpha}")
print("==== Konflik Peran ====")
cleaned_data['Konflik_Peran'] = cleaned_data[konflik_columns].astype(float).mean(axis=1)
alpha, _ = cronbach_alpha(konflik_items)
print(f"Cronbach's Alpha: {alpha}")

cleaned_data['Stres_Akademik'] = stres_items.mean(axis=1)
cleaned_data['Dukungan_Sosial'] = dukungan_items.mean(axis=1)
cleaned_data['Konflik_Peran'] = konflik_items.mean(axis=1)

import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(cleaned_data['Stres_Akademik'], kde=True, bins=20)
plt.title('Distribusi Skor Stres Akademik')
plt.xlabel('Skor Stres')
plt.ylabel('Frekuensi')
plt.show()

sns.histplot(cleaned_data['Dukungan_Sosial'], kde=True, bins=20)
plt.title('Distribusi Skor Dukungan Sosial')
plt.xlabel('Skor Stres')
plt.ylabel('Frekuensi')
plt.show()

sns.histplot(cleaned_data['Konflik_Peran'], kde=True, bins=20)
plt.title('Distribusi Skor Konflik Peran')
plt.xlabel('Skor Stres')
plt.ylabel('Frekuensi')
plt.show()

cleaned_data['Jenis Kelamin'] = cleaned_data['Jenis Kelamin'].replace({1: 'Laki-laki', 2: 'Perempuan'})
sns.boxplot(x='Jenis Kelamin', y='Stres_Akademik', data=cleaned_data)
plt.title('Stres Akademik Berdasarkan Jenis Kelamin')
plt.show()

cleaned_data['Jenis Kelamin'] = cleaned_data['Jenis Kelamin'].replace({1: 'Laki-laki', 2: 'Perempuan'})
sns.boxplot(x='Jenis Kelamin', y='Dukungan_Sosial', data=cleaned_data)
plt.title('Dukungan Sosial Berdasarkan Jenis Kelamin')
plt.show()

cleaned_data['Jenis Kelamin'] = cleaned_data['Jenis Kelamin'].replace({1: 'Laki-laki', 2: 'Perempuan'})
sns.boxplot(x='Jenis Kelamin', y='Konflik_Peran', data=cleaned_data)
plt.title('Konflik Peran Berdasarkan Jenis Kelamin')
plt.show()

from scipy import stats

# Uji normalitas Stres Akademik
stat, p_value = stats.shapiro(cleaned_data['Stres_Akademik'])
print(f"Shapiro-Wilk Test untuk Stres Akademik: Statistik={stat}, p-value={p_value}")

# Uji normalitas Dukungan Sosial
stat, p_value = stats.shapiro(cleaned_data['Dukungan_Sosial'])
print(f"Shapiro-Wilk Test untuk Dukungan Sosial: Statistik={stat}, p-value={p_value}")

# Uji normalitas Konflik Peran
stat, p_value = stats.shapiro(cleaned_data['Konflik_Peran'])
print(f"Shapiro-Wilk Test untuk Konflik Peran: Statistik={stat}, p-value={p_value}")

#Sommersd Dukungan Sosial
import numpy as np

# Fungsi untuk menghitung Somers' D secara manual
def somers_d_manual(x, y):
    n = len(x)
    concordant = 0
    discordant = 0
    
    # Menghitung concordant dan discordant pairs
    for i in range(n):
        for j in range(i + 1, n):
            if (x[i] < x[j] and y[i] < y[j]) or (x[i] > x[j] and y[i] > y[j]):
                concordant += 1
            elif (x[i] < x[j] and y[i] > y[j]) or (x[i] > x[j] and y[i] < y[j]):
                discordant += 1
    
    # Menghitung Somers' D
    if (concordant + discordant) != 0:
        return (concordant - discordant) / (concordant + discordant)
    else:
        return np.nan  # Menghindari pembagian dengan 0 jika tidak ada pasangan yang valid

# Data untuk Stres Akademik dan Konflik Peran
stres_akademik = cleaned_data['Stres_Akademik'].dropna().values
dukungan_sosial = cleaned_data['Dukungan_Sosial'].dropna().values

# Menghitung Somers' D
somersd_result = somers_d_manual(stres_akademik, dukungan_sosial)

# Menampilkan hasil
print(f"Somers' D antara Stres Akademik dan Dukungan Sosial: {somersd_result}")

#Sommersd Konflik Peran
import numpy as np

# Fungsi 
def somers_d_manual(x, y):
    n = len(x)
    concordant = 0
    discordant = 0
    
    # Menghitung concordant dan discordant pairs
    for i in range(n):
        for j in range(i + 1, n):
            if (x[i] < x[j] and y[i] < y[j]) or (x[i] > x[j] and y[i] > y[j]):
                concordant += 1
            elif (x[i] < x[j] and y[i] > y[j]) or (x[i] > x[j] and y[i] < y[j]):
                discordant += 1
    
    # Menghitung Somers' D
    if (concordant + discordant) != 0:
        return (concordant - discordant) / (concordant + discordant)
    else:
        return np.nan  

# Data untuk Stres Akademik dan Konflik Peran
stres_akademik = cleaned_data['Stres_Akademik'].dropna().values
konflik_peran = cleaned_data['Konflik_Peran'].dropna().values

# Menghitung Somers' D
somersd_result = somers_d_manual(stres_akademik, konflik_peran)

# Menampilkan hasil
print(f"Somers' D antara Stres Akademik dan Konflik Peran: {somersd_result}")


from scipy import stats

# Uji t untuk perbandingan Stres Akademik antara Jenis Kelamin
t_stat, p_value = stats.ttest_ind(cleaned_data[cleaned_data['Jenis Kelamin'] == 'Pria']['Stres_Akademik'],
                                  cleaned_data[cleaned_data['Jenis Kelamin'] == 'Wanita']['Stres_Akademik'])
print(f"T-test Stres Akademik: Statistik t={t_stat}, p-value={p_value}")

# Korelasi antar variabel
correlation = cleaned_data[['Stres_Akademik', 'Dukungan_Sosial', 'Konflik_Peran']].corr()
print(correlation)

# PairPlot
sns.pairplot(cleaned_data[['Stres_Akademik', 'Dukungan_Sosial', 'Konflik_Peran']])
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

correlation_matrix = cleaned_data[['Stres_Akademik', 'Dukungan_Sosial', 'Konflik_Peran']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Heatmap Korelasi')
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
cleaned_data['Jenis Kelamin'] = cleaned_data['Jenis Kelamin'].replace({1: 'Laki-laki', 2: 'Perempuan'})
g = sns.FacetGrid(cleaned_data, col="Jenis Kelamin")
g.map(sns.scatterplot, 'Stres_Akademik', 'Dukungan_Sosial')
plt.show()

# 3D Scatter Plot
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

X = cleaned_data[['Stres_Akademik', 'Dukungan_Sosial', 'Konflik_Peran']]
X_scaled = StandardScaler().fit_transform(X)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2])
ax.set_xlabel('Stres Akademik')
ax.set_ylabel('Dukungan Sosial')
ax.set_zlabel('Konflik Peran')
plt.show()

# Model regresi dengan robust standard errors
import statsmodels.api as sm
X = cleaned_data[['Dukungan_Sosial', 'Konflik_Peran']]
X = sm.add_constant(X)
y = cleaned_data['Stres_Akademik']

model = sm.OLS(y, X)
results = model.fit(cov_type='HC3')  # Menggunakan robust standard errors (HC3)

print(results.summary())
