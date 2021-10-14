# Laporan Proyek Machine Learning - Kivlan Khair Wijayanto

## Domain Proyek
Pada Projek **Machine Learning Terapan Pertama: membuat model Predictive Analysis**, saya menggunakan dataset berdomain ekonomi mengenai prediksi harga mobil Mercedes Benz C-Class.

Latar Belakang pemilihan topik ini adalah dikarenakan ingin melihat tingkat penjualan mobil bekas, dimana dalam kasus ini mercedes C-Class, dengan fitur-fitur tertentu yang dapat mempengaruhi nilai di pasar. 
## Business Understanding
Penting bagi para pemilik mobil jika ingin menjual mobilnya melihat harga yang terdapat di pasaran, namun untuk para penjual cukup sulit untuk menentukan harga mobilnya agar mendapatkan harga yang sesuai keinginannya dan juga dapat terjual dengan mudah, oleh karena itu pembuatan prediksi harga yang cocok penting.
### Problem Statements
 - Berapa harga mobil bekas dengan jenis transmisi, jarak tempuh dan ukuran mesin tertentu?
 - Dari Karakteristik yang tersedia manakah yang paling berpengaruh?
### Goals
Membuat model Machine Learning yang dapat memberi prediksi harga mobil bekas C-Class dengan karakteristik yang tersedia

### Solution statements
Solusi yang diajukan antara lain adalah KNN dan Random Forest.
Dengan pengertian:
- **KNN/K-Nearest Neighbor**. Algoritma ini merupakan supervised learning dan mengelompokkan suatu label dengan cara mencari kesesuaian dengan tetangga terdekat. KNN dipilih karena merupakan algoritma yang cocok dengan kasus regresi
- **RF/Random Forest**. Algoritma ini adalah supervised learning dan dapat menyelesaikan masalah regresi. Random Forest ini merupakan model yang terdiri dari beberapa model dan bekerja secara bersama-sama dan tiap model membuat prediksi secara independen dan digabungkan untuk membuat prediksi akhir. 
- **Boosting Algorithm**. Algoritma ini memiliki tujuan untuk meningkatkan performa dan akurasi prediksi, dengan cara menggabungkan model sedeharan dan "lemah" sehingga membantuk suatu model yang kuat.

## Data Understanding
Dataset yang digunakan adalah dataset [*100,000 UK Used Car Data set*](https://www.kaggle.com/adityadesai13/used-car-dataset-ford-and-mercedes?select=cclass.csv) dari situs Kaggle yang berisi data tentang mobil bekas yang terjual di UK dengan variable, mileage, model, engineSize, year, transmission, fuelType, dan price yang menjadi label pada data ini. Dataset ini memiliki 3899 dengan 7 kolom dengan 2 kategorikal dan 5 numerikal.

Variabel pada dataset ini adalah sebagai berikut.
- model : Jenis mobil yang terjual
- *mileage* : Jarak Tempuh Mobil
- *engine Size*: Ukuran Volume Bahan Bakar dalam satuan liter
- *year* : Tahun Pembelian Mobil
- *transmission* : Jenis Setelan Transmisi
- *fuel Type* : Jenis Bahan Bakar yang digunakan
- *Price* : Harga dalam kurs euro

Apabila dilakukan Data Loading adalah sebagai berikut
|id|Model|year|price|transmission|mileage|fuelType| engineSize|
|--|:----:|----|-----|:------------:|-------:|:--------:|-----------:|
|0|C-Class|2020|30495|Automatic|1200|Diesel|2.0|
|1|C-Class|2020|29989|Automatic|1000|Petrol|1.5|
|2|C-Class|2020|37899|Automatic|500|Diesel|2.0|
|3|C-Class|2019|30399|Automatic|5000|Diesel|2.0|
|4|C-Class|2019|29899|Automatic|4500|Diesel|2.0|
|...|...|...|...|...|...|...|...|
|3894|C-Class|2017|14700|Manual|31357|Diesel|1.6|
|3895|C-Class|2018|18500|Automatic|28248|Diesel|2.1|
|3896|C-Class|2014|11900|Manual|48055|Diesel|2.1|
|3897|C-Class|2014|11300|Automatic|49865|Diesel|2.1|
|3898|C-Class|2014|14800|Automatic|55445|Diesel|2.1|

Dataset tersebut juga dapat dilihat deskripsi statistiknya seperti berikut:
|Jenis|year|price|mileage|engineSize|
|---|---|---|---|---|
|count|	3899.000000|3899.000000|3899.000000|3899.000000|
|mean|2017.338548|23674.286997|22395.709156|2.037394|
|std|2.213416|8960.218218|22630.438426|0.487769|
|min|1991.000000|1290.000000|1.000000|0.000000|
|25%|2016.000000|17690.000000|6000.000000|2.000000|
|50%|2018.000000|22980.000000|14640.000000|2.000000|
|75%|2019.000000|28900.000000|32458.500000|2.100000|
|max|2020.000000|88995.000000|173000.000000|6.200000|

Dikarenakan terdapat satu jenis yang memiliki nilai 0, maka data tersebut dihilangkan karena data tersebut hanya 1 yang memiliki nilai 0, dan karena penggunaan model semuanya C-Class maka dapat dihilangkan.

Jenis Data diatas dapat dibedakan menjadi dua kategori:
Kategorikal : Transmission dan FuelType
Numerikal : Year, Price, Mileage, engineSize, dan Price

#### Visualisasi Data
Apabila Jenis data dikategorikan seperti diatas dapat dilihat bentuk tabel dan grafik masing-masing data sebagai berikut:
|Jenis transmisi|Jumlah Sampel|Persentase|
|:---:|:---:|:---:|
|Semi-Auto|2071|53.1|
|Automatic|1628|41.8|
|Manual|198|5.1|
|Other|1|0.0|

![Tabel Transmisi](https://raw.githubusercontent.com/plannnn/assetspenting/main/assetMLT/MLT_transmission.png)
|Jenis Fuel|Jumlah Sampel|Persentase|
|:---:|:---:|:---:|
|Diesel|2339|60.0|
|Petrol|1402|36.0|
|Hybrid|151|3.9|
|Other|6|0.2|

![Tabel Fuel](https://github.com/plannnn/assetspenting/blob/main/assetMLT/MLT_fuelType.png?raw=true)
Data Numerikal
![Tabel Fuel](https://github.com/plannnn/assetspenting/blob/main/assetMLT/MLT_numerikal.png?raw=true)

Correlation Matrix
![Tabel Fuel](https://github.com/plannnn/assetspenting/blob/main/assetMLT/MLT_correlation.png?raw=true)
Dapat dilihat dari *Correlation Matrix* dengan fitur target *price*, memiliki nilai korelasi yang cukup besar dengan fitur-fitur lainnya 
## Data Preparation
Sebelum dataset melalui proses training, model sebelumnya perlu melalui proses pemisahan antara data latih dan data test lalu melakukan scaling
#### Train-Test Split
Proses pembagian dataset menjadi data latih *(train)* dan data uji *(test)* merupakan hal yang saya pilih untuk lakukan sebelum membuat model. Hal ini karena data uji berperan sebagai data baru yang benar-benar belum pernah dilihat oleh model sebelumnya sehingga informasi yang terdapat pada data uji tidak mengotori informasi yang terdapat pada data latih, alasan lain mengapa menggunakan *train test split* karena untuk efisiensi dan tidak melakukan *data leakage* ketika melakukan scaling. 
#### Standardisasi
Data numerik yang terdapat di dataset akan dilakukan **Standardisasi** sehingga menghasilkan distribusi dengan nilai standar deviasi 1 dan mean 0. Hal tersebut dilakukan dengan tujuan untuk meningkatkan peforma algoritma machine learning dan membuatnya konvergen lebih cepat
```python
numerical_features = ['year', 'mileage', 'engineSize']
scaler = StandardScaler()
scaler.fit(x_train[numerical_features])
x_train[numerical_features] = scaler.transform(x_train.loc[:, numerical_features])
x_train[numerical_features].head()
```
## Modeling
Pada Proyek yang dibuat, digunakan model algoritma *Machine Learning* yaitu **K-Nearest Neighbours**, **Random Forest**, dan **Boosting Algorithm**. Model tersebut dipilih dikarenakan permasalahan dari model *Machine Learning* yang dibuat adalah permasalahan regresi. hasil dari model yang dipilih akan dibandingkan berdasarkan label yang telah terpilih sebelumnya yaitu *price*. Berikut adalah potongan kode dari model tersebut.
```python
# KNN
knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(x_train, y_train)
y_pred_knn = knn.predict(x_train)
# RF
RF = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1, min_samples_split=3)
RF.fit(x_train, y_train)

models.loc['train_mse','RandomForest'] = mean_squared_error(y_pred=RF.predict(x_train), y_true=y_train) 
#Adaptive
boosting = AdaBoostRegressor(n_estimators=50, learning_rate=0.05, random_state=55)                             
boosting.fit(X_train, y_train)
models.loc['train_mse','Boosting'] = mean_squared_error(y_pred=boosting.predict(X_train), y_true=y_train)
```
## Evaluation
Untuk mengevaluasi model digunakan metrik **MSE (Mean Squarred Error)**

#### Mean Squarred Error
MSE, menghitung selisih rata-rata nilai sebenarnya dengan nilai prediksi.

|Jenis|train|test|
|---|---|---|
|KNN|6485.57|7809.62|
|RF|2218.91|8659.94|
|Boosting|12004.1|11898.7|

![akurasi](https://github.com/plannnn/assetspenting/blob/main/assetMLT/MLT_akurasiPT1.png?raw=true)
Nilai Error yang muncul dari *KNN*,*RF*, dan *Boosting* dapat dilihat diatas, dengan nilai error yang terlihat di *RF* yang paling bagus.

Apabila dilakukan prediksi pada *KNN*, *RF*, *Boosting*
|id|y_true|prediksi_KNN|prediksi_RF|prediksi_Boosting|
|:---:|:---:|:---:|:---:|:---:|
|3682|12450|15166.5|13470.0|18896.5|

Dari Tabel dapat dilihat bahwa nilai *RF* lebih mendekati dengan nilai aslinya, sehingga model yang paling cocok adalah *RF* atau *Random Forest*
Setelah melakukan Modeling dan prediksi, dapat dipilih model yang memiliki nilai prediksi paling mendekati dengan harga yang terjual. Berdasarkan hal tersebut, model *Random Forest* yang memiliki nilai prediksi paling mendekati dengan harga terjual.
