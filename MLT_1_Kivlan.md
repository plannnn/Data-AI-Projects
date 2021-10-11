# Laporan Proyek Machine Learning - Kivlan Khair Wijayanto

## Domain Proyek
Pada Projek **Machine Learning Terapan Pertama: membuat model Predictive Analysis**, saya menggunakan dataset berdomain ekonomi mengenai prediksi harga mobil Mercedes Benz C-Class.

Latar Belakang pemilihan topik ini berdasarkan penggunaan mobil yang semakin cenderung untuk ke listrik, sehingga penggunaan mobil yang menggunaan bahan bakar alami akan semakin menurun. Oleh karena itu, dengan dibuatnya model prediksi ini, kita dapat memperkirakan harga mobil Mercedes Benz C-Class apabila dijual sebagai mobil bekas. 
## Business Understanding
Seiring dengan makin banyaknya jumlah mobil yang ada di muka bumi, penjualan mobil bekas seharusnya menjadi semakin banyak agar jumlah mobil di jalanan tidak semakin banyak. Oleh karena itu, dibuat **model prediksi harga mobil bekas** untuk para pemilik mobil dengan niatan untuk mempermudah dan memberi kecocokan dengan harga yang tersedia di pasaran.
### Problem Statements
 - Berapa harga mobil bekas dengan jenis transmisi, jarak tempuh dan ukuran mesin tertentu?
 - Dari Karekteristik yang tersedia manakah yang paling berpengaruh?
### Goals
Membuat model Machine Learning yang dapat memberi prediksi harga mobil bekas dengan karekteristik yang tersedia

### Solution statements
Solusi yang diajukan antara lain adalah KNN dan Random Forest, yang kemudian dipilih berdasarkan tingkat akurasi yang paling tinggi.
Sebagai contoh:
- **KNN/K-Nearest Neighbor**. Algoritma ini merupakan supervised learning dan mengelompokan suatu label dengan cara mencari kesesuaian dengan tetangga terdekat. KNN dipilih karena merupakan algoritma yang cocok dengan kasus regresi
- **RF/Random Forest**. Algoritma ini adalah supervised learning dan dapat menyelesaikan masalah regresi. Random Forest ini merupakan model yang terdiri dari beberapa model dan bekerja secara bersama-sama dan tiap model membuat prediksi secara independen dan digabungkan untuk membuat prediksi akhir. 

## Data Understanding
Dataset yang digunakan adalah dataset [*100,000 UK Used Car Data set*](https://www.kaggle.com/adityadesai13/used-car-dataset-ford-and-mercedes?select=cclass.csv) dari situs Kaggle yang berisi data tentang mobil bekas yang terjual di UK dengan variable, mileage, model, engineSize, year, transmission, fuelType, dan price yang menjadi label pada data ini. Dataset ini memiliki 3899 dengan 7 kolom dengan 2 kategorikal dan 5 numerikal.

Variabel pada dataset ini adalah sebagai berikut.
- model : C-Class
- mileage : Jarak Tempuh Mobil
- engineSize: Ukuran Volume Bahan Bakar
- year : Tahun Pembelian Mobil
- transmission : Jenis Setelan Transmisi
- fuelType : Jenis Bahan Bakar
- Price : Harga 

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
Kategorikal: Transmission dan FuelType
Numerikal: Year, Price, Mileage, engineSize, dan Price

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
## Data Preparation
Sebelum dataset melalui proses training, model sebelumnya perlu melalui proses pemisahan antara data latih dan data test lalu melakukan scaling
#### Train-Test Split
Proses pembagian dataset menjadi data latih *(train)* dan data uji *(test)* merupakan hal yang harus dilakukan sebelum membuat model. Hal ini karena data uji berperan sebagai data baru yang benar-benar belum pernah dilihat oleh model sebelumnya sehingga informasi yang terdapat pada data uji tidak mengotori informasi yang terdapat pada data latih.

Proses Train-Test Split ini perlu dilakukan karena tidak akan ada cukup data dalam set data pelatihan bagi model untuk mempelajari pemeetaan input ke output yang efektif. Juga tidak akan ada cukup data dalam set pengujian untuk mengevaluasi kinerja model secara efektif. Terutama untuk dataset size yang terbilang cukup besar, proses ini sangat membantu computational efficiency. Selain itu dalam proses transformasi data uji dan data latih harus dilakukan secara terpisah. Pada dataset yang dipakai pembagian antara data latih dan data uji adalah rasio 80:20, Sehingga dari total 3079 jumlah sampel yang ada, 2463 sampel merupakan data latih dan 616 sampel merupakan data uji.
#### Standardisasi
Data numerik yang terdapat di dataset akan dilakukan **Standardisasi** sehingga menghasilkan distribusi dengan nilai standar deviasi 1 dan mean 0. Hal tersebut dilakukan dengan tujuan untuk meningkatkan peforma algoritam machine learning dan membuatnya konvergen lebih cepat
```python
numerical_features = ['year', 'mileage', 'engineSize']
scaler = StandardScaler()
scaler.fit(x_train[numerical_features])
x_train[numerical_features] = scaler.transform(x_train.loc[:, numerical_features])
x_train[numerical_features].head()
```
## Modeling
Pada Projek yang dibuat, digunakan model algoritma *Machine Learning* yaitu **K-Nearest Neighbours**, dan **Random Forest**. Model tersebut dipilih dikarenakan permasalahan dari model *Machine Learning* yang dibuat adalah permasalahan regresi. hasil dari model yang dipilih akan dibandingkan berdasarkan label yang telah terpilih sebelmunya yaitu *price*. Berikut adalah potongan kode dari model tersebut.
```python
# KNN
knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(x_train, y_train)
y_pred_knn = knn.predict(x_train)
# RF
RF = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1, min_samples_split=3)
RF.fit(x_train, y_train)
 
models.loc['train_mse','RandomForest'] = mean_squared_error(y_pred=RF.predict(x_train), y_true=y_train)   
```
Setelah model dibuat dan dilakukan, dapat dilihat akurasi dari masing-masing model
```python
mse = pd.DataFrame(columns=['train', 'test'], index=['KNN','RF'])
model_dict = {'KNN': knn, 'RF': RF}
for name, model in model_dict.items():
    mse.loc[name, 'train'] = mean_squared_error(y_true=y_train, y_pred=model.predict(x_train))/1e3 
    mse.loc[name, 'test'] = mean_squared_error(y_true=y_test, y_pred=model.predict(x_test))/1e3
 
mse
```

|Jenis|train|test|
|---|---|---|
|KNN|6485.57|7809.62|
|RF|2218.91|8659.94|
![akurasi](https://github.com/plannnn/assetspenting/blob/main/assetMLT/MLT_akurasi.png?raw=true)
Nilai Error yang muncul dari *KNN* dan *RF* dapat dilihat diatas, dengan nilai error yang terlihat di *RF* yang paling bagus.

Apabila dilakukan prediksi pada *KNN* dan *RF*
|y_true|prediksi_KNN|prediksi_RF|
|:---:|:---:|:---:|
|3682|12450|15166.5|13470.0|
Dari Tabel dapat dilihat bahwa nilai *RF* lebih mendekati dengan nilai aslinya, sehingga model yang paling cocok adalah *RF* atau *Random Forest*
## Evaluation
Dari Hasil yang didapat dari model yang telah dilakukan, model yang memliki nilai prediksi paling mendekati dengan nilai aslinya adalah model ***Random Forest*** yang nilai prediksinya paling bagus. Dari hasil yang didapat, model ***Random Forest*** yang diterapkan dan dikembangkan.
