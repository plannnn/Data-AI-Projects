# Laporan Proyek Machine Learning - Kivlan Khair Wijayanto

## Project Overview

Pada Projek **Machine Learning Terapan : membuat model Sistem Rekomendasi**, saya menggunakan dataset berdomain Hiburan mengenai Anime.

Saat ini penonton dapat mengakses dan menonton *series* Anime dengan mudah tanpa perlu pergi ke bioskop, namun beberapa penonton tidak hanya ingin menonton, melainkan ingin mempelajari hal yang baru selama menonton, salah satu yang dapat dipelajari selama menonton adalah mempelajari bahasa. Dengan dibuatnya sebuah sistem rekomendasi dapat membantu para penonton anime untuk mempelajari bahasa jepang dengan menonton anime sesuai dengan genre yang disukai penonton 
[Penggunaan Anime untuk mempelajari Bahasa Jepang](https://files.eric.ed.gov/fulltext/EJ1142396.pdf)

## Business Understanding

Dengan semakin banyaknya konten yang menarik untuk pengguna layanan maka dibutuhkan sebuah sistem rekomendasi yang dapat memberi rekomendasi dengan baik dan akurat kepada pengguna sehingga pengguna dapat menikmati konten-konten yang disediakan oleh penyedia layanan.

### Problem Statements

- Rekomendasi seperti apa yang dapat diberikan kepada pengguna?

### Goals

- Sistem rekomendasi dapat memberi rekomendasi berdasarkan judul Anime yang mereka sukai

### Solution Approach

Langkah-langkah yang dilakukan dalam  menyelesaikan proyek ini:

- Persiapan data
  Pada bagian ini data yang sudah dibersihkan akan diolah kembali untuk memudahkan proses pemodelan sistem rekomendasi:

  - Menghapus *Missing Value* pada data
  - Melakukan perubahan kata
  - Menghapus data duplikat
    - Konversi label kolom menjadi *one-hot encoding*
    - Standarisasi label numerik

- Pemodelan
  Sistem rekomendasi yang akan dibuat project ini berbasis konten (*content-based filtering*) berdasarkan genre, type, jumlah episode, dan rating yang dapat memberi rekomendasi berdasar item yang mirip dengan item yang disukai penonton lainnya. Keuntungan dari *content-base filtering* adalah model tidak perlu data dari penonton lain, sehingga rekomendasi dapat dilakukan ke penonton secara spesifik, namun kekurangan dari *content-base filtering* adalah model hanya dapat memberikan rekomendasi dari content yang sebelumnya disukai penonton, sehingga tidak dapat memberikan rekomendasi ke hal lain. [Content Base Filtering]([Content-based Filtering Advantages & Disadvantages (google.com)](https://developers.google.com/machine-learning/recommendation/content-based/summary))

  Pada proyek ini akan dibuat sebuah sistem rekomendasi *content-based filtering* dengan algoritma *cosine similarity* yang mengukur kemiripan antara dua buah vektor item dan kesamaan arahnya, dan algortima K-Nearest Neighbors, karena cocok pada kasus clustering di sistem rekomendasi dan dapat mengelompokan item yang mirip

## Data Understanding

Dataset yang digunakan adalah dataset [*Anime Recommendations Dataset*](https://www.kaggle.com/CooperUnion/anime-recommendations-database) dari situs Kaggle yang berisi data seputar Anime yang pernah ditayangakan

Apabila dilakukan Data Loading adalah sebagai berikut
![info](https://cdn.discordapp.com/attachments/841304133868191794/903894352013717504/unknown.png)<br>

- Variabel pada dataset ini adalah sebagai berikut.
  - Kolom <code>anime_id</code> berisi id dari anime
  - Kolom <code>name</code> berisi judul anime
  - Kolom <code>genre</code> berisi genre dari anime tersebut
  - Kolom <code>type</code> berisi jenis publikikasi dari anime 
  - Kolom <code>episode</code> berisi jumlah episode yang ditayangkan
  - Kolom <code>rating</code> berisi nilai rating rata-rata dari situs myanimelist.com
  - Kolom <code>members</code> berisi jumlah anggota komunitas yang ada di dalam anime 

- Tipe Data dari tiap kolom

![info](https://cdn.discordapp.com/attachments/841304133868191794/903896570255261716/unknown.png)

1. anime_id bertipe integer

2. name bertipe object

3. genre bertipe object

4. type bertipe object

5. episodes bertipe object

6. rating bertipe float 

7. members bertipe integer<br>
   <br>

- Melihat Jumlah Anime dan Genre<br>
  ![info](https://cdn.discordapp.com/attachments/841304133868191794/903897480754761738/unknown.png)<br>

- Dari bagian Data understanding dapat dilihat pengertian members. Data tersebut dapat didrop karena tidak merepresentasikan data dengan baik.

![info](https://media.discordapp.net/attachments/841304133868191794/903911174133940235/Gmdw8wV1Vzw66bZAdwoareAt5mMKIqSVLvOIYqSdpoXmdw7HnSW4AzgMPL3ONq0nOApsZ3Bu5nHXvB15M8idwmcG9kJIk9U6qFk7SSJIkSZI2OsdQJUmSJEkthkVJkiRJUothUZIkSZLUYliUJEmSJLUYFiVJkiRJLYZFSZIkSVKLYVGSJEmS1PI3q4vZhtDgjkAAAAASUVORK5CYII.png?width=433&height=411)

​	

#### Visualisasi Data

- Distribusi Type pada dataset <br>
  ![info](https://cdn.discordapp.com/attachments/841304133868191794/903905298434105344/unknown.png)<br>

- Melihat Distribusi Genre pada dataset

![info](https://cdn.discordapp.com/attachments/841304133868191794/903906041585086474/unknown.png)

- Melihat distribusi episodes pada dataset

  ![info](https://cdn.discordapp.com/attachments/841304133868191794/903914668073025546/unknown.png)

  Dapat dilihat dari grafik diatas, terdapat jumlah episode *'unknown'*, data tersebut dapat didrop.

  ## Data Preparation

  - Menghapus *Missing Value* 

  ![info](https://cdn.discordapp.com/attachments/841304133868191794/903898199209685012/unknown.png)

  ​		Dapat dilihat dari gambar, bahwa terdapat *missing value* di dataset, sehingga dapat dihapus karena dataset lebih dari 10000.

  - Melakukan Perubahan kata

    ![info](https://cdn.discordapp.com/attachments/841304133868191794/903898803583717396/unknown.png)

    Dapat dilihat dari genre yang tersedia, terdapat genre 'Sci-Fi', dan genre 'Slice of Life'. Untuk mempermudah proses berikutnya, nama tersebut dapa diubah menjadi Scifi dan Slice.

  - Memeriksa data duplikat

    Mengecek duplikasi data dan menghapusnya bila ada. Hal ini dilakukan agar tidak ada dua rekomendasi item yang sama persis. Proses ini dilakukan dengan menggunakan *method* `drop_duplicate` pada dataframe.

  - Menghapus kolom yang tidak diperlukan

    Dari bagian data understanding, dapat dilihat pengertian dari *members*. Data tersebut dapat di drop karena tidak merepresentasikan dataset dengan baik. 

  - Merubah label kategori dengan *one-hot encoding*. Proses ini dilakukan terhadap semua data kategorikal supaya memudahkan pencarian nilai terdekat dari setiap kategori.

  - Standarisasi label numerik. Standarisasi dilakukan agar rentang nilai pada data numerik bernilai 0-1 sehingga dapat mempercepat pelatihan model *machine learning*.

## Modeling

Pada Proyek yang dibuat, digunakan model *machine learning* sistem rekomendasi *content based* dengan algoritam K-Nearest Neighbors dan Cosine Similarity

##### K-Nearest Neighbors

Model dibangung dengan menggunakan  [NearestNeighbor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html), dengan parameter metriks 'manhattan', dengan alasan, dalam proses clustering, penggunaan iterasi lebih sedikit dibandingkan dengan *euclidean*<sup>[[1]]([Different Types of Distance Metrics used in Machine Learning | by Kunal Gohrani | Medium](https://medium.com/@kunal_gohrani/different-types-of-distance-metrics-used-in-machine-learning-e9928c5e26c7))</sup>. Model sebelumnya dilatih terlebih dahulu pada dataset dengan <code>fit</code>, lalu dibuat fungsi untuk memberikan rekomendasi Anime apabila penonton menyukai suatu Anime. 
Contoh Hasil rekomendasi 
![info](https://cdn.discordapp.com/attachments/841304133868191794/903926914522165248/unknown.png)

##### Cosine Similarity

Model ini dibangun dengan menghitung  *cosine similarity* dari setiap item yang ada pada dataset. Hal dilakukan dengan menggunakan fungsi [cosine_similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html) dari SKLearn. Untuk membangun model pertama-tama melakukan perhitungan *cosine similarity* terhadap dataset kemudian menyimpan hasil perhitungannya. Setelah itu dibuat sebuah fungsi untuk memberikan rekomendasi film yang mirip dan diurutkan berdasakan nilai perhitungan *cosine similarity*-nya. 

Contoh hasil rekomendasi dari film.

![info](https://cdn.discordapp.com/attachments/841304133868191794/903927145267625994/unknown.png)<br>



# Evaluasi

Untuk mengevaluasi model digunakan metrik *Similarity* yang diperoleh dari cosine similarity dan metrik *manhattan* untuk KNN. Pada tahap ini juga akan mengevaluasi data testing beserta menampilkan nilai similaritynya. Pertama-tama, dipilih aplikasi secara acak.

```python
id = 89
anime_titleID.loc[[id]]
```

Kemudia akan ditampilkan judul Anime sesuai dengan id yeng terpilih

![info](https://media.discordapp.net/attachments/841304133868191794/903963069028401192/unknown.png)

#### Similarity 

Similarity adalah ukuran seberapa mirip dua objek data. Ukuran similarity adalah metrik yang memanfaatkan besar jarak dengan dimensi yang mewakili fitur objek. Pada fungsi sistem rekomendasi, getRecommendedApps sudah dilengkapi dengan nilai similarity.  Sehingga, sesuai dengan id yang dipilih di atas, didapatkan:

- Similarity berdasarkan KNN metrick *manhattan*

  ![info](https://cdn.discordapp.com/attachments/841304133868191794/903963144223883274/unknown.png)

- Similarity berdasarkan *cosine similarity*

  ![info](https://cdn.discordapp.com/attachments/841304133868191794/903963214545563678/unknown.png)

#### Dengan Metriks *Silhouette Coefficient*

Tingkat klustering pada data dapat dievaluasi dengan menggunakan metrik Silhouette Coefficient. Berikut persamaan dari metriks :

![info](https://cdn.discordapp.com/attachments/841304133868191794/903967480098201621/silhoutte.png)

Dengan nilai a sebagai rata-rata jarak antara sampel dan titik lainnya di kelas yang sama, dan nilai b sebagai rata-rata jarak antara sampel dan titik lainnya pada kelas terdekat.

Skala nilai di metrik *Silhouette Coefficient* bernilai -1 sampai 1, dengan nilai -1 menunjukkan clustering buruk dan skor 1 menunjukkan clustering yang bagus, dan nilai 0 menunjukkan adanya overlapping. [*Silhoutte Coefficient*](https://www.sciencedirect.com/science/article/pii/0377042787901257)

Nilai yang didapat adalah -0.0002. Skor tersebut menunjukan masih ada overlaping cluster pada data.

## Penutup

Sistem model rekomendasi Anime telah berhasil dibuat dengan menggunakan model KNN dan Cosine Similiarity. Model yang dibuat dapat memberikan 3 rekomendasi Anime yang mirip dengan yang ditonton oleh pengguna dengan cukup baik. Meskipun pada contoh notebook model hanya memberikan 3 rekomendasi teratas namun model dapat merekomendasikan lebih dari angka 3 (misal 10 - 15) hanya saja  tingkat kemiripan dan relevansi semakin berkurang. Bila dilihat dari tingkat kesamaan model KNN dapat memberikan hasil yang lebih baik daripada model cosine similiarity.  Kedepannya model dapat dititingkatkan lagi dengan membuat clustering yang lebih baik karena cluster di model masih terdapat overlap seperti yang ditunjukan oleh skor *Silhouette Coefficient*.
