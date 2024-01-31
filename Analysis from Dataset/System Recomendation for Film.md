# Laporan Proyek Machine Learning - Kivlan Khair Wijayanto

## Domain Proyek
Pada Projek **Machine Learning Terapan : membuat model Sistem Rekomendasi**, saya menggunakan dataset berdomain Hiburan mengenai Film.

Saat ini pengguna dapat mengakses dan menonton film dengan mudah tanpa perlu pergi ke bioskop, namun dengan memanfaatkan jaringan internet layanan penyedia film berbasis *subscription* seperti Netflix atau Disney+ pengguna dapat menonton film dimana saja dan kapan saja. 

Namun layanan penyedia film berbasis *subscription*  semakin ramai dan persaingan semakin ketat. Oleh sebab itu penyedia  layanan film, harus bisa menyediakan film-film yang  disukai penggunanya dan merekomendasikannya dengan baik agar kualitas konten dan pelayanan. 
[Jumlah Streaming Service yang Tersedia](https://www.cnbc.com/2021/07/05/streaming-services-compared-revenue-arpu-for-netflix-disney-more.html)
## Business Understanding
Dengan semakin pentingnya penyediaan konten yang menarik untuk pengguna layanan maka dibutuhkan sebuah sistem rekomendasi yang dapat memberi rekomendasi dengan baik dan akurat kepada pengguna sehingga pengguna dapat menikmati konten-konten yang disediakan oleh penyedia layanan film.
### Problem Statements
- Rekomendasi seperti apa yang dapat diberikan kepada pengguna?

### Goals
- Sistem rekomendasi dapat memberi rekomendasi berdasarkan judul film yang mereka sukai

### Solution statements
Langkah-langkah yang dilakukan dalam  menyelesaikan proyek ini:.
- Persiapan data
  Pada bagian ini data yang sudah dibersihkan akan diolah kembali untuk memudahkan proses pemodelan sistem rekomendasi:
    - Membersihkan Missing Value
    - Merapihkan Nama Genre
    - Menghapus Data Duplikat 
- Pemodelan
  Sistem rekomendasi yang akan dibuat project ini berbasis konten (*content-based filtering*) berdasarkan genre yang dapat memberi rekomendasi berdasar item yang mirip dengan item yang disukai pengguna sebelumnya. Pada proyek ini akan dibuat sebuah sistem rekomendasi *content-based filtering* dengan algoritma *cosine similarity* yang mengukur kemiripan antara dua buah vektor item dan kesamaan arahnya. 

## Data Understanding
Dataset yang digunakan adalah dataset [*Movie Recommender System Dataset*](https://www.kaggle.com/gargmanas/movierecommenderdataset) dari situs Kaggle yang berisi data seputar film-film yang telah ditayangkan beserta genrenya. 

Apabila dilakukan Data Loading adalah sebagai berikut
![info](https://github.com/plannnn/assetspenting/blob/main/assetMLT/MLT_loading.jpg?raw=true)<br>
Variabel pada dataset ini adalah sebagai berikut.
- title: Judul dari Film
- movieID: Berisi ID film tersebut

Mengecek genre pada dataset<br>
![info](https://github.com/plannnn/assetspenting/blob/main/assetMLT/MLT_Genre.jpg.png?raw=true)<br>

Melihat Jumlah Film dan Genre<br>
![info](https://media.discordapp.net/attachments/894146882232786944/903204701397463110/unknown.png)<br>
#### Visualisasi Data
Apabila Genre pada dataset dibentuk Tabel <br>
![info](https://github.com/plannnn/assetspenting/blob/main/assetMLT/MLT_genres.png?raw=true)<br>
## Data Preparation
Pada bagian Data Preparation dilakukan proses sebagai berikut:
- Menghapus Data Duplikat
    Apabila terdapat data duplikat, dapat digunakan kode sebagai berikut.
    ```movies = movies.drop_duplicates('movieId')```
- Membersihkan Missing Value
    Apabila terdapat data yang tidak memiliki nilai dapat didrop <br>
    ```movies.isnull().sum()``` <br>
    ![info](https://cdn.discordapp.com/attachments/894146882232786944/903205207255691275/unknown.png)<br>
- Merapihkan Nama Genre
    Apabila dilihat dari genres diatas, terdapat jenis genre Sci-Fi, dan Film Noir. Untuk mempermudah kita merubah nama dari genre 'Sci-Fi' menjadi 'scifi', dan merubah genre ' Film Noir' menjadi 'Noir'
## Modeling
Pada Proyek yang dibuat, digunakan model algoritma *cosine similarity*
##### Cosine Similarity
Model ini dibangun dengan menghitung  *cosine similarity* dari setiap item yang ada pada dataset. Hal dilakukan dengan menggunakan fungsi [cosine_similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html) dari SKLearn. 
Untuk membangun model pertama-tama melakukan perhitungan *cosine similarity* terhadap dataset kemudian menyimpan hasil perhitungannya. Setelah itu dibuat sebuah fungsi untuk memberikan rekomendasi film yang mirip dan diurutkan berdasakan nilai perhitungan *cosine similarity*-nya. Berikut ini potongan kode dari fungsi tersebut:
```
from sklearn.metrics.pairwise import cosine_similarity
# Menghitung cosine similarity pada matrix tf-idf
cosine_sim = cosine_similarity(movie_final) 
cosine_sim_df = pd.DataFrame(cosine_sim, index=movies['title'], columns=movies['title'])
print('Shape:', cosine_sim_df.shape)
cosine_sim_df.sample(5, axis=1).sample(10, axis=0)
```
Fungsi untuk menghasilkan rekomendasi
```python
def movie_recommendations(nama_film, similarity_data=cosine_sim_df, items=movies[['title', 'genres']], k=5):
        index = similarity_data.loc[:,nama_film].to_numpy().argpartition(
            range(-1, -k, -1))
        # Mengambil data dengan similarity terbesar dari index yang ada
        closest = similarity_data.columns[index[-1:-(k+2):-1]]
        
        # Drop nama_film agar nama film tsb tidak muncul dalam daftar rekomendasi
        closest = closest.drop(nama_film, errors='ignore')
        return pd.DataFrame(closest).merge(items)
```

## Evaluation
Untuk mengevaluasi, model menghasilkan rekomendasi dari film.

![info](https://github.com/plannnn/assetspenting/blob/main/assetMLT/MLT_hasil.jpg?raw=true)<br>

