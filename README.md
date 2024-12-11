# Laporan Proyek Machine learning - Muhammad Khalish

## Domain Proyek
Domain yang dipilih pada proyek ini adalah **Agriculture**, dengan judul ***Predictive Analytics* : Rekomendasi Jenis Tanaman**.

### Latar Belakang
![image](https://github.com/user-attachments/assets/21074c2d-47cf-4b43-bfa4-c7ea4bb23bda)

Indonesia merupakan negara agraris tropis terbesar kedua setelah brazil dan salah satu penyumbang agrikultur terbesar di dunia. Indonesia memiliki beragam komoditas unggulan seperti beras, kopi, singkong dsb. Pada triwulan ketiga tahun 2023, sektor pertanian mencatat pertumbuhan sebesar 1,46% (yoy) dan memberikan kontribusi sebesar 13,57% terhadap Produk Domestik Bruto (PDB). Keberhasilan ini tidak terlepas dari peran petani yang bekerja keras untuk mendukung ketahanan pangan nasional[[1](https://www.kemenkeu.go.id/informasi-publik/publikasi/berita-utama/Sektor-Pertanian-Fokus-Utama-Pemerintah)]. Unsur hara tanah seperti kandungan nitrogen, fosfor, potasium dan musim yang mempengaruhi curah hujan dan kelembaban tanah serta pH merupakan salah satu faktor yang mempengaruhi kualitas dan produktivitas pertanian[[2](https://github.com/user-attachments/files/18051954/yusriel.%2BJournal%2Bmanager.%2BAwalia%2BArdiana%2B.120403010036.%2BAMAK.pdf)].
Penerapan machine learning untuk memprediksi kecocokan tanaman berdasarkan kecocokan tanah dan musim berdasarkan curah hujan dapat menjadi solusi untuk menjaga dan meningkatkan kualitas serta produktivitas agrikultur, khususnya pertanian dan perkebunan.

## Bussiness Understanding

### Problem Statement
Berdasarkan latar belakang diatas, rincian masalah yang dapat diselesaikan pada proyek ini sebagai berikut
- Bagaimana membuat model machine learning yang dapat memprediksi jenis tanaman yang cocok dengan tanah dan musim?
- Algoritma apa yang cocok untuk memprediksi jenis tanaman dengan akurasi yang tinggi?

### Goals
Tujuan Proyek ini adalah sebagai berikut.
- Mendapatkan model machine learning yang dapat memprediksi jenis tanaman.
- Mendapatkan model machine learning yang cocok dengan akurasi yang tinggi
  
### Solution Statement
Berikut solusi yang harus dilakukan untuk mendapatkan jawaban dari pertanyaan diatas.
- Menganalisis data dengan melakukan Exploratory Data Analysis (Univariate dan Multivariate Analysis).
- Membuat beberapa variasi model untuk membandingkan performa berdasarkan akurasi dalam memprediksi jenis tanaman. diantaranya sebagai berikut.
    * *Support Vector Machine* (SVM) merupakan algoritma *supervised learning* yang digunakan untuk klasifikasi dan regresi. Algoritma ini mencari hyperplane optimal yang memisahkan data dari kelas yang berbeda dengan margin maksimum[[3](https://ieeexplore.ieee.org/document/8369054)].
    * *Multi Layer Perceptron* (MLP) merupakan jenis jaringan saraf tiruan yang terdiri dari setidaknya tiga lapisan yakni input, hidden, dan output. Setiap neuron dalam satu lapisan terhubung ke neuron di lapisan berikutnya. MLP menggunakan fungsi aktivasi non-linear dan algoritma backpropagation untuk mempelajari pola dalam data, membuatnya mampu menangani masalah yang tidak linear[[4](https://arxiv.org/abs/2012.13796)].
    * *Random Forest* (RF) merupakan metode ensemble yang membangun sekumpulan pohon keputusan selama pelatihan dan outputnya adalah mode dari kelas (klasifikasi) atau rata-rata prediksi (regresi) dari pohon individu. Dengan menggabungkan beberapa pohon, RF meningkatkan akurasi prediksi dan mengurangi overfitting[[5](https://www.mdpi.com/2072-4292/16/4/665)]. 
    * *Gradient Boosting* (GB) merupakan teknik ensemble yang membangun model prediktif secara bertahap, biasanya dengan menambahkan pohon keputusan baru yang mengoreksi kesalahan dari model sebelumnya. Pendekatan ini mengoptimalkan fungsi loss dengan menggunakan algoritma gradient descent, sehingga model secara iteratif meningkatkan akurasinya[[6](https://f1000research.com/articles/11-1069)].

## Data Understanding

<div align='center'>
  Tabel 1. Informasi Dataset

| Jenis | Keterangan |
| ------ | ------ |
| Title | Crop Recommendation |
| Source | [Kaggle](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) |
| Maintainer | [Atharva Ingle ⚡](https://www.kaggle.com/atharvaingle) |
| License | Apache 2.0 |
| Visibility | Publik |
| Tags | Tabular, Agriculture, Recommender Systems |
| Usability | 7.06 |

</div>

Berikut langkah pada tahap data understanding.

- Pengumpulan data (*Data Gathering*) untuk mengumpulkan informasi dataset seperti jumlah baris dan kolom, tipe data dsb.
- Pemeriksaan data (*Data Accessing*) untuk menemukan apakah terdapat informasi data yang hilang (*missing value*), duplikasi data (*duplicated data*) atau terdapat *outlier*.
- Pembersihan data (*Data Cleaning*) untuk membersihkan data dari *missing value*, *duplicate data*, dan *outlier*. Pada tahap ini juga dilakukan encoding label untuk mengubah label data menjadi dalam bentuk data numerik agar dapat dikenali oleh model.

### Exploratory Data Analysis (EDA)

**EDA - Deskripsi Variabel**

Berikut adalah variabel yang terdapat pada dataset *crop recommendation*.
- `N` --> perbandingan kandungan nitrogen pada tanah
- `P` --> perbandingan kandungan fosfor pada tanah
- `K` --> perbandingan kandungan potasium pada tanah
- `temperature` --> temperatur pada satuan celsius
- `humidity` --> persentase kelembaban tanah
- `ph` --> nilai pH tanah
- `rainfall` --> intensitas hujan dalam mm
- `label` --> jenis tanaman yang cocok

Pemeriksaan yang dilakukan menggunakan fungsi `isna.sum()` dan `duplicated.sum()` tidak menemukan informasi data yang hilang dan duplikasi data, namun terdapat *outlier* pada fitur Phosphorus, Potassium, Temperatur, Humidity, pH dan Rainfall menggunakan visualisasi data boxplot. *Outlier* dapat diatasi dengan melakukan *dropping* menggunakan metode IQR (*Interquartile Range*). IQR dapat dihitung dengan menggunakan rumus berikut.

$$IQR = Q_3 - Q_1$$

- Q1 adalah kuartil pertama 
- Q3 adalah kuartil ketiga.

Setelah dilakukan metode IQR, jumlah data yang awalnya berjumlah `2200` menjadi `1768`.

**EDA - Univariate**

Distribusi data yang tidak seimbang terlihat pada grafik yang ditunjukkan oleh gambar 1. karakteristik data yang ditunjukkan oleh gambar sebagai berikut
- Data sampel Nitrogen berkisar antara 0 - 140 kg/ha
- Data sampel Phosphorus berkisar antara 5 - 145 kg/ha
- Data sampel Potassium berkisar antara 5 - 205 kg/ha
- Data sampel Temperatur berkisar antara 8.8 - 43.7 °C
- Data sampel kelembaban berkisar antara 14.3 - 99.98 %
- Data sampel pH tanah berkisar antara 3.5 - 9.9
- Data sampel Curah Hujan berkisar antara 20.2 - 298.6 mm
Berdasarkan informasi diatas, skala tiap karakteristik memiliki rentang yang berbeda-beda. Hal ini menunjukkan bahwa data belum dilakukan standarisasi.

<div align="Center">
  
![image](https://github.com/user-attachments/assets/8c0db3c4-889f-465a-8854-cf71d7553cbf)

Gambar 1. Analisis Univariate

</div>

**EDA - Multivariate**

Analisis Multivariate dilakukan dengan menggunakan *library seaborn* sehingga terlihat pada gambar 2 pola sebaran data yang acak. Dapat dilihat bahwa Phosphorus dan Potassium memiliki korelasi negatif dengan pH tanah yang menandakan bahwa semakin rendah kadar Phosporus dan Potassium maka semakin tinggi pH tanah (Basa).

<div align="Center">
  
![image](https://github.com/user-attachments/assets/59ed0ab5-a2b1-4820-be62-2e1231f6fc13)

Gambar 2. Analisis Multivariate

</div>

Hubungan antara parameter atau karakteristik data dapat dilihat pada matriks korelasi yang ditunjukkan pada gambar 3. Dan juga, matriks korelasi menampilkan nilai yang menjelaskan seberapa besar pengaruh suatu parameter dengan parameter lan seperti parameter kadar Phosphorus dengan Potassium yang bernilai `0.74` yang menandakan bahwa kedua parameter ini sangat mempengaruhi satu sama lain.

<div align="Center">
  
![image](https://github.com/user-attachments/assets/dbdf6309-911f-4013-a042-e844291b65d4)

Gambar 3. Matriks Korelasi

</div>

## Data Preparation

Berikut langkah pada tahap data preparation.
- Pembagian data menjadi data latih dan data uji (*Train Test Split*).
- Normalisasi data (*Normalization*) untuk mentransformasi data ke dalam skala yang seragam sehingga fitur memiliki rentang nilai yang sebanding).

Pada langkah *train test split*, data dibagi dengan perbandingan antara data uji dan data latih sebesar `20:80` dengan *random state* sebesar `123`. Kemudian dilakukan Normalisasi menggunakan *library sklearn.processing.MinMaxScaler*.

## Modeling

Terdapat 4 algoritma yang digunakan pada proyek ini, yakni

***Support Vector Machine***

*Support Vector Machine* (SVM) merupakan algoritma *supervised learning* yang digunakan untuk klasifikasi dan regresi dengan cara untuk mencari hyperplane terbaik yang memisahkan data ke dalam kelas yang berbeda.

kelebihan
- Efektif untuk dimensi tinggi
- Kinerja baik pada data kecil
- Flexibilitas kernel

Kekurangan
- Sulit ditangani pada dataset besar
- Pemilihan kernel
- Kurang robust terhadap outlier

***Multi Layer Perceptron***

*Multi Layer Perceptron* (MLP) merupakan jenis jaringan saraf tiruan (ANN) yang terdiri dari lapisan input, lapisan tersembunyi, dan lapisan output yang cocok untuk tugas klasifikasi dan regresi, terutama yang melibatkan pola non-linear.

kelebihan
- Kemampuan generalisasi
- Fleksibilitas
- Dukungan banyak framework

Kekurangan
- Butuh data besar untuk performa terbaik
- Rentan overfitting jika model terlalu kompleks atau data terlalu kecil
- Proses training lambat

***Random Forest***

*Random Forest* (RF) merupakan algoritma ensemble berbasis Decision Tree yang menggabungkan banyak pohon keputusan (*trees*)Setiap pohon dibangun menggunakan subset data yang berbeda, dan prediksi akhir dihasilkan dari rata-rata (regresi) atau voting (klasifikasi). Hyperparameter yang digunakan pada proyek ini yaitu `max_depth` sebesar 20 yang berfungsi untuk mengatur kedalaman maksimum tiap pohon keputusan dalam *Random Forest*.

kelebihan
- Robust terhadap overfitting.
- Mampu menangkap hubungan non-linear antara variabel.
- Skalabilitas dimana model dapat bekerja dengan baik pada dataset besar dan paralelisasi mudah dilakukan.

Kekurangan
- Kurang interpretasi
- Memori dan waktu
- Kurang optimal untuk data sangat tinggi

***Gradient Boosting***

*Gradient Boosting* (GB) merupakan metode ensemble yang membangun model secara bertahap dengan menambahkan pohon keputusan baru untuk mengurangi error residual dari model sebelumnya dimana tiap pohon berkontribusi untuk mengoreksi kesalahan model sebelumnya. Hyperparameter yang digunakan pada proyek ini, yaitu

- `learning_rate` sebesar 0.001 yang berfungsi untuk mengontrol besarnya langkah koreksi yang dilakukan oleh model baru (weak learner) terhadap kesalahan prediksi dari model sebelumnya.
- `n_estimator` sebesar 200 berfungsi untuk enentukan jumlah pohon keputusan (weak learners) yang akan dibangun dalam model Gradient Boosting.

kelebihan
- Kinerja tinggi: Sangat efektif untuk data terstruktur.
- Mengatasi bias dan varians: Dapat menyesuaikan kekuatan masing-masing model individu.
- Parameter tuning yang fleksibel: Banyak hyperparameter yang dapat disesuaikan untuk optimasi.

Kekurangan
- Proses training bisa lambat, terutama pada dataset besar.
- Rentan terhadap overfitting jika parameter tuning tidak hati-hati.
- Kompleksitas implementasi karena banyak hyperparameter yang harus diatur.

## Evaluasi

Metrik yang digunakan pada tahap ini adalah `accuracy_score`. Akurasi dapat dihitung dengan menggunakan rumus.

$$\text{Accuracy} = \frac{\text{TP + TN}}{\text{TN + TP + FN + FP}} \times 100\%$$

**Penjelasan**
- `TP` (*True Positive*): Jumlah data positif yang diprediksi dengan benar sebagai positif.
- `TN` (*True Negative*): Jumlah data negatif yang diprediksi dengan benar sebagai negatif.
- `FP` (*False Positive*): Jumlah data negatif yang diprediksi secara tidak benar sebagai positif (Kesalahan Tipe I).
- `FN` (*False Negative*): Jumlah data positif yang diprediksi secara tidak benar sebagai negatif (Kesalahan Tipe II).

Berikut nilai akurasi dari 4 model yang dilatih

<div align="Center">

  tabel 2. Matriks Korelasi

|       | train accuracy | test accuracy |
| ----- | ----- | ----- |
| Model |       |       |
| SVM | 0.98 | 0.99 |
| MLP | 0.98 | 0.99 |
| RF | 1.00 | 1.00 |
| GB | 0.96 | 0.91 |

![image](https://github.com/user-attachments/assets/ad17545d-dd3c-4d88-9eee-37cbcea1600c)

Gambar 4. Hasil Akurasi Model

</div>

Berdasarkan tabel 2 dan gambar 3 dapat dilihat bahwa nilai akurasi tertinggi dicapai dengan menggunakan algoritma *Random Forest* sebesar `100%`. Oleh karena itu, model yang dipilih adalah model dengan algoritma *random forest*. Hal ini menyelesaikan *prolem statement* dan memenuhi *goals* dari proyek ini. Selain tingkat akurasi, algoritma *random forest* dipilih dikarenakan robust terhadap overfitting, dapat bekerja dengan baik untuk dataset yang besar dan memiliki hubungan non-linear antar fitur. Berdasarkan hasil tersebut, diharapkan tingkat produktivitas agrikultur dibidang pertanian dapat meningkat dengan adanya proyek ini.

## Referensi
[1] https://www.kemenkeu.go.id/informasi-publik/publikasi/berita-utama/Sektor-Pertanian-Fokus-Utama-Pemerintah

[2] Ardiana, A., Prasetyo, A.Y.E. 2017. "Sistem Prediksi Penentuan Jenis Tanaman Sayuran Berdasarkan Kondisi Musim dengan Pendekatan Metode *Trend Moment*". Universitas Kanjuruhan : Malang.

[3] Ahmad, I., et al. 2018. "*Performance Comparison of Support Vector Machine, Random Forest, and Extreme Learning Machine for Intrusion Detection*". in IEEE Access, vol. 6, pp. 33789-33795, 2018, doi: 10.1109/ACCESS.2018.2841987

[4] Hu, Y., & Sokolova, M. 2020. "*Explainable Multi-class Classification of Medical Data*". arXiv preprint arXiv:2012.13796.

[5] Zhenfeng, Shao, Muhammad Nasar Ahmad, and Akib Javed. 2024. "*Comparison of Random Forest and XGBoost Classifiers Using Integrated Optical and SAR Features for Mapping Urban Impervious Surface*". Remote Sensing 16, no. 4: 665. https://doi.org/10.3390/rs16040665

[6] Ramdani, F. and Furqon MT. 2022. "*The simplicity of XGBoost algorithm versus the complexity of Random Forest, Support Vector Machine, and Neural Networks algorithms in urban forest classification*". F1000Research 2022, 11:1069 (https://doi.org/10.12688/f1000research.124604.1)
