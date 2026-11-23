# Konteks Aplikasi DoomEvasion AI

## 1. Ringkasan aplikasi

DoomEvasion AI adalah aplikasi demo live untuk proyek Ujian Akhir Semester (UAS) mata kuliah Keamanan Informasi, Topik 2 Offensive AI. Aplikasi ini menunjukkan sebuah model Artificial Intelligence (AI) pendeteksi musuh di permainan Doom, lalu menunjukkan bagaimana model itu bisa dibohongi dengan serangan adversarial, dan bagaimana mitigasi membuat model itu lebih tahan.

Aplikasi ini bukan permainan baru dan bukan alat untuk menyerang sistem orang lain. Permainan hanya dipakai sebagai lingkungan lab yang terkendali dan bisa diulang, supaya frame, label, dan perilaku musuh semuanya berasal dari satu skenario yang sama.

Pengguna aplikasi ini adalah anggota kelompok proyek dan dosen pengampu, untuk keperluan presentasi, demo live, dan diskusi hasil eksperimen.

## 2. Masalah yang dijawab

Aplikasi ini menjawab dua pertanyaan yang berbeda:

- Apakah model yang akurat di data bersih juga tahan terhadap serangan?
- Berapa harga yang harus dibayar untuk membuat model itu lebih tahan?

Dua pertanyaan itu dijawab dengan cara yang sama: latih model baseline, serang model itu, latih model mitigasi, lalu serang ulang model mitigasi dan bandingkan hasilnya.

## 3. Komponen utama

- ViZDoom: platform permainan Doom untuk riset AI. ViZDoom menyediakan frame permainan dan label objek di dalam frame.
- SmallCNN: model Convolutional Neural Network (CNN) kecil yang dilatih kelompok sendiri. Tugas model ini adalah klasifikasi biner: menjawab ada musuh atau tidak ada musuh dari satu frame.
- Skenario defend_the_center: satu-satunya skenario permainan yang dipakai untuk melatih model. Model tidak dilatih pada peta ViZDoom yang lain.
- live_demo.py: server demo live. Server ini menjalankan permainan ViZDoom, menjalankan prediksi model pada setiap frame, dan menyajikan dashboard web lokal.
- index.html: dashboard web untuk demo live. Dashboard ini menampilkan frame permainan, prediksi live, hasil evaluasi resmi, contoh serangan, dan hasil mitigasi.
- Skrip data dan training: skrip untuk mengumpulkan data, membangun dataset, melatih model baseline, menyerang model dengan Fast Gradient Sign Method (FGSM), dan melatih model mitigasi dengan adversarial training.
- Skrip serangan: skrip untuk menjalankan FGSM pada model baseline dan pada model mitigasi, dengan tiga nilai epsilon yang sama.

## 4. Istilah yang dipakai di aplikasi

- Convolutional Neural Network (CNN): jaringan saraf untuk mengenali pola pada gambar. Di sini CNN dipakai untuk klasifikasi biner ada musuh atau tidak ada musuh.
- Fast Gradient Sign Method (FGSM): serangan adversarial satu langkah. Serangan ini menghitung gradien loss terhadap gambar input, lalu menggeser setiap piksel sebesar epsilon ke arah tanda gradien, supaya prediksi model berubah.
- Adversarial training: mitigasi utama di proyek ini. Model baru dilatih dari nol dengan contoh adversarial yang dibuat selama training, supaya batas keputusannya lebih tahan terhadap gangguan kecil.
- Attack Success Rate (ASR): ukuran keberhasilan serangan. ASR adalah proporsi sampel yang awalnya diprediksi benar, lalu berhasil dibalik prediksinya oleh serangan.
- False positive: frame tanpa musuh yang diprediksi salah sebagai ada musuh.
- False negative: frame bermusuh yang diprediksi salah sebagai tidak ada musuh.
- Label: aturan label di proyek ini adalah label 1 kalau ada kategori Monster yang terlihat di frame, label 0 kalau tidak ada, dan label Self selalu diabaikan.

## 5. Alur kerja aplikasi

Alur kerja aplikasi, dari awal sampai demo, adalah sebagai berikut:

- ViZDoom menghasilkan frame permainan pada skenario defend_the_center.
- Frame disimpan dan diberi label, lalu menjadi dataset final yang dipakai untuk semua training dan serangan.
- Dataset final melatih SmallCNN baseline.
- FGSM menyerang SmallCNN baseline.
- Adversarial training melatih SmallCNN mitigasi dari nol.
- FGSM menyerang ulang SmallCNN mitigasi dengan epsilon yang sama.
- Dashboard demo membaca model baseline dan berkas hasil resmi saat server dinyalakan, lalu menampilkan prediksi live dan tabel hasil.

Alur kerja demo live, saat aplikasi sedang jalan, adalah sebagai berikut:

- Pengguna membuka dashboard di alamat lokal dan memainkan permainan dengan keyboard dan mouse.
- Server mengambil frame permainan, mengubahnya menjadi gambar grayscale berukuran kecil, lalu meminta prediksi dari model yang aktif.
- Dashboard menampilkan frame, prediksi live, keyakinan model, probabilitas ada musuh, dan label game sebagai pembanding.
- Pengguna bisa mengambil capture dari satu frame, memberi label sebenarnya untuk frame itu, lalu menguji capture itu dengan FGSM.
- Pengguna bisa beralih model dengan tombol alih di dashboard.

## 6. Alih Baseline dan Mitigasi

Dashboard demo memiliki tiga mode model:

- Baseline: hanya model baseline yang dipakai untuk prediksi live dan uji FGSM.
- Mitigasi: hanya model mitigasi yang dipakai untuk prediksi live dan uji FGSM.
- Bandingkan: kedua model dipakai pada frame yang sama, dan prediksinya tampil berdampingan.

Mode Bandingkan dibuat supaya penonton bisa melihat trade-off secara langsung: baseline lebih akurat di gambar bersih, mitigasi lebih tahan saat diserang. Mode ini tidak menggabungkan dua model menjadi satu model, dan tidak melatih ulang model apa pun. Mode ini hanya memuat dua checkpoint yang sudah ada dan menjalankan keduanya pada frame yang sama.

Catatan penting soal berkas model mitigasi: berkas model mitigasi resmi berada di folder hasil mitigasi, sedangkan berkas model yang tersedia untuk demo saat ini berasal dari folder latihan mitigasi. Keduanya adalah hasil adversarial training dengan konfigurasi yang sama, tetapi berkas latihan bukan berkas resmi. Angka resmi tetap merujuk pada berkas hasil resmi, bukan pada berkas latihan.

## 7. Struktur folder

Struktur folder proyek, tanpa isi data di dalamnya, adalah sebagai berikut:

- Berkas utama di root proyek: berkas demo live (`live_demo.py`), berkas dashboard (`index.html`), berkas README, berkas GEMINI, berkas daftar dependensi (`requirements.txt`), dan berkas `.gitignore`.
- Folder skrip (`scripts/`): berisi skrip pengumpulan data final, analisis duplikat, pembangunan dataset (split train, validasi, dan test per episode), training baseline, training mitigasi, serangan FGSM, perekaman demo, dan smoke test.
- Folder data (`data_final/`): berisi frame dan split yang dipakai untuk semua training dan serangan resmi.
- Folder hasil (`runs/`): berisi folder hasil baseline (`baseline/`), hasil mitigasi (`mitigation/`), hasil serangan pada baseline (`fgsm/`), hasil serangan sesudah mitigasi (`fgsm_mitigated/`), dan rekaman demo (`demo/`). Folder hasil resmi wajib dipertahankan dan tidak boleh dihapus.
- Folder virtual environment (`.venv/`): folder environment Python lokal. Folder ini besar, diabaikan oleh git, dan dibutuhkan supaya demo bisa jalan. Jangan hapus folder ini.

Nama folder hasil tidak boleh diubah sembarangan, karena nama itu dipakai langsung di skrip demo, di skrip training, dan di laporan akhir.

## 8. Cara menjalankan

Prasyarat: Python dan virtual environment aktif, dengan dependensi yang terpasang dari berkas daftar dependensi.

Semua perintah diketik dari folder proyek, dengan virtual environment aktif, dan dengan urutan sebagai berikut:

- Kumpulkan data final.
- Bangun dataset dari data final.
- Latih model baseline.
- Serang model baseline dengan FGSM.
- Latih model mitigasi dengan adversarial training.
- Serang ulang model mitigasi dengan FGSM.
- Jalankan demo live.

Perintah demo live, dari folder proyek:

- Jalankan server demo dengan model baseline sebagai model utama.
- Tambahkan argumen model mitigasi kalau ingin mengaktifkan alih Baseline dan Mitigasi.
- Buka dashboard di alamat lokal yang dicetak server saat dinyalakan.
- Mainkan permainan dengan keyboard dan mouse, lalu gunakan tombol capture dan uji FGSM di dashboard.

Server demo harus di-restart setiap kali berkas demo diubah, karena server memakai kode yang dimuat saat server dinyalakan, bukan kode yang baru saja diubah.

## 9. Batasan yang perlu diketahui

- Model hanya dilatih pada skenario defend_the_center. Prediksi pada peta ViZDoom yang lain tidak dapat diandalkan, dan dashboard memberi peringatan tentang hal ini.
- Prediksi live selalu memakai satu frame grayscale berukuran kecil, tanpa konteks waktu. Frame yang berbeda dari pola latih, misalnya musuh di tepi frame, jauh, kecil, atau sebagian tertutup, lebih mudah terlewat.
- Demo live memakai model baseline secara default. Model mitigasi hanya dipakai kalau mode dialihkan atau kalau mode Bandingkan dipilih.
- Serangan yang diuji di proyek ini hanya FGSM satu langkah. Serangan iteratif yang lebih kuat tidak diuji di sini.
- Mitigasi utama yang diuji hanya adversarial training. Preprocessing sederhana, deteksi input, dan pertahanan lain tidak diuji sebagai mitigasi utama.
- Tabel hasil resmi di dashboard dihitung pada kumpulan test yang tetap, bukan dari frame live. Hasil live dan hasil test tidak boleh dicampur.

## 10. Aturan dan etika

- Semua eksperimen hanya dilakukan di lingkungan lab lokal, di satu laptop, tanpa menyerang sistem orang lain.
- Kode, payload, dan teknik serangan hanya untuk proyek ini dan tidak disebarkan untuk keperluan lain.
- Menjalankan ulang skrip pengumpulan data, training, serangan, atau rekaman demo dengan folder output yang sama akan menimpa data lama secara bersih (termasuk menghapus sisa file/episode lama di folder target). Jika ingin menyimpan eksperimen terpisah tanpa menimpa, arahkan argumen `--out` ke folder lain.
- Commit dan push ke git dilakukan oleh Ahmad sendiri, bukan oleh asisten.
- Perubahan di laptop hanya dilakukan di dalam folder proyek ini. Jangan mengubah, menghentikan, atau me-restart apa pun di luar folder proyek ini tanpa izin eksplisit.

## 11. Catatan untuk pengembangan lanjutan

Ide pengembangan yang sudah dicatat, tanpa mengubah hasil resmi yang ada, adalah sebagai berikut:

- Menguji serangan lain yang lebih kuat, untuk melihat apakah mitigasi FGSM juga tahan terhadap serangan itu.
- Melatih mitigasi dengan campuran beberapa epsilon latih, supaya ketahanan di epsilon yang lebih besar ikut membaik.
- Mengumpulkan data dari lebih dari satu skenario ViZDoom, supaya model tidak hanya bekerja di satu peta.
- Menguji pertahanan pendukung, seperti deteksi input adversarial dan preprocessing, dan mengukur harga masing-masing dengan metrik yang sama.
- Menampilkan probabilitas kedua kelas di dashboard, bukan hanya keyakinan pemenang, supaya kasus yang tipis terlihat jelas.
- Menambahkan smoothing temporal pada prediksi live, supaya prediksi tidak berkedip dari frame ke frame.
