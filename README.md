# DoomEvasion AI

Proyek akhir Keamanan Informasi (KP70065004), Topik 2 Offensive AI, semester Ganjil 2026/2027.

DoomEvasion AI menguji ketahanan sebuah Convolutional Neural Network (CNN) kecil yang dilatih dari frame game ViZDoom terhadap serangan Fast Gradient Sign Method (FGSM). Tugas modelnya adalah klasifikasi biner: menjawab apakah ada musuh di dalam frame (kelas 1) atau tidak (kelas 0). Serangan diukur dengan Attack Success Rate (ASR), yaitu persentase sampel yang awalnya dijawab benar lalu berubah menjadi salah setelah diserang.

## Hasil ringkas

| Ukuran | Baseline | Setelah mitigasi |
| --- | --- | --- |
| Akurasi bersih | 0,9628 | 0,9080 |
| Macro-F1 bersih | 0,9297 | 0,8327 |
| ASR, epsilon 0,01 | 0,5846 | 0,0546 |
| ASR, epsilon 0,03 | 0,9963 | 0,2593 |
| ASR, epsilon 0,05 | 0,8603 | 0,2982 |

Mitigasinya adalah adversarial training dari nol dengan epsilon latih 0,03. ASR turun 0,53 sampai 0,74 di semua epsilon, dengan tradeoff nyata: akurasi bersih turun 0,0548 dan macro-F1 turun 0,0970. ASR sebelum mitigasi tidak monoton (epsilon 0,03 tertinggi) karena efek overshoot satu langkah FGSM, bukan bug. Penyebut ASR berbeda (544 sebelum mitigasi, 513 sesudah) karena ASR dihitung hanya dari sampel yang awalnya benar.

## Struktur proyek

| Berkas / folder | Isi |
| --- | --- |
| smoke_test.py | Tes instalasi dan aturan label ViZDoom. |
| collect_data.py, collect_data_center.py, collect_data_center_v2.py | Skrip pilot (skenario basic dan defend_the_center awal). Tidak dipakai training final. |
| collect_data_final.py | Kumpul data final: 30 episode, policy shoot. |
| analyze_duplicates.py | Analisis frame duplikat untuk protokol skip dan stride. |
| build_dataset.py | Buang duplikat, lalu split train, validation, dan test per episode. |
| train_baseline.py | Training CNN baseline. |
| attack_fgsm.py | Evaluasi serangan FGSM pada 3 epsilon. |
| train_mitigation.py | Adversarial training dari nol. |
| mnist_latihan.py | Latihan CNN dan FGSM pada dataset MNIST, untuk belajar. Bukan hasil resmi. |
| data_final/ | Frame dan split yang dipakai semua training dan serangan. |
| data_raw/, data_center/, data_center_v2_turn/, data_center_v2_shoot/ | Hanya manifest.csv. PNG pilot dihapus saat perapian 15 November 2026. |
| data_mnist/ | Cache dataset MNIST untuk latihan. |
| runs/baseline/, runs/mitigation/ | Hasil training. |
| runs/fgsm/, runs/fgsm_mitigated/ | Hasil serangan. |
| runs/mnist_latihan/ | Hasil latihan MNIST. |
| requirements.txt | Versi software lab yang terpin. |

## Cara menjalankan

Prasyarat: Python 3.11 dan virtual environment aktif. Install dependensi sekali:

```powershell
python -m pip install -r requirements.txt
```

Semua perintah di bawah diketik dari folder proyek, dengan virtual environment aktif, urut dari atas:

```powershell
python .\collect_data_final.py --episodes 30 --max-steps 600 --seed-start 400 --out data_final --policy shoot --skip-steps 25 --stride 2
python .\build_dataset.py --data data_final --split-seed 12345
python .\train_baseline.py
python .\attack_fgsm.py
python .\train_mitigation.py
python .\attack_fgsm.py --model runs/mitigation/best_model.pt --out runs/fgsm_mitigated
```

Latihan MNIST (tidak memengaruhi hasil resmi):

```powershell
python .\mnist_latihan.py --data data_mnist --epochs 3 --out runs/mnist_latihan
```

## Catatan

- data_final dapat dibuat ulang secara deterministik dengan dua perintah pertama, seed 12345.
- Berkas model .pt serta folder runs/fgsm dan runs/fgsm_smoke diabaikan oleh .gitignore. Log, metrik, dan gambar hasil baseline ikut ter-commit.
- Skenario: defend_the_center. Aturan label: label 1 kalau ada kategori Monster, label Self diabaikan.
- Status: Langkah 1 sampai 5 selesai (instalasi, data, baseline, serangan, mitigasi). Demo Streamlit, laporan akhir, dan video demo masih dikerjakan.
