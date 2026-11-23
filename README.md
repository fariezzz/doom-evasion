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
| live_demo.py, index.html | Server demo live dan dashboard web lokal. |
| scripts/smoke_test.py | Tes instalasi dan aturan label ViZDoom. |
| scripts/collect_data_final.py | Kumpul data final: 30 episode, policy shoot. |
| scripts/analyze_duplicates.py | Analisis frame duplikat untuk protokol skip dan stride. |
| scripts/build_dataset.py | Buang duplikat, lalu split train, validation, dan test per episode. |
| scripts/train_baseline.py | Training CNN baseline. |
| scripts/attack_fgsm.py | Evaluasi serangan FGSM pada 3 epsilon. |
| scripts/train_mitigation.py | Adversarial training dari nol. |
| scripts/demo_capture.py | Rekam gameplay ViZDoom dengan prediksi CNN. |
| data_final/ | Frame dan split yang dipakai semua training dan serangan. |
| runs/baseline/, runs/mitigation/ | Hasil training resmi. |
| runs/fgsm/, runs/fgsm_mitigated/ | Hasil serangan resmi. |
| runs/demo/ | Rekaman video demo gameplay. |
| requirements.txt | Versi software lab yang terpin. |

## Cara menjalankan

Prasyarat: Python 3.11 dan virtual environment aktif. Install dependensi sekali:

```powershell
python -m pip install -r requirements.txt
```

Semua perintah di bawah diketik dari folder proyek, dengan virtual environment aktif, urut dari atas:

```powershell
python .\scripts\collect_data_final.py --episodes 30 --max-steps 600 --seed-start 400 --out data_final --policy shoot --skip-steps 25 --stride 2
python .\scripts\build_dataset.py --data data_final --split-seed 12345
python .\scripts\train_baseline.py
python .\scripts\attack_fgsm.py
python .\scripts\train_mitigation.py
python .\scripts\attack_fgsm.py --model runs/mitigation/best_model.pt --out runs/fgsm_mitigated
python .\live_demo.py --model runs/baseline/best_model.pt --map defend_the_center
```

## Catatan

- data_final dapat dibuat ulang secara deterministik dengan dua perintah pertama, seed 12345.
- Berkas model .pt serta folder runs/fgsm/ dan runs/fgsm_mitigated/ diabaikan oleh .gitignore (sesuai aturan). Log, metrik, dan gambar hasil baseline ikut ter-commit.
- Skenario: defend_the_center. Aturan label: label 1 kalau ada kategori Monster, label Self diabaikan.
- Status: Instalasi, pengumpulan data, training baseline, serangan FGSM, mitigasi adversarial training, demo live web (`live_demo.py` + `index.html`), dan rekaman demo (`runs/demo/gameplay.mp4`) sudah selesai.
