# DoomEvasion AI

Proyek akhir Keamanan Informasi (KP70065004), Topik 2 Offensive AI, semester Ganjil 2026/2027.

DoomEvasion AI menguji ketahanan sebuah Convolutional Neural Network (CNN) kecil (`SmallCNN`) yang dilatih dari frame game ViZDoom (`defend_the_center`) terhadap serangan adversarial satu langkah (**FGSM**) dan multi-langkah iteratif (**PGD-10**), serta membandingkan strategi pertahanan **Adversarial Training v1**, **Adversarial Training v2 (FGSM-RS Multi-Epsilon)**, dan **Input Preprocessing (Defense-in-Depth)**.

## Hasil Ringkas (Test Set Tetap, 565 Frame)

| Metrik | Baseline | Mitigasi v1 (FGSM ε=0,03) | Mitigasi v2 (FGSM-RS Multi-ε) |
| --- | --- | --- | --- |
| Akurasi bersih | 0,9628 | 0,9080 | 0,9009 |
| Macro-F1 bersih | 0,9297 | 0,8327 | 0,8243 |
| FGSM ASR, ε=0,01 | 0,5846 | 0,0546 | 0,0727 |
| FGSM ASR, ε=0,03 | 0,9963 | 0,2593 | 0,3104 |
| FGSM ASR, ε=0,05 | 0,8603 | 0,2982 | 0,4479 |
| **PGD-10 ASR, ε=0,01** | 0,8474 | 0,0702 | 0,0904 |
| **PGD-10 ASR, ε=0,03** | **1,0000** | 0,4893 | 0,5128 |
| **PGD-10 ASR, ε=0,05** | **1,0000** | 0,7602 | **0,5796** |

### Temuan Utama Eksperimen Lanjutan
1. **Bukti Overshoot FGSM via PGD-10**: Penurunan FGSM ASR Baseline di $\varepsilon=0,05$ (`0,8603`) terbukti murni efek *single-step overshoot*; pada PGD-10, Baseline kolaps **100% ASR** (`544/544`) di $\varepsilon=0,03$ dan $\varepsilon=0,05$.
2. **Keunggulan Mitigasi v2 di Serangan Kuat**: Pada PGD-10 $\varepsilon=0,05$, Mitigasi v2 memangkas ASR sebesar **18,07 poin** dibanding v1 (`76,02%` → `57,96%`), serta menekan *False Alarm* FGSM $\varepsilon=0,05$ dari `63/67` (v1) menjadi `20/68` (v2).
3. **Transferabilitas Black-Box ($\varepsilon=0,03$)**: Serangan PGD-10 dari Baseline ke Mitigasi v1 hanya berhasil **11,1%** (dan ke v2 **24,2%**), jauh di bawah ancaman White-Box.
4. **Defense-in-Depth ($\varepsilon=0,03$)**: Input preprocessing saja pada Baseline gagal menahan PGD-10 (`bit_depth_4`: `100%` ASR, `median_3x3`: `99,33%` ASR). Kombinasi terbaik tanpa kolaps kelas adalah **Mitigasi v1 + Bit-Depth 4-bit** (`clean 90,62%`, `PGD-10 ASR 44,14%`) dan **Mitigasi v2 + Bit-Depth 3-bit** (`clean 90,27%`, `F1 0,7798`, `PGD-10 ASR 27,25%`).

## Struktur Proyek

| Berkas / folder | Isi |
| --- | --- |
| `live_demo.py`, `index.html` | Server demo live & dashboard interaktif (Live FGSM/PGD, Grad-CAM XAI, Defense Input, EMA Smoothing, kurva SVG, matriks transferabilitas). |
| `scripts/adversarial_utils.py` | Modul bersama FGSM, PGD-k, Grad-CAM, 6 preprocessing defense, dan pemuatan checkpoint aman (`weights_only=True`). |
| `scripts/collect_data_final.py` | Kumpul data final: 30 episode, policy `shoot`. |
| `scripts/analyze_duplicates.py` | Analisis frame duplikat untuk protokol skip dan stride. |
| `scripts/build_dataset.py` | Buang duplikat MD5, lalu split train/val/test per episode. |
| `scripts/train_baseline.py` | Training CNN baseline (`runs/baseline/`). |
| `scripts/train_mitigation.py` | Adversarial training v1 (`runs/mitigation/`) dan preset `--v2` FGSM-RS multi-epsilon + seleksi robust val (`runs/mitigation_v2/`). |
| `scripts/attack_fgsm.py` | Evaluasi serangan FGSM atau PGD (`--attack pgd`) + opsi `--defense` serta breakdown Evasion (`1→0`) vs False Alarm (`0→1`). |
| `scripts/evaluate_suite.py` | Suite analisis otomatis: Kurva ketahanan 8 epsilon, Matriks Transferabilitas Black-Box, dan Perbandingan Defense-in-Depth (`runs/analysis/`). |
| `scripts/demo_capture.py` | Rekam video gameplay ViZDoom dengan prediksi CNN (`runs/demo/`). |
| `scripts/smoke_test.py` | Tes instalasi dan aturan label ViZDoom. |
| `tests/test_project.py` | Suite pengujian otomatis (`unittest`) untuk integritas dataset, batas $L_\infty$ serangan, Grad-CAM, defense, dan API HTTP demo. |
| `data_final/` | Frame dan split yang dipakai seluruh training dan evaluasi. |
| `runs/` | Hasil resmi (`baseline/`, `mitigation/`, `fgsm/`, `fgsm_mitigated/`, `demo/`) serta hasil lanjutan (`mitigation_v2/`, `pgd*`, `fgsm_mitigated_v2/`, `analysis/`). |
| `requirements.txt` | Versi paket Python lab yang terpin. |

## Cara Menjalankan

Prasyarat: Python 3.11 dan virtual environment aktif (`.venv`).

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Pipeline lengkap dari awal:

```powershell
python .\scripts\collect_data_final.py --episodes 30 --max-steps 600 --seed-start 400 --out data_final --policy shoot --skip-steps 25 --stride 2
python .\scripts\build_dataset.py --data data_final --split-seed 12345
python .\scripts\train_baseline.py
python .\scripts\attack_fgsm.py
python .\scripts\train_mitigation.py
python .\scripts\attack_fgsm.py --model runs/mitigation/best_model.pt --out runs/fgsm_mitigated
```

Pipeline lanjutan v2, PGD-10, dan suite analisis:

```powershell
python .\scripts\train_mitigation.py --v2
python .\scripts\attack_fgsm.py --attack pgd --model runs/baseline/best_model.pt --out runs/pgd
python .\scripts\attack_fgsm.py --attack pgd --model runs/mitigation/best_model.pt --out runs/pgd_mitigated
python .\scripts\attack_fgsm.py --attack fgsm --model runs/mitigation_v2/best_model.pt --out runs/fgsm_mitigated_v2
python .\scripts\attack_fgsm.py --attack pgd --model runs/mitigation_v2/best_model.pt --out runs/pgd_mitigated_v2
python .\scripts\evaluate_suite.py
```

Menjalankan tes otomatis & demo live:

```powershell
python -m unittest discover -s tests -v
python .\live_demo.py --map defend_the_center
```

## Catatan

- `data_final` dapat dibuat ulang secara deterministik dengan seed `12345`.
- Berkas model `.pt`, `run_config.txt`, serta folder output serangan (`runs/fgsm/`, `runs/fgsm_mitigated/`, `runs/fgsm_mitigated_v2/`, `runs/pgd*/`) diabaikan oleh `.gitignore`.
- Skenario: `defend_the_center`. Aturan label: `1` jika ada kategori `Monster`, `0` jika tidak (`Self` diabaikan).
