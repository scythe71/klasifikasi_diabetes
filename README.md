# Klasifikasi Diabetes

Aplikasi web berbasis **Flask + Random Forest** untuk mengklasifikasikan risiko diabetes berdasarkan data kesehatan pasien. Input berupa kadar gula darah, berat badan, lingkar pinggang, dan lingkar pinggul — output berupa **Tidak** (sehat) atau **Terkena** (berisiko diabetes).

---

## Fitur

- **Prediksi real-time** via web form dengan SweetAlert2
- **Validasi input** lengkap (null, non-numeric, ≤0)
- **Dark theme UI** responsif (desktop & mobile)
- **Confidence score** — seberapa yakin model dengan prediksinya
- **Metrik model** ditampilkan: akurasi, presisi, recall, F1-score

---

## Cara Kerja

### Target Klasifikasi

Berdasarkan kadar HbA1c (`glyhb`) dalam dataset:

| glyhb | Label | Keterangan |
|---|---|---|
| `< 5.7` | **Tidak** | Normal, tidak berisiko diabetes |
| `≥ 5.7` | **Terkena** | Berisiko / sudah terkena diabetes |

### Fitur yang Digunakan

4 fitur dari dataset `diabetes_raw.csv`:

| Fitur | Satuan Input | Deskripsi |
|---|---|---|
| `stab.glu` | mg/dL | Gula darah puasa stabil |
| `weight` | kg (dikonversi ke lbs) | Berat badan |
| `waist` | cm (dikonversi ke inci) | Lingkar pinggang |
| `hip` | cm (dikonversi ke inci) | Lingkar pinggul |

### Model

- **Algoritma**: `RandomForestClassifier` (100 trees, `class_weight='balanced'`)
- **Scaling**: `StandardScaler` (fitur dinormalisasi sebelum training)
- **Split**: 80% train, 20% test (`stratify=y`)
- **random_state**: 42 (reproducible)

### Performa Model

| Metrik | Kelas Terkena |
|---|---|
| **Akurasi** | 88.75% |
| **Presisi** | 92.31% |
| **Recall** | 60.00% |
| **F1-Score** | 72.73% |

---

## Struktur Proyek

```
klasifikasi_diabetes/
├── app.py                  # Flask backend — training, prediksi, API
├── diabetes_raw.csv        # Dataset (403 pasien, 18 kolom)
├── static/
│   ├── script.js           # Frontend JS — fetch API + SweetAlert2
│   └── style.css           # Dark theme CSS
├── templates/
│   └── index.html          # Halaman utama form input
├── hasil_klasifikasi_diabetes.csv  # Hasil prediksi pada data uji
├── prediksi_data_baru.csv          # Riwayat prediksi data baru
└── README.md
```

---

## Cara Menjalankan

### 1. Install dependensi

```bash
pip install flask pandas scikit-learn
```

### 2. Jalankan aplikasi

```bash
python app.py
```

### 3. Buka browser

```
http://127.0.0.1:5000
```

---

## API Endpoint

### `POST /predict`

Menerima JSON dan mengembalikan hasil klasifikasi.

**Request body:**

```json
{
    "glucose": 82,
    "weight": 55,
    "waist": 74,
    "hip": 97
}
```

**Response sukses (200):**

```json
{
    "classification": "Tidak",
    "confidence": 100.0,
    "accuracy": 88.75,
    "precision": 92.31,
    "recall": 60.0,
    "f1_score": 72.73
}
```

**Response error (400):**

```json
{
    "error": "Field \"weight\" harus lebih dari 0"
}
```

### Catatan
- Berat badan (`weight`) diterima dalam **kg**, dikonversi ke **lbs** sebelum prediksi
- Lingkar pinggang (`waist`) dan pinggul (`hip`) diterima dalam **cm**, dikonversi ke **inci** sebelum prediksi
- Semua field wajib diisi, harus angka positif
- Confidence adalah probabilitas tertinggi dari `predict_proba`

---

## Dataset

**Sumber**: `diabetes_raw.csv` — data pasien diabetes dari studi medis.

- **403 baris**, 18 kolom
- Kolom penting: `glyhb` (HbA1c — digunakan sebagai target), `stab.glu`, `weight`, `waist`, `hip`, `age`, `chol`, `hdl`, `bp.1s`, `bp.1d`
- Tidak ada missing value pada 4 fitur utama + target setelah preprocessing

---

## Teknologi

| Stack | Library |
|---|---|
| Backend | Python, Flask |
| Machine Learning | scikit-learn (RandomForest, StandardScaler) |
| Frontend | HTML, CSS, JavaScript |
| Visualisasi | SweetAlert2 |
