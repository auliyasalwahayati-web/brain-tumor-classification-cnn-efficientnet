# Brain Tumor MRI Classification (SimpleCNN vs EfficientNet-B0)

Proyek klasifikasi citra MRI tumor otak (4 kelas: `glioma`, `meningioma`, `notumor`, `pituitary`) menggunakan PyTorch di **CPU** dengan membandingkan **SimpleCNN** dan **EfficientNet-B0**.

## Struktur Proyek
- `src/config.py`: konfigurasi path, hyperparameter, dan pengaturan CPU.
- `src/dataset.py`: pembacaan data, stratified train/val split, transformasi citra, dan `DataLoader`.
- `src/models.py`: arsitektur `SimpleCNN` dan `EfficientNet-B0`.
- `src/train.py`: loop training, validasi, penyimpanan bobot terbaik, dan riwayat metrik.
- `src/evaluate.py`: evaluasi pada data test (Accuracy, Precision, Recall, F1-Score, Confusion Matrix) serta penyimpanan grafik.
- `main.py`: entrypoint CLI untuk menjalankan training dan evaluasi.

## Cara Menjalankan (CPU)
1. Aktifkan virtual environment `.venv`:
   ```powershell
   .\.venv\Scripts\Activate.ps1