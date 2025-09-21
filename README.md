# Palmprint Recognition (Image Classification)

Proyek ini mengimplementasikan sistem pengenalan telapak tangan menggunakan deep learning dengan TensorFlow/Keras untuk klasifikasi gambar telapak tangan dari 99 subjek berbeda.

## Dataset

Dataset yang digunakan berasal dari Kaggle: [Palmprint 100 People](https://www.kaggle.com/datasets/saqibshoaibdz/palmprint100people)

### Struktur Dataset
- **Total subjek**: 99 orang
- **Format gambar**: 128x128 pixels
- **Pembagian data**:
  - Training: 297 gambar
  - Validation: 198 gambar  
  - Test: 99 gambar

## Dependencies

```python
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from collections import Counter
```

## Arsitektur dan Eksperimen

### Eksperimen 1: CNN Sederhana (50 epochs)
**Arsitektur:**
- Conv2D (32 filters, 3x3, ReLU)
- MaxPooling2D (2x2)
- Conv2D (64 filters, 3x3, ReLU)
- MaxPooling2D (2x2)
- Flatten
- Dense (128, ReLU)
- Dense (100, Softmax)

**Hasil:**
- Akurasi Test: 15%
- Total Parameter: 7,405,220

### Eksperimen 2: CNN dengan Lebih Banyak Epoch (100 epochs)
**Arsitektur:** Sama dengan Eksperimen 1

**Hasil:**
- Akurasi Test: 53%
- Peningkatan signifikan dengan training lebih lama

### Eksperimen 3: Preprocessing dengan CLAHE
**Preprocessing:**
- Konversi ke grayscale
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Normalisasi (0-1)

**Hasil:**
- Akurasi Test: 51%
- Sedikit penurunan dari eksperimen 2

### Eksperimen 4: Transfer Learning dengan MobileNetV3 (25 epochs)
**Arsitektur:**
- Base model: MobileNetV3Large (ImageNet pretrained)
- GlobalAveragePooling2D
- Dense (128, ReLU)
- Dense (100, Softmax)
- Base model frozen (tidak dilatih)

**Hasil:**
- **Akurasi Test: 69%** ✨ (Terbaik)
- Total Parameter: 3,132,260
- Trainable Parameter: hanya 135,908

## Struktur Kode

### 1. Data Loading dan Preprocessing
```python
def load_data(data_dir):
    # Fungsi untuk memuat gambar dan label dari direktori
    
def apply_clahe(images):
    # Fungsi untuk menerapkan CLAHE preprocessing
```

### 2. Visualisasi Data
```python
def plot_specific_labels(images, labels, target_labels, dataset_name):
    # Fungsi untuk menampilkan sampel data dari label tertentu
```

### 3. Model Training
Setiap eksperimen menggunakan:
- Optimizer: Adam
- Loss function: Categorical Crossentropy
- Metrics: Accuracy
- Validation split berdasarkan pembagian manual

### 4. Evaluasi
- Classification report per subjek
- Visualisasi training history (accuracy dan loss)

## Cara Penggunaan

1. **Persiapan Dataset**
   ```python
   base_path = '/path/to/palmprint/dataset'
   train_path = os.path.join(base_path, 'train')
   test_path = os.path.join(base_path, 'valid')
   ```

2. **Load Data**
   ```python
   X_train, y_train = load_data(train_path)
   X_test, y_test = load_data(test_path)
   ```

3. **Preprocessing (untuk model terbaik)**
   ```python
   # Konversi ke grayscale dan CLAHE
   X_train_clahe = apply_clahe(X_train)
   
   # Konversi ke RGB untuk transfer learning
   X_train_rgb = np.stack([X_train_clahe]*3, axis=-1)
   X_train_processed = tf.keras.applications.mobilenet_v3.preprocess_input(X_train_rgb)
   ```

4. **Training Model Terbaik (MobileNetV3)**
   ```python
   # Buat model dengan transfer learning
   base_model = tf.keras.applications.MobileNetV3Large(
       input_shape=(128, 128, 3),
       include_top=False,
       weights='imagenet'
   )
   base_model.trainable = False
   
   # Tambahkan classifier layers
   x = base_model.output
   x = GlobalAveragePooling2D()(x)
   x = Dense(128, activation='relu')(x)
   predictions = Dense(num_classes, activation='softmax')(x)
   
   model = Model(inputs=base_model.input, outputs=predictions)
   ```

5. **Evaluasi**
   ```python
   y_pred_probs = model.predict(X_test_processed)
   y_pred_classes = np.argmax(y_pred_probs, axis=1)
   report = classification_report(y_test, y_pred_classes)
   ```

## Hasil dan Analisis

### Perbandingan Eksperimen

| Eksperimen | Arsitektur | Epochs | Preprocessing | Akurasi Test | Parameter |
|------------|------------|---------|---------------|--------------|-----------|
| 1 | CNN Sederhana | 50 | RGB Normalisasi | 15% | 7.4M |
| 2 | CNN Sederhana | 100 | RGB Normalisasi | 53% | 7.4M |
| 3 | CNN Sederhana | 50 | Grayscale + CLAHE | 51% | 7.4M |
| 4 | MobileNetV3 | 25 | Grayscale + CLAHE + RGB | **69%** | 3.1M |

### Insight Utama
1. **Transfer learning** memberikan hasil terbaik dengan waktu training lebih singkat
2. **Jumlah epoch** sangat berpengaruh pada model CNN sederhana (15% → 53%)
3. **CLAHE preprocessing** tidak memberikan improvement yang signifikan
4. **MobileNetV3** lebih efisien dalam parameter namun memberikan akurasi tertinggi

## Tantangan dan Keterbatasan

1. **Dataset kecil**: Hanya 3 gambar per subjek untuk training
2. **Imbalanced performance**: Beberapa subjek memiliki precision 0%
3. **Overfitting**: Training accuracy mencapai 100% namun validation accuracy hanya ~79%
4. **Single sample testing**: Setiap subjek hanya memiliki 1 gambar untuk testing

## Rekomendasi Pengembangan

1. **Data Augmentation**: Rotasi, flip, crop untuk memperbanyak data training
2. **Regularization**: Dropout, batch normalization, early stopping
3. **Fine-tuning**: Unfreeze beberapa layer terakhir dari pre-trained model
4. **Ensemble**: Kombinasi multiple model untuk hasil lebih robust
5. **Cross-validation**: Evaluasi yang lebih reliable dengan k-fold CV

## License

Dataset dari Kaggle dengan lisensi sesuai platform. Kode dapat digunakan untuk tujuan penelitian dan pembelajaran.

## Author
1. Rakha Dhifiargo Hariadi (2209489)
2. Defrizal Yahdiyan Risyad (2206131)
