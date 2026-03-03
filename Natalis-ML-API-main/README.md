
# 👶 Natalis: Fetal Head Biometry & GA Assessment

**Natalis** is a deep learning-based medical imaging tool designed to automate the segmentation of the fetal head from ultrasound images. It performs precise **Head Circumference (HC)** measurements, estimates **Gestational Age (GA)**, and provides **abnormality classifications** based on population-specific growth charts.

---

## 🧠 Model Architecture & AI Components

The core of this project is a sophisticated computer vision pipeline that combines deep learning segmentation with advanced post-processing:

### 1. Segmentation Model
* **Architecture:** `DeepLabV3+` — utilizes Atrous Spatial Pyramid Pooling (ASPP) to capture the fetal head boundary at multiple scales.
* **Encoder:** `timm-efficientnet-b4` (Pre-trained on ImageNet).
* **Input:** Single-channel (grayscale) ultrasound images.
* **Working Resolution:** 256x256 pixels (configured in `configs.py`).

### 2. Learnable Resizer
Instead of standard bilinear interpolation, the project utilizes a **Learnable Resizer**. This module learns to resize input images (from 512x512 down to 256x256) in a way that preserves the most critical features for the segmentation model.

### 3. Inference Pipeline Enhancements
To ensure high-fidelity masks, the inference process (`inference.py`) includes:
* **Test-Time Augmentation (TTA):** Averages predictions to improve robustness.
* **Largest Connected Component (LCC):** Removes noise by keeping only the primary head region.
* **DenseCRF Refinement:** Refines boundaries based on the original image intensities.
* **Morphological Cleaning:** Smooths the final binary mask using closing operations.

---

## 📊 Biometry & Medical Logic

Once the fetal head is segmented, the system performs the following clinical calculations:

### Head Circumference (HC)
Uses an ellipse-fitting algorithm on the segmentation mask. The circumference is calculated using the **Ramanujan II approximation**:

C ≈ π [3(a+b) − √((3a+b)(a+3b))]

### Gestational Age (GA)
Estimated from the computed HC using the **Intergrowth-21st** formula, converting measurements into weeks and days.

### Abnormality Classification
The system utilizes racial and ethnic-specific data (**Non-Hispanic White, Black, Hispanic, Asian**) to determine if a measurement is an outlier.
* **Microcephaly:** HC < 3rd Percentile.
* **Normal:** HC between 5th and 95th Percentile.
* **Macrocephaly:** HC > 97th Percentile.

---

## 📂 Project Structure

| Module | Purpose | Key Functionality |
| :--- | :--- | :--- |
| **app.py** | UI/UX | Streamlit dashboard, Plotly growth charts, and PDF reporting. |
| **model.py** | AI Core | DeepLabV3+ architecture with EfficientNet-B4 encoder. |
| **inference.py** | Pipeline | Orchestrates TTA, CRF, and post-processing switches. |
| **age_cal.py** | Biometry | Ellipse fitting and GA calculation via Intergrowth-21st. |
| **abnormality.py**| Diagnostics| Excel-based percentile lookup and classification logic. |
| **configs.py** | System | Pixel-to-MM calibration and model hyperparameters. |

---

## 🚀 Setup and Installation

### 1. Environment
Clone the repository and install the required dependencies:
```bash
git clone [https://github.com/yourusername/natalis.git](https://github.com/yourusername/natalis.git)
cd natalis
pip install -r requirements.txt

```

### 2. Data & Calibration

* **Growth Charts:** Ensure your `FGCalculatorPercentileRange.xlsx` is placed in the path specified in `configs.py`.
* **Calibration:** Update `PIXEL_SIZE_MM` in `configs.py` to match your ultrasound machine's specific scale (default is 0.15).





