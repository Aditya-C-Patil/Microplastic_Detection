# 🔬 Edge-AI Microplastic Detection System using Computer Vision

An automated, cost-effective hardware-software integration pipeline designed to detect and classify Sudan III stained microplastics in water samples. By combining optical excitation with deterministic computer vision feature engineering, this system isolates microplastic particles and eliminates false positives caused by organic matter, air bubbles, and specular reflections.

---

## 🚀 The Core Engineering Approach

Detecting microplastics in aqueous solutions is challenging because microscopic synthetic particles lack contrast and are easily confused with ambient dust, air bubbles, or container reflections. This project overcomes this through a synchronized optical-digital pipeline.

### 1. Optical Hardware Setup & Differential Excitation

* **Targeted Staining:** Samples are treated with Sudan III dye, which selectively adsorbs onto hydrophobic synthetic polymers.
* **Dual-Frame Optical Capture:**

  * **$I_{\text{off}}$:** Ambient pre-excitation reference frame (blue LED OFF).
  * **$I_{\text{on}}$:** Active optical excitation frame (blue LED ON).
* **Physical Filtering:** An orange bandpass barrier filter blocks blue excitation scatter while transmitting red/orange fluorescence emissions to the sensor.

### 2. Signal Processing & Image Preprocessing

**`src/preprocess.py`**

* **Red-Channel Subtraction:** Computes $I_{\text{diff}} = \max(0, I_{\text{on,R}} - I_{\text{off,R}})$ to cancel ambient light and non-fluorescent floaters.
* **Spatial Denoising:** Applies a $3 \times 3$ 2D median filter to eliminate sensor shot noise.
* **Dual Mask Gating:** Intersects the differential intensity mask with an HSV color threshold ($H \in [5,25]$, $S \in [80,255]$, $V \in [50,255]$) via bitwise logical AND.
* **Morphological Refinement:** Performs sequential opening and closing with an elliptical structuring element to seal internal voids and prune edge artifacts.
* **Blob Filtering:** Uses 8-connectivity connected-component analysis to remove sub-resolution noise ($\text{Area} < 30\text{ px}$).

---

## 📊 Geometric Feature Engineering & Classification

Instead of high-latency deep learning models, the system computes deterministic geometric descriptors on extracted particle contours.

### Morphology Descriptors

* **Circularity:** $4\pi A / P^2$
* **Aspect Ratio:** $W / H$
* **Solidity:** $A / A_{\text{hull}}$
* **Extent:** $A / (W \cdot H)$
* **Eccentricity:** $\sqrt{1 - (b/a)^2}$ via fitted ellipse

### Hu Invariant Moments

Seven rotation-, scale-, and translation-invariant shape moments ($h_1$ to $h_7$) are extracted for each detected particle.

### Composite Confidence Score

The confidence score combines normalized stain intensity with shape regularization:

$\text{Confidence} = 0.7 \cdot \left(\frac{\bar{I}_{\text{red}}}{255}\right) + 0.3 \cdot \min(1.0, 1.5 \cdot \text{Circularity})$

### Multi-Conditional Filtering

**`src/predict.py`**

Raw candidate detections pass through a multi-attribute filter to separate probable microplastics from organic background debris:

```python
mask = (
    (df["confidence"] >= 0.65) &
    (df["area_px"] >= 30) &
    (df["circularity"] >= 0.50) &
    (df["solidity"] >= 0.75) &
    (df["extent"] >= 0.40) &
    (df["aspect_ratio"] <= 4.0)
)
```

| Class        | Classification Criteria     | Target Polymer Profile                     |
| ------------ | --------------------------- | ------------------------------------------ |
| **Fiber**    | $\text{Aspect Ratio} > 3.0$ | Synthetic textile threads, nylon filaments |
| **Bead**     | $\text{Circularity} > 0.70$ | Microbeads, cosmetic polymers              |
| **Fragment** | Default fallback            | Jagged PET/PP structural shards            |

---

## 🔬 Experimental Validation & Parameter Tuning

Because microscopic dyed particles lack public annotated datasets, the pipeline was calibrated and validated through a multi-stage empirical testing protocol.

* **Surrogate Reference Calibration:** Precision-cut PET micro-glitter particles with known geometric boundaries were used to calibrate optical red-channel subtraction thresholds and verify the HSV color gate.
* **Spike-and-Recovery Testing:** Tested on spiked water samples containing transparent consumer plastics, including PET bottle shards and LDPE films, along with ambient interferents such as air bubbles and dust.
* **Baseline Approach:** Single-channel thresholding yielded approximately **65% recovery accuracy** due to under-segmentation of transparent edges and false positives from reflective container boundaries.
* **Optimized Pipeline:** Introducing dual-frame subtraction, HSV bitwise gating, and strict geometric boundary thresholds ($\text{Solidity} \ge 0.75$, $\text{Circularity} \ge 0.50$) raised true-positive particle recovery to approximately **94%** while eliminating false positives from non-stained organic contaminants.

---

## 📁 Repository Structure

```text
Microplastic_Detection/
│
├── data/
│   ├── sample_led_off.jpg        # Ambient baseline reference frame
│   └── sample_led_on.jpg         # Active excitation frame
│
├── notebooks/
│   └── demo.ipynb                # Visual inspection & step-by-step debug notebook
│
├── src/
│   ├── preprocess.py             # Image subtraction, HSV masking, morphology & feature extraction
│   └── predict.py                # Multi-attribute particle filter & bounding box overlay
│
├── output/                       # Saved output images and detection_metrics.csv
│
├── main.py                       # CLI execution script
├── requirements.txt              # Project dependencies
├── README.md                     # Technical documentation
└── LICENSE                       # MIT License
```

---

## 🛠️ Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Aditya-C-Patil/Microplastic_Detection.git
cd Microplastic_Detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Detection Pipeline

```bash
python main.py
```
