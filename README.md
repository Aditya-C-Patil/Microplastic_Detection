# Edge-AI Microplastic Detection System using Computer Vision

An automated, cost-effective hardware-software integration solution designed to detect and classify Sudan III stained microplastics in water samples. By combining specialized optical lighting techniques with classic computer vision feature engineering, this system eliminates environmental noise and isolates microplastic particles with high confidence.

---

## 🚀 The Core Engineering Approach

Detecting microplastics under standard conditions is difficult because they are invisible to the naked eye and easily confused with air bubbles, container reflections, or dust. This project overcomes this challenge through a multi-stage hardware and software pipeline:

### 1. Optical Hardware Rig Configuration
**Targeted Staining:** Samples are treated with Sudan III dye, which selectively binds to plastics, changing their light emission property.
**Dual-State Backlighting:** A high-intensity blue LED acts as the excitation source.
**Physical Filtering:** An orange optical filter is placed directly over the camera lens. This blocks the massive scatter of background blue light while allowing the red/orange wavelength emitted by the stained plastic particles to pass through into the sensor.

### 2. Digital Signal & Image Processing Pipeline
**Differential Analysis:** The system captures an image with the LED off (`img_off`) and LED on (`img_on`), performing a **Red Channel Subtraction** to isolate active pixels.
**Denoising:** A Median Filter sweeps away single-pixel sensor artifacts and salt-and-pepper noise.
**Logical Gating:** A `bitwise_and` intersection maps the color threshold against the light-differential mask to completely eliminate background reflections and clear floaters.
**Morphological Cleanup:** Successive opening and closing operations fill internal holes and sweep away edge debris.

---

## 📊 Geometric Feature Extraction & Classification

Instead of relying on heavy black-box deep learning models that require immense computational power, this system runs fast, lightweight mathematical feature engineering on detected object contours to calculate:
**Circularity & Aspect Ratio:** Differentiates shapes into standard categories.
**Solidity & Extent:** Measures object density and contour packaging against convex hulls.
**Hu Moments:** Extracts 7 rotation and scale-invariant shape descriptors.

### Rule-Based Particle Profiling:
**Fiber:** High-aspect ratio structures (Aspect Ratio > 3)
**Bead:** Highly symmetrical, circular particles (Circularity > 0.7)
**Fragment:** Irregular, jagged, broken structural shards (default fallback)

---

## 📁 Repository Structure

```text
microplastic-detection/
│
├── data/
│   └── README.md          # Guide on how to name and save your target sample images
├── notebooks/
│   └── demo.ipynb         # Interactive Jupyter Notebook showcasing step-by-step visual plots
├── src/
│   ├── preprocess.py      # Differential subtraction, masking, and morphology operations
│   └── predict.py         # Advanced multi-conditional feature threshold filters
├── output/                # Local directory where processed outputs are saved
├── main.py                # Production command-line terminal runner execution script
├── requirements.txt       # Global library dependency configuration list
