# Architecture & Engineering Decision Records (ADR)

This document records the key architectural, hardware, and algorithmic decisions made during the design and development of the Edge-AI Microplastic Detection System.

---

## 📌 Context & Problem Statement

Conventional microplastic detection relies on centralized laboratory analytical techniques (such as FTIR Spectroscopy, Raman Microscopy, and Pyrolysis-GC/MS). While highly accurate, these methods present major bottlenecks:
* **High Operational Cost**: Testing a single water sample in specialized analytical labs is expensive.
* **Turnaround Latency**: Samples must be preserved, transported, and batched, causing delays ranging from days to weeks.
* **Lack of Portability**: In-field, real-time water quality monitoring is not feasible with benchtop spectrometers.

**Objective**: Develop a cost-effective, portable hardware-software system that allows field operators to perform rapid, one-time investment microplastic screening in under a minute with low computational overhead.

---

## ADR 01: Classical Computer Vision vs. Deep Learning (YOLO / CNNs)

* **Status**: Accepted
* **Context**: Deep learning models (e.g., YOLO, Mask R-CNN) require thousands of annotated bounding boxes/masks across various water matrices and extensive GPU compute for real-time edge execution. No standardized, publicly available image datasets exist for dyed microplastics under differential excitation.
* **Decision**: Implement deterministic computer vision and contour morphology (OpenCV) using differential red-channel subtraction, connected components, and invariant Hu moments.
* **Trade-offs**:
  * *Pros*: Zero training dataset dependency; runs at < 50ms latency on low-cost CPUs (Raspberry Pi / Jetson); fully explainable parameter gating.
  * *Cons*: Requires fixed optical conditions and manual threshold tuning for extreme edge cases.

---

## ADR 02: Physical Optical Filtering (Orange Barrier Filter)

* **Status**: Accepted
* **Context**: High-intensity blue LEDs (used to excite Sudan III dye) produce significant scattered blue light that over-saturates the camera sensor, completely washing out faint particle emissions.
* **Decision**: Integrate a physical orange optical long-pass/bandpass filter over the camera lens.
* **Trade-offs**:
  * *Pros*: Mechanically attenuates the intense short-wavelength blue illumination scatter while transmitting the shifted red/orange wavelength emitted by stained polymers directly into the sensor.
  * *Cons*: Adds a minimal physical hardware cost to the optical rig assembly.

---

## ADR 03: Synchronized Dual-Frame Differential Red-Channel Subtraction

* **Status**: Accepted
* **Context**: Aqueous samples naturally contain non-plastic artifacts (e.g., container reflections, ambient light gradients, mineral floaters) that produce false positives in single-frame thresholding.
* **Decision**: Implement a two-state image capture pipeline—subtracting the ambient baseline frame ($I_{\text{off}}$) from the blue-LED active excitation frame ($I_{\text{on}}$) exclusively across the red channel.
* **Trade-offs**:
  * *Pros*: Cancels ambient lighting variations and static reflective boundaries, ensuring only actively fluorescing particles are passed to the segmentation mask.
  * *Cons*: Requires mechanical stabilization between successive frame captures to prevent motion registration blur.

---

## ADR 04: Empirical Calibration & Baseline Validation Protocol

* **Status**: Accepted
* **Context**: Standard classification accuracy (e.g., ROC-AUC) cannot be computed directly due to the lack of ground-truth labeled benchmark datasets for microscopic dyed particles.
* **Decision**: Validate system sensitivity and establish baseline thresholds using controlled physical surrogate standards:
  1. *Optical Calibration*: Precision-cut PET micro-glitter particles of known geometry used to calibrate red-channel subtraction and HSV threshold bounds.
  2. *Spike-and-Recovery Field Testing*: Transparent consumer plastics (PET bottle shards, LDPE films) introduced into distilled and ambient water matrices to measure particle recovery rates against organic noise.
* **Trade-offs**:
  * *Pros*: Provides rigorous, repeatable empirical validation for hardware and software thresholds without relying on synthetic labels.
  * *Cons*: Validation metrics are tied to controlled physical trials rather than large-scale automated data splits.
