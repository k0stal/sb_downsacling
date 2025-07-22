## Downscaling Utilities

---

### Overview

This repository provides a set of Python utilities for training and evaluating super-resolution models designed to downscale the **ERA5** and **CERRA** datasets.

- **ERA5** (ECMWF reanalysis)
  *Source:* [ERA5 hourly data on single levels from 1940 to present](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels)
- **CERRA** (The Copernicus European Regional ReAnalysis)
  *Source:* [CERRA sub-daily regional reanalysis data for Europe on single levels from 1984 to present](https://cds.climate.copernicus.eu/datasets/reanalysis-cerra-single-levels)

---

### Implemented Models

The following super-resolution models are currently implemented:

- **Bicubic Interpolation** – Baseline for upscaling.
- **SRCNN** (Super-Resolution Convolutional Neural Network)  
  *Original paper:* [Learning a Deep Convolutional Network for Image Super-Resolution (Dong et al., 2016)](https://arxiv.org/abs/1501.00092)
- **ESPCN** (Efficient Sub-Pixel Convolutional Neural Network)  
  *Original paper:* [Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network (Shi et al., 2016)](https://arxiv.org/abs/1609.05158)
- **FNO** (Fourier Neural Operator)  
  *Original paper:* [Fourier Neural Operator for Parametric Partial Differential Equations (Li et al., 2021)](https://arxiv.org/abs/2010.08895)

 *Note:* While the baseline *FNO* model is fully impelmented, additional modifications and extensions to the FNO architecture are currently not finished yet.

---

### Framework & Logging

All models and training routines are built on top of the **[PyTorch](https://pytorch.org/)**. Model training and evaluation metrics are automatically logged to **[TensorBoard](https://www.tensorflow.org/tensorboard)**.

---

### Script Automation

Additionally, the repository includes a Python script to automatically generate `.sh` shell scripts for training and evaluation tasks.

---
