# FSEA-Net
# FSEA-Net core modules

This repository contains the `CFE` (adaptive frequency extraction) and `HFDSF` (high–low frequency interaction guidance) PyTorch modules used in the FSEA-Net study. 
## Files

- `CFE.py` — multi-scale frequency evidence construction and allocation.
- `HFDSF.py` — structural–semantic interaction and optional feature-map visualization.

## Datasets used in the study

- **USE:** [Urinary Sediment Dataset](https://github.com/174614361/Urinary-Sediment-Dataset). The dataset page documents 4,256 training, 852 validation, and 268 test images.
- **Tryp:** [Figshare dataset](https://doi.org/10.6084/m9.figshare.22825787). Its published partitions contain 1,893 training, 610 validation, and 612 test positive images.
- **Urised11:** Described in [Wang et al., *Applied Sciences* 2023](https://doi.org/10.3390/app132111676), which reports 5,891 training and 1,473 evaluation images. The publication states that the data are available from the corresponding author on reasonable request; **this citation is not a public download link**.

The above links identify the original dataset sources. They do not by themselves document the exact image IDs or any modified splits used for the present study.

## Reproducibility scope

Reproduction of the full FSEA-Net results additionally requires the model configuration, IPAM implementation, training and evaluation settings, exact split identifiers, checkpoint-selection rule, and dataset access. No raw image data are redistributed here.
