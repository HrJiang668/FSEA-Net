# FSEA-Net

This repository provides the CFE and HFDSF modules and the image identifiers for the USE and Tryp train, validation, and test splits used in the study.

## Contents

- [`FSEA-Net-code-release/CFE.py`](FSEA-Net-code-release/CFE.py) and [`FSEA-Net-code-release/HFDSF.py`](FSEA-Net-code-release/HFDSF.py): core modules.
- [`split_ids/USE/`](split_ids/USE/): 4,256 training, 852 validation, and 268 test image filenames.
- [`split_ids/Tryp/`](split_ids/Tryp/): 1,893 training, 610 validation, and 612 test image filenames.

Each split file contains one `.jpg` filename per line, sorted alphabetically. These lists record the dataset assignment used in the study; they do not contain images or annotation files. Filenames can be matched to the corresponding images by basename.

## Original dataset sources

- USE: [Urinary Sediment Dataset](https://github.com/174614361/Urinary-Sediment-Dataset).
- Tryp: [Figshare dataset](https://doi.org/10.6084/m9.figshare.22825787).
- Urised11: [Wang et al., *Applied Sciences* 2023](https://doi.org/10.3390/app132111676). The source publication describes the access route; Urised11 images and split identifiers are not distributed here.

For the images, annotations, and applicable reuse terms, consult the original dataset sources. No source images or annotations are redistributed in this repository.
