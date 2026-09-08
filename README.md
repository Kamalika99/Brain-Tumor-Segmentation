# Brain Tumor Segmentation

This project implements a U-Net based deep learning model for segmenting brain tumors from MRI scans. It includes preprocessing, training with data augmentation, evaluation metrics, and visualization tools.

## Overview

Given a brain MRI slice, the model predicts a binary mask highlighting tumor regions. The notebook covers the full pipeline end to end:

- Loading and pairing MRI images with their ground-truth masks
- Preprocessing and normalization
- Data augmentation
- A U-Net model with batch normalization and dropout
- A custom weighted Dice loss to handle severe class imbalance
- Training with learning-rate scheduling
- Visualization of training curves and predictions vs. ground truth

## Dataset

You can download the dataset from [https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation).

- **Source:** LGG MRI Segmentation (Kaggle), originally from The Cancer Imaging Archive (TCIA), collected as part of a 2019 study by Buda et al. on FLAIR abnormality segmentation.
- **Format:** `.tif` MRI slices, each with a corresponding `<name>_mask.tif` binary segmentation mask.
- **Size used in this run:** 3,929 image/mask pairs, split 80/20 into 3,143 training and 786 validation pairs.
- **Expected layout:**
  ```
  kaggle_3m/
    <patient_id>/
      <patient_id>_<slice>.tif
      <patient_id>_<slice>_mask.tif
      ...
  ```

## Model

A standard U-Net (Ronneberger et al., 2015):

- **Input:** 256×256×3 RGB
- **Encoder:** 4 downsampling blocks (64 → 128 → 256 → 512 filters), each with two `Conv2D` + `BatchNorm` + `ReLU`, followed by `MaxPooling2D`
- **Bridge:** 1024 filters, dropout 0.5
- **Decoder:** 4 upsampling blocks using `Conv2DTranspose`, with skip connections concatenated from the matching encoder stage
- **Output:** `Conv2D(1, 1, activation='sigmoid')` — per-pixel tumor probability
- Built-in `RandomFlip` / `RandomRotation` / `RandomZoom` augmentation layers applied at the input (active only during training)
- Trained under `tf.distribute.MirroredStrategy` for multi-GPU training (as on Kaggle's dual-GPU sessions)

**Loss:** a custom weighted Dice loss (`weighted_dice_loss`), which up-weights tumor (foreground) pixels by 20× to counter the strong class imbalance (tumor pixels are a small fraction of each slice).

**Metrics:** accuracy, IoU, precision, recall.

**Training config:** Adam (lr=1e-4), batch size 16, 25 epochs, `ReduceLROnPlateau` (factor 0.2, patience 5).

## Results

Final epoch (25/25) on the validation set:

| Metric | Value |
|---|---|
| Accuracy | 0.969 |
| IoU | 0.168 |
| Precision | 0.194 |
| Recall | 0.651 |
| Loss (weighted Dice) | 0.219 |

## Requirements

```
tensorflow>=2.15
numpy
pillow
matplotlib
```

Install with:
```bash
pip install tensorflow numpy pillow matplotlib
```

GPU is strongly recommended — this notebook was originally run on Kaggle with 2×T4 GPUs.

## Known Limitations

This is a working baseline, not a tuned, production-ready model. Worth knowing before building on it:

- **Segmentation quality is modest.** Final validation IoU (~0.17) and precision (~0.19) are low for this task — well-tuned U-Nets on this dataset typically reach substantially higher Dice/IoU. Recall (~0.65) is reasonable, meaning the model finds tumor regions but with many false positives, likely a side effect of the aggressive 20× foreground weighting in the loss.
- **No held-out test set.** Only a train/validation split is used; validation numbers double as the only evaluation, with no fully independent test set.
- **No model checkpointing.** The trained model isn't saved to disk (no `ModelCheckpoint` or `model.save()`), so a training run isn't reusable without re-running the whole notebook.
- **Hardcoded Kaggle path.** `base_dir` points at `/kaggle/input/...` and needs manual editing to run elsewhere.
- **Data loading uses `tf.py_function`** for TIFF decoding, which is simpler but slower than a pure `tf.data` pipeline — fine for experimentation, not ideal for large-scale training.
- **Some duplicated code** — the augmentation pipeline is defined twice (once standalone, once inline inside `create_unet_model`).

## Suggested Next Steps

- Tune or replace the foreground weight in the Dice loss (or try combined Dice + BCE / focal loss) to improve precision.
- Add a proper test split and report metrics on it.
- Add `ModelCheckpoint` / `model.save()` to persist trained weights.
- Threshold predictions explicitly (e.g., 0.5) before computing IoU/precision/recall to make metric semantics unambiguous.

## Acknowledgments

- Dataset: Mateusz Buda, Ashirbani Saha, Maciej A. Mazurowski — *"Association of genomic subtypes of lower-grade gliomas with shape features automatically extracted by a deep learning algorithm,"* Computers in Biology and Medicine, 2019.
- Architecture: Ronneberger, Fischer, Brox — *"U-Net: Convolutional Networks for Biomedical Image Segmentation,"* MICCAI 2015.

## License

No license specified. Add a `LICENSE` file (e.g., MIT, Apache-2.0) if you intend others to reuse this code.
