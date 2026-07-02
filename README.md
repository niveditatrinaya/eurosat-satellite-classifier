# EuroSAT Satellite Image Classifier
Fine-tuning a pretrained ResNet50 to classify satellite images into 10 land-use types (forest, farmland, urban, water, etc.) using the EuroSAT dataset. Built as a warm-up project for a bigger deep learning project (gravitational lens detection).

## Dataset
EuroSAT dataset — 27,000 satellite images (64x64 px) across 10 classes: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake.

Split: 21,600 train / 2,700 validation / 2,700 test.

## Approach
1. Load pretrained ResNet50, freeze it, replace final layer to output 10 classes
2. Train just the new final layer
3. Unfreeze the whole network and fine-tune at a lower learning rate
4. Evaluate on the test set

Trained in PyTorch on Google Colab (T4 GPU).

## Status

- [x] Data pipeline
- [x] Model setup
- [x] Phase 1 training
- [ ] Fine-tuning
- [ ] Test evaluation
