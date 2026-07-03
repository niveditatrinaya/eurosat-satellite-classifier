# EuroSAT Satellite Image Classifier

A convolutional neural network for land-use classification from satellite imagery, built using transfer learning on ResNet50. The model classifies Sentinel-2 satellite images into 10 land-cover categories and reaches 98% accuracy on held-out test data.

This project was built as a warm-up exercise before a larger portfolio project, The goal was to get comfortable with the full pipeline : data handling, transfer learning, training diagnostics, and evaluation on a well-understood dataset before applying the same workflow to a harder, less forgiving problem.

## Dataset

[EuroSAT](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset), sourced via Kaggle: 27,000 labeled Sentinel-2 satellite images spanning 10 land-use classes — AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, and SeaLake.

Images are natively 64x64 pixels and were resized to 224x224 to match ResNet50's expected input dimensions. Standard ImageNet normalization (mean/std) was applied, since the backbone was pretrained on ImageNet and expects inputs in that distribution.

The dataset was split 80/10/10 into training, validation, and test sets (21,600 / 2,700 / 2,700 images respectively), using a fixed random seed for reproducibility. Training data was augmented with random horizontal and vertical flips; validation and test data were left unaugmented to keep evaluation consistent.

## Model and training approach

The model is a ResNet50 backbone pretrained on ImageNet (`IMAGENET1K_V2` weights), with the final fully-connected layer replaced to output 10 classes instead of ImageNet's 1000.

Training was done in two phases rather than fine-tuning the entire network from the start:

**Phase 1 — feature extraction.** The pretrained backbone was frozen entirely, and only the newly added final layer was trained. This lets the randomly-initialized classification head adjust to the new task without disturbing the pretrained convolutional features. Trained for 10 epochs using Adam (lr=0.001).

**Phase 2 — fine-tuning.** All layers were unfrozen and trained together at a substantially lower learning rate (Adam, lr=0.0001), for 5 epochs. This allows the pretrained features to adapt slightly to satellite imagery specifically, while the smaller learning rate limits how much they're allowed to shift, reducing the risk of destroying useful pretrained representations.

Model checkpointing was used during fine-tuning: after each epoch, the model's weights were saved to disk only if that epoch's validation accuracy exceeded the previous best. This ensures the final saved model reflects genuine peak performance rather than whatever the last epoch happened to produce, which matters here since validation accuracy in phase 2 doesn't necessarily improve monotonically.

Training was run on Colab's free-tier T4 GPU, using automatic mixed precision (`torch.amp`) to reduce memory usage and training time without a meaningful accuracy tradeoff.

## Results

Phase 1 (frozen backbone) converged to approximately 94% validation accuracy after 10 epochs. Fine-tuning in phase 2 improved this to a peak of 98.37% validation accuracy.

Final evaluation was run on the test set — 2,700 images held out entirely from both training and validation — to get an unbiased estimate of generalization performance:

- **Overall test accuracy: 98%**
- Per-class precision and recall range from 0.96 to 1.00, with no class performing substantially worse than the rest
- The confusion matrix shows that nearly all misclassifications occur between visually similar categories: River and Highway (both long, linear features), Industrial and Residential (both dense built-up areas), and Pasture, AnnualCrop, and HerbaceousVegetation (all variants of open vegetated land). No confusion occurs between visually or semantically unrelated classes, which suggests the model has learned meaningful, generalizable features rather than superficial shortcuts.

## Project status

Data pipeline, model architecture, both training phases, checkpointing, and full test-set evaluation are complete.

## Tech stack

PyTorch, torchvision (ResNet50, transforms, ImageFolder), scikit-learn (evaluation metrics), Google Colab (T4 GPU).
