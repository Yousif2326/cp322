# Pneumonia Detection from Chest X-rays

This project implements and compares multiple deep learning models for detecting pneumonia from chest X-ray images using transfer learning and data augmentation techniques.

## Project Overview

The goal of this project is to build and evaluate different deep learning models for binary classification of chest X-ray images into **Normal** and **Pneumonia** classes. The project compares baseline models with transfer learning approaches and analyzes the impact of data augmentation on model performance.

## Models Evaluated

Four different models were trained and evaluated:

1. **SmallCNN** - A simple convolutional neural network trained from scratch
2. **ResNet50_Aug** - Pre-trained ResNet-50 fine-tuned with data augmentation
3. **ResNet50_NoAug** - Pre-trained ResNet-50 fine-tuned without data augmentation
4. **ViT_Tiny** - Vision Transformer (ViT-Tiny) fine-tuned from pre-trained weights

## Dataset

- **Training Set**: 200 images (100 per class, stratified sampling)
- **Validation Set**: 16 images (8 per class)
- **Test Set**: 100 images (50 per class)
- **Original Dataset**: Chest X-ray dataset with 3,875 pneumonia and 1,341 normal training images

## Results Summary

### Performance Metrics

Based on the evaluation metrics and visualizations:

| Model | Accuracy | AUC | Key Characteristics |
|-------|----------|-----|---------------------|
| **ViT_Tiny** | **~0.83** | **0.96** | Best overall performance, balanced predictions |
| **ResNet50_Aug** | ~0.57 | 0.93 | High AUC but class imbalance issues |
| **ResNet50_NoAug** | ~0.52 | 0.93 | High AUC but severe class bias |
| **SmallCNN** | ~0.74 | 0.82 | Moderate performance, more balanced than ResNet50 |

### Detailed Model Analysis

#### 1. ViT_Tiny (Best Performer)
- **Confusion Matrix**:
  - True Negatives: 34, False Positives: 16
  - True Positives: 49, False Negatives: 1
- **Performance**:
  - Highest accuracy (~83%) and AUC (0.96)
  - Excellent recall for pneumonia class (49/50 correct)
  - Good precision with only 16 false positives
- **Training Curves**:
  - Smooth convergence with training and validation metrics closely aligned
  - No significant overfitting observed
  - Validation accuracy reaches ~87.5% by epoch 5

#### 2. ResNet50_Aug (With Data Augmentation)
- **Confusion Matrix**:
  - True Negatives: 7, False Positives: 43
  - True Positives: 50, False Negatives: 0
- **Performance**:
  - High AUC (0.93) but low accuracy (~57%)
  - Perfect recall for pneumonia (50/50) but poor precision for normal class
  - Strong bias towards predicting pneumonia
- **Training Curves**:
  - Training loss decreases steadily
  - Validation loss shows instability around epoch 1.5-2
  - Evidence of overfitting despite augmentation

#### 3. ResNet50_NoAug (Without Data Augmentation)
- **Confusion Matrix**:
  - True Negatives: 2, False Positives: 48
  - True Positives: 50, False Negatives: 0
- **Performance**:
  - High AUC (0.93) but lowest accuracy (~52%)
  - Severe class imbalance - predicts pneumonia for almost all cases
  - Only 2 correct normal predictions out of 50
- **Training Curves**:
  - Training loss drops to near zero (overfitting)
  - Validation loss increases significantly after epoch 1.5
  - Clear overfitting pattern

#### 4. SmallCNN (Baseline)
- **Confusion Matrix**:
  - True Negatives: 42, False Positives: 8
  - True Positives: 32, False Negatives: 18
- **Performance**:
  - Moderate accuracy (~74%) and AUC (0.82)
  - Better balance between classes compared to ResNet50 variants
  - More false negatives (18) but fewer false positives (8)
- **Training Curves**:
  - Stable training with training and validation metrics tracking closely
  - Less overfitting compared to ResNet50 models

## Key Findings

### 1. **Vision Transformer Outperforms CNN Architectures**
   - ViT_Tiny achieved the best balance between accuracy and AUC
   - Better generalization and less prone to overfitting
   - More balanced predictions across both classes

### 2. **ResNet50 Models Show Class Imbalance Issues**
   - Both ResNet50 variants developed a strong bias towards predicting pneumonia
   - Despite high AUC scores, practical accuracy is low due to class imbalance
   - This suggests the models learned to maximize recall for pneumonia at the expense of normal class predictions

### 3. **Data Augmentation Impact**
   - ResNet50_Aug showed slightly better performance than ResNet50_NoAug (7 vs 2 correct normal predictions)
   - However, augmentation did not fully resolve the class imbalance problem
   - Both ResNet50 variants still suffer from overfitting

### 4. **Baseline Model Performance**
   - SmallCNN provides a reasonable baseline with balanced predictions
   - While accuracy and AUC are lower than transfer learning models, it shows more consistent behavior
   - Less prone to extreme class bias

### 5. **ROC Curve Analysis**
   - All models show good discriminative ability (AUC > 0.82)
   - ViT_Tiny has the highest AUC (0.96), indicating excellent separation between classes
   - High AUC scores for ResNet50 models are misleading given their poor accuracy

## Visualizations

The project includes comprehensive visualizations in the `Images/` directory:

- **Confusion Matrices**: For each model showing classification breakdown
- **Training Curves**: Loss and accuracy plots for training and validation sets
- **ROC Curves**: Receiver Operating Characteristic curves with AUC scores
- **Grad-CAM Visualization**: Attention heatmap for ResNet50 model
- **Comparison Bar Plot**: Side-by-side comparison of accuracy and AUC across all models
- **Dashboard**: Comprehensive overview of all results

## Model Interpretability

- **Grad-CAM Visualization**: Generated for ResNet50 to visualize which regions of the chest X-ray images the model focuses on when making predictions
- The heatmap shows the model's attention regions, providing insights into decision-making

## Recommendations

1. **Use ViT_Tiny for Production**: Best overall performance with balanced predictions
2. **Address Class Imbalance**: ResNet50 models would benefit from:
   - Class weighting in loss function
   - Focal loss to handle imbalanced data
   - More aggressive data augmentation for minority class
3. **Further Training**: Consider training for more epochs with early stopping for ResNet50 models
4. **Ensemble Methods**: Could combine predictions from multiple models for improved robustness



## Training Configuration



- **Epochs**: 5
- **Batch Size**: 32
- **Learning Rate**: 1e-4
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss
- **Image Size**: 224×224
- **Data Augmentation** (when applied):
  - Random horizontal flip
  - Random rotation (±10°)
  - Color jitter (brightness: 0.2)


## Usage

### Prerequisites

Before running the demo, ensure you have the required dependencies installed:

```bash
pip install torch torchvision gradio timm pillow
```

### Running the Demo

The project includes a Gradio-based demo interface (`code/Demo.py`) that allows interactive testing of all trained models.

#### Quick Start


1. **Run the demo:**
   ```bash
   python code/Demo.py
   ```

2. **Access the interface:**
   - The demo will start a local web server
   - A URL will be displayed in the terminal (typically `http://127.0.0.1:7860`)
   - Open this URL in your web browser
   - A public shareable link will also be generated if `share=True` is enabled

#### Demo Features

The interactive demo provides:
- **Image Upload**: Drag and drop or browse to upload chest X-ray images
- **Model Selection**: Choose from all 4 trained models:
  - SmallCNN
  - ResNet50_Aug
  - ResNet50_NoAug
  - ViT_Tiny (Recommended - best performance)
- **Real-time Predictions**:
  - Predicted class (Normal or Pneumonia)
  - Confidence score (percentage)
  - Model used for prediction

#### Model Paths

The demo automatically detects model files in the `Models/` directory:
- `Models/SmallCNN.pth`
- `Models/ResNet50_Aug.pth`
- `Models/ResNet50_NoAug.pth`
- `Models/ViT_Tiny.pth`



## Conclusion

This project demonstrates that Vision Transformers can outperform traditional CNN architectures for medical image classification tasks. The ViT_Tiny model achieved the best balance of accuracy and AUC while maintaining good class balance. The ResNet50 models, despite high AUC scores, suffer from class imbalance issues that make them less practical for real-world deployment.

The comprehensive evaluation including confusion matrices, ROC curves, training curves, and attention visualizations provides deep insights into model behavior and decision-making processes.
