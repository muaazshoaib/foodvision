# FoodVision: Multi-Class Food Image Classifier using TensorFlow

This project demonstrates the use of **transfer learning** to classify images of food into 101 different categories using a deep learning model built with TensorFlow and Keras.

## Table of Contents
- [Overview](#overview)
- [Technologies](#technologies)
- [Dataset](#dataset)
- [Approach](#approach)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Overview
**FoodVision** is a computer vision project where a convolutional neural network (CNN) is trained to classify food images into 101 categories using transfer learning.

## Technologies
- Python
- TensorFlow & Keras
- EfficientNetB0
- TensorBoard for model tracking
- Matplotlib, NumPy, Pandas

## Dataset
The project uses the [Food101 dataset](https://www.vision.ee.ethz.ch/datasets_extra/food-101/), which contains 101,000 images of food across 101 classes (750 training, 250 testing per class).

## Approach
1. Loaded and prepared the dataset using TensorFlow Datasets (TFDS).
2. Applied data preprocessing and augmentation (rescaling, flipping, rotating).
3. Built and fine-tuned a transfer learning model using **EfficientNetB0**.
4. Tracked model performance using **TensorBoard** and implemented **EarlyStopping** and **ModelCheckpoint**.
5. Evaluated the model on the test set and analyzed classification metrics.

## Model Architecture
- EfficientNetB0 base model (pretrained on ImageNet)
- GlobalAveragePooling2D
- Dense output layer with softmax activation

## Installation
```bash
git clone https://github.com/muaazshoaib/foodvision-tensorflow.git
cd foodvision-tensorflow
pip install -r requirements.txt
