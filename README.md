# Intel Image Classification Project
Project Overview
This project implements a deep learning model for scene classification using the Intel Image Classification dataset. The goal is to accurately categorize images into six natural scenes: buildings, forest, glacier, mountain, sea, and street. The project includes data preprocessing, model training, and comprehensive evaluation with metrics such as accuracy, precision, recall, F1 score, and confusion matrix visualization. It provides a reproducible workflow and is ideal for experimenting with image classification techniques using PyTorch
## Execution Instructions

### 1. Clone or Download the Repository

Download or clone this repository to your local machine.

### 2. Install Python

Make sure you have **Python 3.8+** installed.  
Download from: https://www.python.org/downloads/

### 3. Install Required Packages

Open a terminal in the project directory and run:

    pip install torch torchvision scikit-learn matplotlib pillow

### 4. Download the Dataset

- Download the [Intel Image Classification dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) from Kaggle.
- Extract the dataset so that the folders `seg_train` and `seg_test` are inside:

     
  *(Ensure to update the paths in the scripts to your local machine stored location.)*

### 5. Preprocess and Verify the Data

Run the preprocessing script to verify your data and see sample batches:

    python preprocessing_images.py

### 6. Train and Evaluate the Model

Run the main training and evaluation script:

    python training_samples.py

- This will train the model, print training progress, show per-class accuracy (including percent correctly and incorrectly identified for each class), and display advanced evaluation metrics (accuracy, precision, recall, F1, confusion matrix, ROC/AUC, PR curves, and model parameter count).

### 7. Troubleshooting

- If you encounter errors about missing packages, double-check you installed all requirements.
- If you use a different dataset location, update the `train_dir` and `test_dir` variables in the scripts.

---

**Requirements:**
- Python 3.8+
- torch
- torchvision
- scikit-learn
- matplotlib
- pillow

---
**Tip:**  
If you want to use a GPU, make sure you have the correct version of PyTorch installed for CUDA. See [PyTorch Get Started](https://pytorch.org/get-started/locally/) for details.# intel-image-ml-project
