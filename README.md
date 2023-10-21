# Galaxy Morphology Classification Using Transfer Learning

This repository contains Python code for classifying galaxies based on their morphologies using transfer learning with PyTorch. The code is designed to help you train a deep learning model on your own dataset of galaxy images and make predictions.

## Overview

This project aims to classify galaxies into different categories based on their morphological features. It uses a pre-trained ResNet-18 model as a feature extractor and a custom fully connected layer for classification.

## Getting Started

To get started with this project, follow these steps:

### Prerequisites

1. Install Python: Make sure you have Python 3.x installed on your system.

2. Install Required Packages: You can install the required packages using the following command:

   ```
   pip install torch torchvision numpy pillow
   ```

### Usage

1. Clone the Repository: Clone this repository to your local machine using Git.

   ```
   git clone https://github.com/your-username/galaxy-morphology-classification.git
   cd galaxy-morphology-classification
   ```

2. Data Preparation:

   - Organize your galaxy images into different folders, each representing a category of galaxies. The default folder structure is:
     ```
     Data/
     ├── Category1/
     │   ├── image1.jpg
     │   ├── image2.jpg
     │   └── ...
     ├── Category2/
     │   ├── image1.jpg
     │   ├── image2.jpg
     │   └── ...
     └── ...
     ```
   - If you want to use your own folder structure, make sure to adjust the `CustomDataset` class accordingly.

3. Train or Predict:

   - Run the script and choose whether you want to train or predict:

     ```
     python galaxy_classification.py
     ```

   - If you choose to train, the script will train a model on your dataset using transfer learning. The best model will be saved to the `Best_TransferLearning_model.pth` file.

   - If you choose to predict, the script will use the trained model to make predictions on a test dataset and save the results in a CSV file named `predictions.csv`.

## Hyperparameters

You can adjust the following hyperparameters in the code:

- `batch_size`: Batch size for training.
- `num_epochs`: Number of training epochs.
- `learning_rate`: Learning rate for optimization.

## Model

The code uses a pre-trained ResNet-18 model as a feature extractor. You can modify the model architecture by changing the `model.fc` layer for your specific classification needs.

## Data Augmentation

Data augmentation techniques are applied to the training dataset to improve model generalization. You can modify the `train_transform` and `test_transform` to adjust the data augmentation and normalization methods.

## Author

- Ninan Sajeeth Philip (Email: nsp@airis4d.com)

## License

This code is provided under an open-source license. You are free to use and modify it for your own purposes.

Please feel free to reach out if you have any questions or need further assistance.

---

*Note: Ensure that you have the necessary permissions and rights to use the data for your specific application, especially if it is for research or commercial use.*
