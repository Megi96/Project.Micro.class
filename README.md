# 🧫 Microorganism Image Classification

This project uses a Convolutional Neural Network (CNN) built with PyTorch to classify images of microorganisms into predefined categories. It’s designed for researchers, students, or lab technicians working with microscopic image datasets.

## 📦 Features
- Loads and preprocesses image data using `torchvision`
- Splits dataset into training and validation sets
- Defines a simple CNN architecture for feature extraction and classification
- Trains the model and tracks accuracy across epochs
- Saves the best-performing model based on validation accuracy
- Visualizes training progress and sample predictions

## 🧠 Model Overview
- Input: RGB images resized to 150×150 pixels  
- Architecture: 3 convolutional layers + 2 fully connected layers  
- Output: Class probabilities for each microorganism type

## 📁 Dataset
Place your labeled image folders inside:  
`C:\Users\C.S.T\Downloads\Micro_Organism`  
Each subfolder should represent one class (e.g., `Bacteria`, `Fungi`, etc.).

## 🚀 How to Run
1. Install dependencies: `torch`, `torchvision`, `matplotlib`
2. Run the script to train the model
3. View saved outputs:
   - Best model: `best_micro_model.pth`
   - Accuracy plot: `training_progress.png`
   - Sample predictions: `sample_predictions.png`

## 📊 Results
The script prints per-class accuracy and highlights correct vs. incorrect predictions in green/red for easy visual inspection.

