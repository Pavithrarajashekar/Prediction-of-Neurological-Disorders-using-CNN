# Prediction-of-Neurological-Disorders-using-CNN
This project presents a deep learning approach to classify brain MRI scans into five categories: Alzheimer’s, Parkinson’s, Meningioma, Glioma, and Healthy. We leverage Convolutional Neural Networks (CNNs), including GoogLeNet, ResNet70, and VGG16, to develop an automated and accurate diagnostic model for neurological disorder detection.

🧬 Dataset:
The dataset contains brain MRI images categorized into five classes:

🧠 Alzheimer
🧠 Parkinson
🧠 Meningioma
🧠 Glioma
✅ Healthy.
All images are preprocessed and resized to a consistent shape. 

⚙️ Preprocessing:
Image resizing to 224x224.
Normalization to [0,1].
Train/Validation/Test split (80/10/10).
Augmentations: Rotation, flip, color jitter.

🏗 Model Architecture:
Pretrained GoogLeNet (Inception v1).
Final layer replaced with nn.Linear for 5-class classification.
Trained with CrossEntropyLoss and Adam Optimizer.

🏃 Training:
Model is trained for 5 separate runs
20 epochs per run
Best model weights saved per run
Loss curves are plotted and saved

📈 Evaluation:
Test Accuracy-97.52%
Classification Report (Precision, Recall, F1-Score)
Standard Deviation of predictions across classes-0.62
Confusion Matrix 
Grouped bar chart for precision, recall, and F1
Accuracy and Loss Plots

✅ Features:
Google Drive Integration for large dataset handling.
Dynamic directory creation and zero-length file filtering.
Easy customization for different models or additional classes.
Modular functions for preprocessing, training, evaluation.

This Flask-based web app uses a pretrained GoogLeNet model to predict neurological disorders (Alzheimer's, Glioma, Meningioma, Parkinson's, or Normal) from brain MRI images.

🚀 How It Works:
User uploads an MRI image via the web interface.
The image is preprocessed (resized, normalized).
A trained GoogLeNet model performs inference.
The predicted disorder is returned.



