Brain Tumor Detection CNN Model

This model was trained using a Convolutional Neural Network (CNN) to classify brain MRI images as either having a tumor or not. It uses Keras with TensorFlow backend and was trained on the publicly available Brain Tumor MRI Dataset from Kaggle. Dataset

The dataset contains 3,762 T1-weighted contrast-enhanced MRI images, labeled as:

Yes – Images with a brain tumor
No – Images without a brain tumor
The data is balanced and preprocessed into two folders: yes/ and no/.

Train Accuracy: ~98% Validation Accuracy: ~96%

🧠 Model Architecture
Type: CNN Transfer Learning
Framework: Keras (TensorFlow backend)
Input shape: (150, 150, 3)
Final Activation: sigmoid
Loss: binary_crossentropy
Optimizer: Adam
