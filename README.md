Realtime Face Emotion Recognition

This project implements a system for real-time human face emotion recognition using computer vision techniques and a pre-trained deep learning model. The system processes live video input (e.g., from a webcam) to detect faces and classify the dominant emotion displayed.

🚀 Getting Started
Follow these steps to clone the project and run the real-time application on your local machine.

1. Clone the Project
To get a local copy of this repository, open your terminal or command prompt and run the following command:

git clone <URL of your GitHub repository>

cd RealtimeFaceEmotionRecognition


2. Run the Application
The model weights are already included in the repository (e.g., as a .h5 or .json/.weights file), so you can start the application immediately after cloning.

Run the main script using Python:

python main.py


Note: You may need to install necessary libraries (like TensorFlow, OpenCV, etc.) if you haven't already. It's recommended to use a virtual environment.

📚 Model Details

Dataset

The model used for emotion classification was trained on the FER-2013 dataset. This dataset is available on Kaggle and consists of 48x48 pixel grayscale images of faces, categorized into seven core emotional expressions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

Training

The process for building and training the convolutional neural network (CNN) model is fully documented in the following Jupyter Notebook:

trainer.ipynb: This notebook contains the full code used for data loading, preprocessing, model architecture definition, training, and saving the final model weights.
