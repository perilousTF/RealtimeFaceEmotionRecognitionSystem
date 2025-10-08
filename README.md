# Realtime Face Emotion Recognition

This project implements a system for real-time human face emotion recognition using computer vision techniques and a pre-trained deep learning model. The system processes live video input (e.g., from a webcam) to detect faces and classify the dominant emotion displayed.

## 🚀 Getting Started

Follow these steps to clone the project and run the real-time application on your local machine.

### 1. Clone the Project

To get a local copy of this repository, open your terminal or command prompt and run the following command:

```bash
git clone https://github.com/perilousTF/RealtimeFaceEmotionRecognitionSystem.git
cd RealtimeFaceEmotionRecognition
```

### 2. Run the Application

The model weights are already included in the repository (e.g., as a `.h5` or `.json/.weights` file), so you can start the application immediately after cloning.

Run the main script using Python:

```bash
python main.py
```

**Note:** You may need to install necessary libraries (like TensorFlow, OpenCV, etc.) if you haven't already. It's recommended to use a virtual environment.

## 📚 Model Details

### Dataset

The model used for emotion classification was trained on the **FER-2013 dataset**. This dataset is available on Kaggle and consists of 48x48 pixel grayscale images of faces, categorized into seven core emotional expressions:

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

### Training

The process for building and training the convolutional neural network (CNN) model is fully documented in the following Jupyter Notebook:

* `trainer.ipynb`: This notebook contains the full code used for data loading, preprocessing, model architecture definition, training, and saving the final model weights.

## 📋 Requirements

- Python 3.x
- TensorFlow
- OpenCV
- NumPy
- Other dependencies (see `requirements.txt` if available)

## 🎯 Features

- Real-time face detection
- Emotion classification into 7 categories
- Live webcam feed processing
- Pre-trained model included

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 👤 Author

Your Name - [Your GitHub Profile](https://github.com/perilousTF)

## 🙏 Acknowledgments

- FER-2013 Dataset from Kaggle
- OpenCV for computer vision capabilities
- TensorFlow for deep learning framework
