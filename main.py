import cv2
import numpy as np
import os
from keras.models import load_model
from tkinter import filedialog
import tkinter as tk

def load_emotion_model():
    """Load the pre-trained emotion detection model"""
    try:
        model = load_model("emotion_model.h5")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def detect_emotions_in_frame(frame, model, face_cascade, emotions):
    """Process a single frame for emotion detection"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]  # Fixed: was x:x+h, should be x:x+w
        roi = cv2.resize(roi, (48, 48)) / 255.0
        roi = roi.reshape(1, 48, 48, 1)

        prediction = model.predict(roi, verbose=0)
        emotion = emotions[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"{emotion} ({confidence:.1f}%)", 
                   (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame

def webcam_detection(model, face_cascade, emotions):
    """Real-time emotion detection from webcam"""
    print("Starting webcam detection. Press 'q' to quit.")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam")
            break
            
        frame = detect_emotions_in_frame(frame, model, face_cascade, emotions)
        cv2.imshow("Emotion Detector - Webcam", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def image_detection(model, face_cascade, emotions):
    """Emotion detection from image file"""
    # Hide the main tkinter window
    root = tk.Tk()
    root.withdraw()
    
    # Open file dialog
    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
            ("All files", "*.*")
        ]
    )
    
    if not file_path:
        print("No file selected.")
        return
    
    # Load and process image
    image = cv2.imread(file_path)
    if image is None:
        print("Error: Could not load image")
        return
    
    # Resize image if too large for display
    height, width = image.shape[:2]
    if width > 1200 or height > 800:
        scale = min(1200/width, 800/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height))
    
    processed_image = detect_emotions_in_frame(image.copy(), model, face_cascade, emotions)
    
    print("Displaying image. Press any key to close.")
    cv2.imshow("Emotion Detector - Image", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def video_detection(model, face_cascade, emotions):
    """Emotion detection from video file"""
    # Hide the main tkinter window
    root = tk.Tk()
    root.withdraw()
    
    # Open file dialog
    file_path = filedialog.askopenfilename(
        title="Select a video",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.webm"),
            ("All files", "*.*")
        ]
    )
    
    if not file_path:
        print("No file selected.")
        return
    
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    print("Processing video. Press 'q' to quit, 'SPACE' to pause/resume.")
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video reached.")
                break
            
            frame = detect_emotions_in_frame(frame, model, face_cascade, emotions)
            cv2.imshow("Emotion Detector - Video", frame)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # Space bar to pause/resume
            paused = not paused
            print("Video paused" if paused else "Video resumed")

    cap.release()
    cv2.destroyAllWindows()

def display_menu():
    """Display the main menu"""
    print("\n" + "="*50)
    print("        EMOTION DETECTION SYSTEM")
    print("="*50)
    print("1. Webcam (Real-time detection)")
    print("2. Image file")
    print("3. Video file")
    print("4. Exit")
    print("="*50)

def main():
    """Main function to run the emotion detection system"""
    # Load model and cascade classifier
    print("Loading emotion detection model...")
    model = load_emotion_model()
    if model is None:
        print("Failed to load model. Please ensure 'emotion_model.h5' exists.")
        return
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]  # Fixed typo: "Happpy" -> "Happy"
    
    print("Model loaded successfully!")
    
    while True:
        display_menu()
        
        try:
            choice = input("Enter your choice (1-4): ").strip()
            
            if choice == '1':
                webcam_detection(model, face_cascade, emotions)
            elif choice == '2':
                image_detection(model, face_cascade, emotions)
            elif choice == '3':
                video_detection(model, face_cascade, emotions)
            elif choice == '4':
                print("Thank you for using Emotion Detection System!")
                break
            else:
                print("Invalid choice. Please enter 1, 2, 3, or 4.")
                
        except KeyboardInterrupt:
            print("\nProgram interrupted by user.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()