# Face Recognition Project

This project demonstrates the implementation of face recognition using OpenCV and K-Nearest Neighbors (KNN). It consists of three main parts:
1. **face_data_collect.py** - Collects face data and stores it in the form of NumPy arrays.
2. **face_recognition.py** - Recognizes faces from a live webcam feed using KNN.
3. **KNN Classifier** - A custom implementation of the K-Nearest Neighbors algorithm used for face recognition.

## Prerequisites

Ensure you have the following libraries installed:
- `opencv-python`
- `numpy`
- `matplotlib` (for visualizations)
- `os`

You can install them using pip:

pip install opencv-python numpy matplotlib

Setup
1. Face Data Collection (face_data_collect.py)
This script is used to collect face data and save it as .npy files.

Run the script to capture images of your face via webcam.
The captured images will be stored in the ./data/ folder as .npy files.
Make sure you have a folder named data where the face images will be stored.
To run the script:

python face_data_collect.py
You will be prompted to enter the name of the person whose face data is being collected.

2. Face Recognition (face_recognition.py)
This script performs live face recognition using the webcam feed and KNN.

It loads the previously collected face data from the ./data/ folder.
It detects faces using Haar Cascade Classifier and predicts the identity using KNN.
To run the script:

python face_recognition.py
It will display the webcam feed with the detected face and the name of the recognized person.

3. KNN Algorithm
The KNN algorithm calculates the Euclidean distance between the test face and the stored faces in the dataset and classifies the face based on the majority of the nearest neighbors. It is implemented in the knn() function in the face_recognition.py script.

Haar Cascade Classifier
The face detection is done using OpenCV's pre-trained Haar Cascade Classifier, which is stored in the haarcascade_frontalface_alt.xml file.

Folder Structure
Face-Recognition-Project/
│
├── data/                           # Folder to store .npy files with face data
│
├── haarcascade_frontalface_alt.xml  # Haar Cascade Classifier for face detection
├── face_data_collect.py             # Script to collect and save face data
├── face_recognition.py              # Script for face recognition using webcam
└── README.md                        # Project documentation
