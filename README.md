# Pose Estimation Project Documentation

## Overview

The Pose Estimation project is designed for predicting the orientation angles (roll, pitch, and yaw) of human faces in images and videos. It encompasses a series of scripts and Jupyter notebooks for data collection, model training, and deployment using facial landmarks obtained from the [MediaPipe Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh) model.

## Project Components

### 1. Data Collection (`data collection.ipynb`)

The data collection notebook is the initial phase of the project, responsible for gathering facial landmarks and orientation angles from a dataset of images. Key functionalities include:

- Utilizing the MediaPipe Face Mesh model for extracting facial landmarks from images.
- Reading pose parameters (yaw, pitch, roll) from corresponding `.mat` files.
- Organizing the data into a structured Pandas DataFrame.
- Saving the DataFrame as a CSV file (`face_landmarks.csv`).

### 2. Model Training (`models.ipynb`)

The model training notebook involves training regression models to predict the roll, pitch, and yaw angles based on the extracted facial landmarks. Key functionalities include:

- Loading the facial landmarks data from `face_landmarks.csv`.
- Applying Principal Component Analysis (PCA) for dimensionality reduction.
- Training multiple regression models, including XGBoost, RandomForest, SVR, GradientBoosting, AdaBoost, and a VotingRegressor.
- Evaluating the models using mean squared error.
- Saving the trained models and PCA object as joblib files.

### 3. Preprocessing and Utility Functions (`preprocessing.py`)

The `preprocessing.py` script contains utility functions used throughout the project. Key functionalities include:

- Drawing 3D axes on facial images to visualize orientation.
- Processing images using the MediaPipe Face Mesh model.
- Extracting facial landmarks and normalizing their coordinates.
- Processing images with PCA and predicting orientation angles.

### 4. Deployment (`deployment.py`)

The deployment script utilizes the Streamlit library to create a web application for pose estimation. It includes functionalities for:

- Loading the trained models and PCA object.
- Processing images and videos to predict orientation.
- Displaying the processed images with orientation axes.

### 5. Testing and Prediction (`test_predict.py`)

The `test_predict.py` script demonstrates how to test the trained models on a set of images and a video. Key functionalities include:

- Loading the trained models and PCA object.
- Iterating through a directory of test images, predicting orientation, and displaying the results.
- Processing a video, predicting orientation for each frame, and saving the output as a new video.

## Usage

### Data Collection

1. Open and run the `data collection.ipynb` Jupyter notebook.
2. Specify the path to the dataset folder containing images and corresponding `.mat` files.
3. Execute the notebook to collect facial landmarks and orientation angles, saving the data as `face_landmarks.csv`.

### Model Training

1. Open and run the `models.ipynb` Jupyter notebook.
2. Load the `face_landmarks.csv` file.
3. Perform PCA for dimensionality reduction and plot cumulative explained variance.
4. Train various regression models, evaluate performance, and save the models and PCA object.

### Deployment

1. Run the `deployment.py` script using Streamlit (`streamlit run deployment.py`).
2. Choose between uploading an image or a video.
3. Upload the file, and the web application will display the processed image or video with predicted orientation axes.

### Testing and Prediction

1. Open and run the `test_predict.py` script.
2. Load the trained models and PCA object.
3. Specify the path to a directory containing test images or a video.
4. Run the script to predict orientation for each image or frame and visualize the results.

## Dependencies

Ensure that the following dependencies are installed in your Python environment before running the code:

- OpenCV
- NumPy
- MediaPipe
- Matplotlib
- Pandas
- Streamlit
- Joblib
- Scikit-learn
- XGBoost

## Conclusion

The Pose Estimation project provides an end-to-end solution for predicting facial orientation angles in images and videos. It includes data collection, model training, deployment, and testing functionalities, making it a comprehensive and versatile tool for pose estimation tasks. The modular structure of the project allows for easy adaptation and extension for different use cases.



## break down the code step by step:

1. preprocessing.py
This script contains utility functions for processing images, drawing 3D axes on images, and extracting facial landmarks.

Functions:
draw_axis(img, pitch, yaw, roll, tdx, tdy, size): Draws 3D axes on an image based on the provided pitch, yaw, and roll angles. The optional tdx and tdy parameters specify the translation of the axes origin, and size sets the length of the axes.

process_image(image_path, faces): Reads an image from the specified path, converts it to RGB format, and processes it using the MediaPipe Face Mesh model (faces).

extract_landmarks(results, imgname, xy, nose): Extracts facial landmarks from the processed results and normalizes their coordinates based on the nose landmark.

process_jpg(imgname, xy, nose): Processes a JPEG image by applying the process_image and extract_landmarks functions.

process_mat(imgname, pitch, yaw, roll): Loads pose parameters (pitch, yaw, roll) from a MATLAB (.mat) file and appends them to the respective lists.

process_image_with_pca(image, pca): Processes an image with PCA dimensionality reduction, returning the transformed points and the nose landmark.

predict_on_image(image, Modelroll, Modelpitch, Modelyaw, pca): Predicts roll, pitch, and yaw angles for a given image using trained regression models and PCA.

predict_on_video2(video_path, output_path, votingModelroll, votingModelpitch, votingModelyaw, pca): Processes a video, predicts orientation for each frame, draws axes, and saves the output as a new video.

2. models.ipynb
This Jupyter notebook covers the model training phase, including loading the facial landmarks data, applying PCA, training various regression models, and evaluating their performance.

Steps:
Load Data: Reads the facial landmarks data from face_landmarks.csv.

PCA Dimensionality Reduction: Applies PCA to reduce the dimensionality of the data.

Explained Variance Plot: Plots the cumulative explained variance to determine the number of components for PCA.

Model Training: Trains XGBoost, RandomForest, SVR, GradientBoosting, AdaBoost, and a VotingRegressor using the PCA-transformed data.

Model Evaluation: Evaluates the models using mean squared error on a test set.

Model Saving: Saves the trained models and PCA object as joblib files.

3. data collection.ipynb
This notebook focuses on data collection, extracting facial landmarks from images, and gathering pose parameters from MATLAB files.

Steps:
Folder Path: Sets the path to the dataset folder containing images and corresponding .mat files.

Data Collection: Iterates through files in the specified folder, processes JPG images, and extracts pose parameters from MATLAB files. Organizes the data into a Pandas DataFrame and saves it as face_landmarks.csv.

4. deployment.py
This script uses Streamlit to deploy the model and create a web application for pose estimation.

Steps:
Model Loading: Loads the trained models (VotingRegressor) and PCA object.

File Upload: Allows users to upload an image or a video.

Image Processing: Processes the uploaded file, predicts orientation, and displays the processed image with orientation axes.

Video Processing: Processes the uploaded video, predicts orientation for each frame, and displays the processed video with orientation axes.

5. test_predict.py
This script demonstrates testing the trained models on a set of test images and predicting orientation for a video.

Steps:
Model Loading: Loads the trained models and PCA object.

Test Images: Iterates through a directory of test images, predicts orientation, and displays the results.

Video Prediction: Processes a video, predicts orientation for each frame, and saves the output as a new video.