# Digits-Automated-speech-Recogntion-b4
# Speech Digit Recognition using CNN and MFCC Features with SpaCy

## Overview

This project performs speech digit recognition using a **Convolutional Neural Network (CNN)** with **MFCC features** extracted from audio files. An optional **SpaCy** step is included for text processing, which can be used to analyze speech-to-text data (if applicable).

The primary components of this project are:
1. **CNN Model**: Trained on MFCC features to recognize digits from speech.
2. **Data Preprocessing**: Extracts MFCC features from the audio data.
3. **Inference**: Runs predictions on unseen audio files and visualizes results.
4. **Workflow Visualization**: Generated using `graphviz` to represent the model workflow.

## Folder Structure


## Prerequisites

Before running the code, make sure to install the following dependencies:

- `numpy`
- `librosa`
- `matplotlib`
- `tensorflow`
- `spacy`
- `scikit-learn`
- `graphviz`
  
You can install these using `pip`:
pip install numpy librosa matplotlib tensorflow spacy scikit-learn graphviz

Additionally, you need to download the SpaCy model (if using text processing):
python -m spacy download en_core_web_sm
Instructions to Run the Code
Step 1: Prepare the Dataset
Place your audio files (in .wav format) in the data/recordings/ folder. The filenames should follow the format <digit>_<random_string>.wav, where <digit> represents the digit to be recognized (0-9).

Step 2: Data Preprocessing
Run the data_preprocessing.py script to load and extract MFCC features from the audio data.


python Codes/data_preprocessing.py

Step 3: Train the Model
To train the CNN model using the preprocessed MFCC features, run the model.py script:
python Codes/model.py


Step 4: Inference
To run inference on new audio files, use the inference.py script. It will load the trained model and make predictions on the provided audio files.
python Codes/inference.py

Step 5: Workflow Visualization
To generate the workflow diagram that visualizes the steps in the project, run the following:
python Codes/workflow_graph.py
