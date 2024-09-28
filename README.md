# Sound-Detection-and-Alert-System-for-Unauthorized-Guests-in-Children-s-Parks-Using-Deep-Learning


This research aims to develop a sound-based alert system that plays a crucial role in ensuring the safety of children. Specifically, the goal is to alert security personnel, teachers, and caregivers in situations where adult human and animal sounds are detected in areas such as nurseries and places where children are present. To achieve this, a CNN architecture has been designed using deep learning methods. The project focuses on the process of creating a sound classification model based on the UrbanSound8K dataset, which contains 8,732 audio files recorded and labeled from different environments.

The project utilizes the Librosa library to extract features from audio files. Specifically, features like MFCC (Mel-frequency cepstral coefficients) are extracted to make the audio files suitable for machine learning algorithms. The dataset is split into training and testing datasets for model training. The model trained on the training dataset is evaluated on the testing dataset. The output of the model is the classification of the audio files. A special feature of the model is its ability to identify dangerous sounds, such as dog barking, and send alert messages. Additionally, it can classify any external sound it detects.

The social impact of the project is to enhance children's safety, providing reassurance to families. Furthermore, it contributes to education and scientific research by demonstrating how deep learning techniques can be used to solve real-world problems. Economically, the increasing availability of such technologies may create new job opportunities and economic value. This project may also contribute to the development of new applications in smart city activities and security systems.

In conclusion, this project represents a significant tool for child safety through sound recognition and classification, showcasing the potential of deep learning techniques. This work provides a valuable tool for professionals working in the field of security to automatically detect threats.

## Keywords: 
Voice detection, Deep Learning, Voice Processing, Python, CNN model

# Project Topic:
Sound Detection and Alert System for Unauthorized Visitors in Children's Parks using Deep Learning

# Project Modules:
- Anaconda
- Python
- Discrete Fourier Transform (DFT)
- CNN ( Convolutional Neural Network ) Model
- TensorFlow and Keras
- Librosa
- Jupyter Notebook




# Description

In our project, the method is particularly important for notifying security personnel, teachers, and caregivers with alerts based on news and sound when there are unauthorized situations, especially in nurseries and areas where children are playing, involving adult human voices and animal sounds. Deep Learning, more specifically machine learning, develops numerous approaches to solve artificial intelligence problems. 

First, we plan to use a free and open-source distribution that simplifies package management for scientific computing with the programming languages Anaconda, Python, and R. To write the necessary code for machine learning and image processing, we use Jupyter Notebook, an open-source program on the Anaconda platform.

Before classification, the Librosa library is utilized to extract features from audio files. This Python library enables the extraction of features (such as MFCC - Mel-frequency cepstrum) from audio files. The feature extraction process converts each audio file into a series of numbers, making it suitable for use with machine learning algorithms. After extracting the features of the audio files, they are transferred to an array, preparing them for the model we will use. The dataset we use in our project is then split into training and testing datasets. Furthermore, the identification of different frequencies within audio signals is performed quickly and easily using DFT (Discrete Fourier Transform) and several mathematical algorithms through a Python library.

The fundamental mechanism of this system is to detect and classify incoming external sounds and provide timely message alerts as needed. The proposed method aims to classify and issue alerts after sound detection using deep learning algorithms.

This project focuses on utilizing deep learning techniques to classify audio signals. The system extracts key features from audio files using **MFCC** and employs a **CNN** for classification. The final step involves using the trained model to predict the class of an audio file and trigger an alert when necessary.

The project is structured as follows:
1. **Data Preparation**: Preparing the dataset for analysis and extracting audio signal features using **MFCC**.
2. **Model Training**: Building and training a **CNN** on the dataset.
3. **Prediction and Alert**: Using the trained model to classify audio files and generate an alert.


## Features
- Extraction of audio features using **Mel-Frequency Cepstral Coefficients (MFCC)**.
- Construction and training of a **Convolutional Neural Network (CNN)** model for audio classification.
- Real-time prediction and alert generation based on audio classifications.


## User Interface Examples

<p align="center">
    <img width="600" alt="Screenshot_1" src="https://github.com/user-attachments/assets/80d28b20-6ea3-4a97-af2c-0865bc14bc07">
    <br>
    Figure 1 Jupyter Notebook Project Start
</p>

<br>

<p align="center">
    <img width="600" alt="Screenshot_2" src="https://github.com/user-attachments/assets/6665d73b-082f-4a0a-9c77-1e204f0dc537">
    <br>
    Figure 2 Python Libraries Used in the Project
</p>

<br>

<p align="center">
    <img width="600" alt="Screenshot_3" src="https://github.com/user-attachments/assets/10f783d9-574e-492f-b9b4-86540a41f533">
    <br>
    Figure 3 Displays mono audio data in Librosa (Jupyter)
</p>

<br>

<p align="center">
    <img width="600" alt="Screenshot_4" src="https://github.com/user-attachments/assets/b016e6c6-1376-47dc-8a81-f5414c43712f">
    <br>
    Figure 4 Extracting features from audio files (Jupyter)
</p>

<br>

<p align="center">
    <img width="600" alt="Screenshot_5" src="https://github.com/user-attachments/assets/9c4b5fd3-c986-4ba5-9553-fedeed2ea31c">
    <br>
    Figure 5 One-Hot Encoding Methods Results
</p>

<br>

<p align="center">
    <img width="600" alt="Screenshot_6" src="https://github.com/user-attachments/assets/849de368-35b2-4ee0-a0ff-58daa2114c00">
    <br>
    Figure 6 CNN Model Results
</p>

<br>

<p align="center">
    <img width="600" alt="Screenshot_7" src="https://github.com/user-attachments/assets/a53486ee-056f-4af4-9901-85078b57aedf">
    <br>
    Figure 7 CNN Trained Model Results
</p>


<br>

<p align="center">
    <img src="https://github.com/user-attachments/assets/4e8bf86d-1487-48c9-a7fb-5f3188669bb9" alt="collage">
    <br>
    Figure 8 CNN Model Results, Test Set
</p>



<br><br>


<p align="center">
    <img width="600" alt="sonn" src="https://github.com/user-attachments/assets/6431c92c-0f15-4474-b7f8-2637bbb02a24">
    <br>
    Figure 9 CNN Model, Test 2 - Not a Child Sound
</p>

<br>

## 🧰 Languages & Tools 

<div style="display: inline;">
    <img src="https://skillicons.dev/icons?i=python,anaconda" style="margin-right: 10px;" />
    <img src="https://github.com/user-attachments/assets/2f671e98-5ab2-48f6-afbf-652af748ed8e" width="50" height="60" &nbsp; &nbsp;/>
    <img src="https://github.com/devicons/devicon/blob/master/icons/numpy/numpy-original.svg" width="50" height="50" alt="NumPy"> &nbsp; &nbsp;
    <img src="https://github.com/user-attachments/assets/2296d705-347b-4bd2-adac-1db552f74d70" width="70" height="70" alt="download"> &nbsp; &nbsp;
    <img src="https://github.com/user-attachments/assets/9b8017e6-752a-4d3a-bf56-22bf1ceb0129" width="80" height="80" alt="download"> &nbsp; &nbsp;

</div>

