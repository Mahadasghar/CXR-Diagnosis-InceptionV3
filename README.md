# 🩺 AI-Powered Diagnosis: Fast & Accurate Chest X-ray Classification with InceptionV3
## 🌟 Project Overview

This project implements a Deep Learning model based on the **InceptionV3 architecture** for the diagnosis and classification of abnormalities in Chest X-ray (CXR) images.

The goal is to provide a user-friendly diagnostic tool via a **Streamlit web application** that can predict common CXR pathologies (e.g., Pneumonia, Edema, etc.) or classify an image as **Negative** for pathology.

## 🚀 Features

* **Model:** Fine-tuned InceptionV3 Convolutional Neural Network (CNN).
* **Interface:** Interactive web application built with **Streamlit** for easy image upload and real-time prediction display.
* **Large File Management:** Model weights are managed externally due to size limitations.

## ⚙️ Setup and Installation

Follow these steps to set up the project locally and install the necessary dependencies.

### 1. Clone the Repository

```bash
git clone git@github.com:Mahadasghar/CXR-Diagnosis-InceptionV3.git
cd CXR-Diagnosis-InceptionV3
```
2. Create and Activate Virtual Environment (Recommended)
```Bash

# Create environment (e.g., using conda)
conda create -n cxr_env python 
conda activate cxr_env
```
3. Install Required Libraries
Install all necessary packages using the provided requirements.txt file:

```Bash
pip install -r requirements.txt
```
💾 Model Weights Access
The model weights (model.h5) are too large for direct GitHub storage and have been uploaded to Google Drive.

Model Weights Download Link:
https://drive.google.com/file/d/1I9-pHWeQlOxXjqtaS5KXxi1pGYWE02E-/view?usp=sharing

Note: If you prefer to train the model yourself or need to inspect the training logic, the full process for creating the .h5 file is documented within the project's Jupyter Notebook (e.g., CXR_InceptionV3_Training.ipynb).

🖥️ Running the Streamlit Application
Download Weights: Ensure you have downloaded the required model.h5 file from the link above and placed it in the root directory (or the directory where your Streamlit app expects it).

Run the App: Execute the Streamlit command from your project root:

```Bash

streamlit run app.py 
# The application will open automatically in your web browser, allowing you to upload a CXR image and receive a diagnosis.
```
📝 Usage

1. Open the Streamlit application URL (usually http://localhost:8501).

2. Use the file uploader to select a Chest X-ray image (.jpg or .png).

3. The model will load and display the prediction (e.g., **Predicted Label: Pneumonia, Confidence: 0.95**) along with its confidence score.
