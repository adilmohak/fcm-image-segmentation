# Introduction

`EnFCM` (Enhanced Fuzzy C-Means), `FCM_S1`, and `FCM_S2` are all algorithms used for **image segmentation**. 
They are commonly used for medical image analysis, including cancer detection from MRI imaging.

To use these algorithms for cancer detection from MRI imaging using Python, you could use a variety of **machine learning** and **image processing** libraries such as `scikit-image`, `OpenCV`, or `SimpleITK`. These libraries provide pre-built functions and methods for image segmentation, and you can use them to implement `EnFCM`, `FCM_S1`, or `FCM_S2`.

Here's a high-level overview of how you could use Python to implement image segmentation with these algorithms:

1. Import the necessary libraries for image processing and machine learning.
2. Load the MRI image dataset that you want to segment.
3. Preprocess the image data, which may include tasks such as **smoothing**, **filtering**, or **normalization**.
4. Use the selected algorithm (`EnFCM`, `FCM_S1`, or `FCM_S2`) to segment the image into distinct regions.
5. Postprocess the segmentation results to remove any noise or artifacts and to refine the boundaries between regions.
6. Visualize the segmented image to evaluate the quality of the segmentation.

## Sample images used
Here is the link where I used for sample MRI images:

- Brain MRI Images for Brain Tumor Detection: https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection

## Installation

- Clone the repo: `git clone https://github.com/adilmohak/fcm-image-segmentation.git` or download the zip file manually, and run the following commands inside the project root directory
- Create python virtual environment and activate it: `python -m venv venv`
- Install requirements: `pip install -r requirements.txt`
- Now you can run each script one-by-one: For example, `python fcm_s1_segmenter.py`

## I hope you learn something from this project, and if you do, give that star a click :)

# Thanks!
