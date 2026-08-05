import cv2
import os
import numpy as np
from skimage import exposure

# Function to extract features from an image
def extract_features(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 128))

    # Extract HOG features
    hog = cv2.HOGDescriptor()
    features = hog.compute(gray)   

    # Normalize the histogram
    features = exposure.rescale_intensity(features, out_range=(0, 255))

    return features.flatten()


# Function to compare an image with a dataset
def compare_with_dataset(image_path, dataset_path):
    # Extract features from the input image
    query_features = extract_features(image_path)

    # Iterate through images in the dataset
    min_distance = float('inf')
    classification = None

    for class_label in os.listdir(dataset_path):
        class_dir = os.path.join(dataset_path, class_label)

        for filename in os.listdir(class_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                dataset_image_path = os.path.join(class_dir, filename)

                # Extract features from the dataset image
                dataset_features = extract_features(dataset_image_path)

                # Calculate distance between features
                distance = np.linalg.norm(query_features - dataset_features)

                # Update minimum distance and classification
                if distance < min_distance:
                    min_distance = distance
                    classification = class_label

    return classification


# Provide paths to the input image and dataset directory
input_image_path = "/Users/anushri/Documents/scabscanner/Wound_dataset/Laceration/laceration (8).jpg"
dataset_path = "/Users/anushri/Documents/scabscanner/Wound_dataset"

# Call the function to compare with the dataset
classification = compare_with_dataset(input_image_path, dataset_path)
print("The input image belongs to class:", classification)