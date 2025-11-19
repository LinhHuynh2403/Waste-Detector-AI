# this script tests the trained model on a single image and prints the predicted class and confidence 
# it uses new load_class_name function to read the class names from class_names.json created by training.py 

import torch
import json
from PIL import Image
import numpy as np

# Import components from utils.py
from utils import load_checkpoint_model, val_transform

# Configuration
MODEL_PATH = 'trash_sorting_finetuned_model.pth'
CLASS_NAMES_PATH = 'class_names.json'
NUM_CLASSES = 6 # Set the number of classes

# Define a list of images to test
# NOTE: Replace these with paths to your actual validation images
TEST_IMAGE_PATHS = [ 
    'data/Garbage classification/validation/plastic/plastic100.jpg',
    'data/Garbage classification/validation/cardboard/cardboard10.jpg',
    'data/Garbage classification/validation/glass/glass50.jpg',
    # Add more paths here to test other categories
]

def load_class_names(path):
    # ... (same function as before) ...
    try:
        with open(path, 'r') as f:
            class_names = json.load(f)
        return class_names
    except FileNotFoundError:
        print(f"Error: Class names file not found at {path}")
        return None

def predict_image_class(image_path, model, device, class_names):
    """
    Loads an image, preprocesses it, and makes a prediction.
    """
    print(f"\n--- Testing Image: {image_path} ---")
    try:
        # 1. Load the image
        image = Image.open(image_path).convert('RGB')

        # 2. Apply transformations
        image_tensor = val_transform(image)
        image_tensor = image_tensor.unsqueeze(0) # Add batch dimension
        image_tensor = image_tensor.to(device)

        # 3. Perform inference
        with torch.no_grad():
            output = model(image_tensor)
        
        # 4. Get the predicted class
        probabilities = torch.nn.functional.softmax(output[0], dim=0) 
        top_p, top_class_index = probabilities.topk(1, dim=0)

        # 5. Map the index to the class name
        predicted_class = class_names[top_class_index.item()]
        
        # 6. Print the results
        print(f"Predicted Class: **{predicted_class}**")
        print(f"Confidence: **{top_p.item():.4f}**")
        
    except Exception as e:
        print(f"An error occurred during prediction for {image_path}: {e}")
        # Hint to check file path or model loading
        if 'No such file or directory' in str(e):
            print("Please double-check that the image path is correct.")


if __name__ == "__main__":
    # 1. Load the class names
    class_names = load_class_names(CLASS_NAMES_PATH)
    if not class_names:
        print("Cannot proceed without class names.")
    else:
        num_classes = len(class_names)
        
        # 2. Load the trained model
        print(f"Loading model from {MODEL_PATH}...")
        try:
            model, device = load_checkpoint_model(MODEL_PATH, num_classes)
            print("Model loaded successfully.")
            
            # 3. Iterate and test all images in the list
            for path in TEST_IMAGE_PATHS:
                predict_image_class(path, model, device, class_names)

        except FileNotFoundError:
            print(f"Error: Model file not found at {MODEL_PATH}. Have you run 'training.py' yet?")
        except RuntimeError as e:
            print(f"Runtime Error during model loading/inference: {e}")