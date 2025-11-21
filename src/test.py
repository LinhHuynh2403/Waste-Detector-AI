import torch
import json
import csv
from PIL import Image
import numpy as np
import argparse
import os

# Import components from utils.py
from utils import load_checkpoint_model, val_transform

# Configuration
CLASS_NAMES_PATH = 'results/class_names.json' 
SUMMARY_CSV_PATH = 'results/training_metrics.csv'
# BASE PATH: User will provide the path relative to this directory
BASE_TEST_DIR = 'data/Garbage classification/validation' 

def find_best_model_params(run_id, csv_path):
    """
    Reads the summary CSV to find the Epochs and LR for the given Run ID.
    """
    try:
        with open(csv_path, mode='r') as file:
            reader = csv.DictReader(file) 
            for row in reader:
                if row.get('Run_ID') == str(run_id): 
                    return int(row['Epoch']), float(row['Learning_Rate'])
            
            raise ValueError(f"Run ID {run_id} not found in the summary CSV.")
            
    except FileNotFoundError:
        print(f"Error: Summary CSV file not found at {csv_path}. Run training.py first.")
        return None, None
    except KeyError as e:
        print(f"Error: Summary CSV is missing required column: {e}. Check if Run_ID column exists.")
        return None, None


def load_class_names(path):
    """Loads the list of class names from a JSON file."""
    try:
        with open(path, 'r') as f:
            class_names = json.load(f)
        return class_names
    except FileNotFoundError:
        print(f"Error: Class names file not found at {path}")
        return None

def predict_image_class(image_path, model, device, class_names):
    """
    Loads an image, preprocesses it, and makes a prediction, printing the trash type.
    """
    print(f"\n--- Testing Image: {image_path} ---")
    try:
        # 1. Load the image
        image = Image.open(image_path).convert('RGB')

        # 2. Apply transformations
        image_tensor = val_transform(image)
        image_tensor = image_tensor.unsqueeze(0) 
        image_tensor = image_tensor.to(device)

        # 3. Perform inference
        model.eval() 
        with torch.no_grad():
            output = model(image_tensor)
        
        # 4. Get the predicted class
        probabilities = torch.nn.functional.softmax(output[0], dim=0) 
        top_p, top_class_index = probabilities.topk(1, dim=0)

        # 5. Map the index to the class name
        predicted_class = class_names[top_class_index.item()]
        
        # 6. Print the results (the predicted trash type and confidence)
        print(f"Predicted Trash Type: **{predicted_class}**")
        print(f"Confidence: **{top_p.item():.4f}**")
        
    except FileNotFoundError:
        print(f"Error: Test image file not found at {image_path}. Please check the relative path.")
    except Exception as e:
        print(f"An error occurred during prediction for {image_path}: {e}")


if __name__ == "__main__":
    
    # 1. Setup Argument Parser
    parser = argparse.ArgumentParser(description='Test a trained model on one or more input images.')
    
    parser.add_argument(
        '--run_id', 
        type=int, 
        required=True,
        help='The Run ID (e.g., 1, 2, 3...) from the training_metrics.csv file that corresponds to the best model.'
    )
    
    # ARGUMENT MODIFIED TO EXPECT RELATIVE PATHS
    parser.add_argument(
        '--image', 
        type=str, 
        nargs='+', 
        required=True,
        help='One or more image paths RELATIVE to the validation folder (e.g., plastic/plastic100.jpg).'
    )
    args = parser.parse_args()
    
    # 2. Find the model parameters from the CSV
    num_epochs, lr = find_best_model_params(args.run_id, SUMMARY_CSV_PATH)
    
    if not num_epochs:
        exit()
    
    # 3. Dynamically construct the model path
    suffix = f"e{num_epochs}_lr{lr:.6f}".replace('.', '')
    model_path = os.path.join('results', f'model_{suffix}.pth')
    
    # 4. Load the class names
    class_names = load_class_names(CLASS_NAMES_PATH)
    if not class_names:
        print("Cannot proceed without class names. Check class_names.json path.")
        exit()
    
    num_classes = len(class_names)
    
    # 5. Load the trained model
    print(f"Loading model from {model_path}...")
    try:
        model, device = load_checkpoint_model(model_path, num_classes)
        print("Model loaded successfully.")
        
        # 6. Iterate and predict for all user-specified images
        print(f"\nUsing Base Test Directory: {BASE_TEST_DIR}") # Inform the user of the base path
        for image_rel_path in args.image: 
            # Construct the full, absolute image path
            full_image_path = os.path.join(BASE_TEST_DIR, image_rel_path) 
            predict_image_class(full_image_path, model, device, class_names)

    except FileNotFoundError as e:
        print(f"Error during file loading: {e}")
    except RuntimeError as e:
        print(f"Runtime Error during model loading/inference: {e}")