import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json
import argparse
import csv 
import os 
from utils import build_model, image_size, val_transform

# Define the image size for resizing and batch size
batch_size = 32

# Define transformations for training data (with augmentation)
train_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomRotation(40),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Paths to the training and validation data directories
train_dir = 'data/Garbage classification/training'
val_dir = 'data/Garbage classification/validation'

# Define the function to load datasets (remains the same)
def load_data():
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, train_dataset.classes

# ---- Define the Loss Function and Optimizer ----
def define_optimizer(model, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr) 
    return criterion, optimizer

# ---- Training the model (logic remains the same) ----
def train_model(num_epochs, train_loader, val_loader, model, criterion, optimizer, device):
    train_loss = []
    val_loss = []
    train_accuracy = []
    val_accuracy = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_accuracy = correct_train / total_train
        train_loss.append(epoch_train_loss)
        train_accuracy.append(epoch_train_accuracy)

        model.eval()
        correct_val = 0
        total_val = 0
        running_val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = running_val_loss / len(val_loader)
        epoch_val_accuracy = correct_val / total_val
        val_loss.append(epoch_val_loss)
        val_accuracy.append(epoch_val_accuracy)

        print(f'Epoch [{epoch+1}/{num_epochs}], '
            f'Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_accuracy:.4f}, '
            f'Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {epoch_val_accuracy:.4f}')

    return train_loss, val_loss, train_accuracy, val_accuracy

# Function to save metrics as JSON 
def save_metrics_json(train_loss, val_loss, train_accuracy, val_accuracy, num_epochs, lr, filepath):
    metrics = {
        'num_epochs': num_epochs,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy,
        'learning_rate': lr
    }
    with open(filepath, 'w') as f:
        json.dump(metrics, f)
    print(f"Training metrics saved to {filepath}")

# Function to save metrics as CSV 
def save_metrics_csv(run_id, num_epochs, lr, final_train_loss, final_val_loss, final_train_accuracy, final_val_accuracy, filepath):
    
    # Header Order: ['Run_ID', 'Epoch', 'Learning_Rate', 'Train_Loss', 'Val_Loss', 'Train_Accuracy', 'Val_Accuracy']
    header = ['Run_ID', 'Epoch', 'Learning_Rate', 'Train_Loss', 'Val_Loss', 'Train_Accuracy', 'Val_Accuracy']
    data_row = [
        run_id, # Correctly uses run_id from function argument
        num_epochs,
        lr,
        f"{final_train_loss:.4f}",
        f"{final_val_loss:.4f}",
        f"{final_train_accuracy * 100:.2f}",
        f"{final_val_accuracy * 100:.2f}"
    ]
    
    # Check if file exists to decide whether to write header
    write_header = not os.path.exists(filepath)
    
    # Open file in append mode ('a')
    with open(filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        
        if write_header:
            writer.writerow(header) # Write header only if file is new
        
        writer.writerow(data_row)
        
    print(f"Final summary metrics appended to CSV: {filepath}")

# Function to save class names 
def save_class_names(classes, filepath):
    with open(filepath, 'w') as f:
        json.dump(classes, f)
    print(f"Class names saved to {filepath}")

# ---- Main function (Modified) ----
if __name__ == "__main__":
    RESULTS_FOLDER = 'results'
    SUMMARY_CSV_PATH = os.path.join(RESULTS_FOLDER, 'training_metrics.csv')
    
    # Clear the summary file at the start of a fresh sweep
    if os.path.exists(SUMMARY_CSV_PATH):
        os.remove(SUMMARY_CSV_PATH)
        print(f"Removed existing summary file: {SUMMARY_CSV_PATH} to start fresh sweep.")
    
    parser = argparse.ArgumentParser(description='Train a garbage classification model.')
    parser.add_argument('--epochs', type=int, nargs='+', default=[10], help='List of epoch counts to test. (Default: 10)')
    parser.add_argument('--lr', type=float, nargs='+', default=[0.001], help='List of learning rates to test. (Default: 0.001)')
    args = parser.parse_args()

    # Load the data once
    train_loader, val_loader, classes = load_data()
    num_classes = len(classes)
    
    # Save class names once
    class_names_path = os.path.join(RESULTS_FOLDER, 'class_names.json')
    save_class_names(classes, class_names_path)

    run_id = 0
    
    # NESTED LOOPS for Grid Search
    for num_epochs in args.epochs:
        for lr in args.lr:
            run_id += 1
            
            print("\n" + "="*50)
            print(f"STARTING RUN {run_id}: Epochs={num_epochs}, LR={lr}")
            print("="*50)

            # --- Unique File Naming for Model and Plotting JSON ---
            suffix = f"e{num_epochs}_lr{lr:.6f}".replace('.', '')
            model_path = os.path.join(RESULTS_FOLDER, f'model_{suffix}.pth')
            metrics_json_path = os.path.join(RESULTS_FOLDER, f'metrics_{suffix}.json') # For plotting script
            
            # 1. Build the model (Re-initialize weights for each run)
            model, device = build_model(num_classes)

            # 2. Define the loss function and optimizer
            criterion, optimizer = define_optimizer(model, lr)
            
            # 3. Train the model
            train_loss, val_loss, train_accuracy, val_accuracy = train_model(
                num_epochs, train_loader, val_loader, model, criterion, optimizer, device
            )

            # 4. Extract Final Metrics
            final_train_loss = train_loss[-1]
            final_val_loss = val_loss[-1]
            final_train_accuracy = train_accuracy[-1]
            final_val_accuracy = val_accuracy[-1]

            # 5. Save per-epoch JSON (for plotting)
            save_metrics_json(train_loss, val_loss, train_accuracy, val_accuracy, num_epochs, lr, metrics_json_path)

            # 6. Log final metrics to the single summary CSV (for comparison)
            save_metrics_csv(
                num_epochs, 
                lr, 
                final_train_loss, 
                final_val_loss, 
                final_train_accuracy, 
                final_val_accuracy, 
                SUMMARY_CSV_PATH
            )
            
            # 7. Save the unique model file
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
            
    print("\nHyperparameter sweep complete!")
    print(f"All final results are summarized in: {SUMMARY_CSV_PATH}")