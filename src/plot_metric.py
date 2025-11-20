import matplotlib.pyplot as plt
import json
import argparse
import os

def plot_metrics(metrics, save_path):
    """
    Plots the training and validation loss and accuracy using data from the metrics dictionary.
    
    Args:
        metrics (dict): Dictionary containing 'num_epochs', 'learning_rate', and metric lists.
        save_path (str): The original input file path, used to derive the PNG output name.
    """
    num_epochs = metrics['num_epochs']
    lr = metrics['learning_rate']
    train_loss = metrics['train_loss']
    val_loss = metrics['val_loss']
    train_accuracy = metrics['train_accuracy']
    val_accuracy = metrics['val_accuracy']
    
    # Create the figure and subplots
    plt.figure(figsize=(12, 5))
    
    # Ensure epochs range starts from 1 for plotting clarity
    epochs_range = range(1, num_epochs + 1)
    
    # Plot 1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss vs. Epochs (LR: {lr})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plot 2: Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracy, label='Training Accuracy')
    plt.plot(epochs_range, val_accuracy, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs. Epochs (LR: {lr})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    # Generate unique PNG file name based on the input JSON file name
    base_name = os.path.basename(save_path)
    plot_name = os.path.splitext(base_name)[0] + '.png'
    final_save_path = os.path.join(os.path.dirname(save_path), plot_name)

    plt.savefig(final_save_path)
    print(f"Plot saved to {final_save_path}")

def load_metrics(filepath):
    """
    Loads metrics from a JSON file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Error: Metrics file not found at {filepath}. Please run training.py with the corresponding parameters first.")
    
    with open(filepath, 'r') as f:
        metrics = json.load(f)
        
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot training and validation metrics for a specific run.')
    
    # Requires the user to provide the specific JSON file path
    parser.add_argument(
        '--file', 
        type=str, 
        required=True, 
        help='Path to the unique JSON file containing the per-epoch training metrics (e.g., results/metrics_e20_lr000500.json).'
    )
    args = parser.parse_args()
    
    try:
        metrics = load_metrics(args.file)
        
        # Pass the loaded metrics dictionary and the input file path for naming the output PNG
        plot_metrics(metrics, save_path=args.file)
        
    except FileNotFoundError as e:
        print(e)
    except KeyError as e:
        print(f"Error: The metrics file is missing a required key: {e}. Check if the JSON structure is correct.")