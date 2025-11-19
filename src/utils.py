import torch
import torch.nn as nn
from torchvision import transforms, models 
from torchvision.models import ResNet18_Weights

# Define the image size for resizing
image_size = (224, 224)

# Define the transformation pipeline for validation and inference
# It includes resizing, converting to Tensor, and normalization
val_transform = transforms.Compose([
    transforms.Resize(image_size),  # Resize the image to 224x224
    transforms.ToTensor(),          # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet statistics
])

# ---- Define the model (ResNet18) ----
def build_model(num_classes, pretrained=True):
    """
    Loads a pretrained ResNet18 model and modifies the final layer.
    """
    if pretrained:
        # Use the recommended 'weights' argument with the specific Enum
        # ResNet18_Weights.IMAGENET1K_V1 is equivalent to the old pretrained=True
        weights = ResNet18_Weights.IMAGENET1K_V1 
    else:
        weights = None # No pretrained weights

    # Load a pretrained ResNet18 model
    # Changed: pretrained=True to weights=weights
    model = models.resnet18(weights=weights)

    # Freeze the layers of the pre-trained model only if we are using the pretrained weights
    for param in model.parameters():
        # param.requires_grad = False logic remains the same
        param.requires_grad = False
        
    # ... (Rest of the function remains the same) ...
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return model, device

# Utility function to load a model for inference (used in test.py)
def load_checkpoint_model(path, num_classes):
    """
    Loads the model architecture and then loads the state dict from a checkpoint.
    """
    # Build the model architecture (we pass pretrained=False because we are loading weights)
    # The weights will be loaded from the checkpoint, so we don't need ImageNet weights.
    model, device = build_model(num_classes, pretrained=False)
    
    # Load the trained weights (state_dict)
    # map_location='cpu' is important if you trained on GPU but are testing on CPU
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint)
    
    # Set model to evaluation mode
    model.eval()
    
    return model, device