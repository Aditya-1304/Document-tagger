import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import timm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Device configuration (must match your training environment)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_test_loader(data_dir, batch_size=32, input_size=224, num_workers=4):
    """
    Create a DataLoader for the test set from the new dataset folder.
    This folder (data_dir) should directly contain two subfolders (e.g. "Ai" and "real").
    """
    # Define the transform (must be the same as used during training)
    data_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    
    # Since your dataset folder contains the class subfolders directly, pass data_dir directly.
    test_dataset = datasets.ImageFolder(data_dir, data_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return test_loader, len(test_dataset), test_dataset.classes

def initialize_model(num_classes):
    """
    Recreate the model architecture exactly as during training.
    (Make sure any modifications done during training are replicated here.)
    """
    model = timm.create_model('xception41', pretrained=True, num_classes=num_classes)
    # If you used full fine-tuning during training, ensure all parameters are trainable.
    for name, param in model.named_parameters():
        param.requires_grad = True
    return model.to(device)

def evaluate_model(model, dataloader, dataset_size, class_names):
    """
    Evaluate the model on the test set and print accuracy, classification report, and confusion matrix.
    """
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = np.sum(all_preds == all_labels) / dataset_size
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

def main():
    # New dataset folder (contains subfolders "Ai" and "real")
    data_dir = "AI-face-detection-Dataset"  # Make sure this is the correct path
    batch_size = 32
    input_size = 224
    # Get test DataLoader from the new dataset
    test_loader, dataset_size, class_names = get_test_loader(data_dir, batch_size, input_size, num_workers=4)
    
    # For the number of classes, you can use the length of class_names.
    num_classes = len(class_names)
    print("Detected classes:", class_names)
    
    # Reinitialize the model and load saved weights
    model = initialize_model(num_classes)
    model.load_state_dict(torch.load("best_deepfake_model.pth", map_location=device))
    model = model.to(device)
    
    # Evaluate the model
    evaluate_model(model, test_loader, dataset_size, class_names)

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()  # For Windows multiprocessing safety
    main()
