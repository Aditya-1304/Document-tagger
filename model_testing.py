import os
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import timm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import albumentations as A
from albumentations.pytorch import ToTensorV2
from multiprocessing import freeze_support

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Set the input size (must match training)
input_size = 300


test_transform = A.Compose([
    A.Resize(input_size, input_size),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def albumentations_transform(image):
    """
    Apply Albumentations transform to a PIL image.
    This function must be defined at the top level so it can be pickled.
    """
    image_np = np.array(image)
    augmented = test_transform(image=image_np)
    return augmented['image']

def get_test_loader(data_dir, batch_size=32, num_workers=16):
    """
    Create a DataLoader for the test set using Albumentations for preprocessing.
    The folder (data_dir) should directly contain the class subfolders.
    For example, if your test folder is "Test" with subfolders "Fake" and "Real".
    """
    test_dataset = datasets.ImageFolder(data_dir, transform=albumentations_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return test_loader, len(test_dataset), test_dataset.classes

def initialize_model(num_classes):
    """
    Recreate the EfficientNetV2-B0 model using timm exactly as during training,
    using the 'tf_efficientnetv2_b0' model.
    """
    model = timm.create_model('tf_efficientnetv2_b0', pretrained=True, num_classes=num_classes)
    for name, param in model.named_parameters():
        param.requires_grad = True
    return model.to(device)

def evaluate_model_multi_tta(model, dataloader, dataset_size, class_names):
    """
    Evaluate the model using multi-crop test-time augmentation (TTA).
    For each batch, we compute predictions on several augmented versions:
      - The original image,
      - Horizontally flipped,
      - Vertically flipped, and
      - 90-degree rotated.
    The logits are averaged and the final prediction is made from the average.
    """
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            tta_logits = []

            # Original
            logits_orig = model(inputs)
            tta_logits.append(logits_orig)

            # Horizontally flipped
            inputs_hflip = torch.flip(inputs, dims=[3])
            logits_hflip = model(inputs_hflip)
            tta_logits.append(logits_hflip)

            # Vertically flipped
            inputs_vflip = torch.flip(inputs, dims=[2])
            logits_vflip = model(inputs_vflip)
            tta_logits.append(logits_vflip)

            # 90-degree rotated
            inputs_rot90 = torch.rot90(inputs, k=1, dims=[2, 3])
            logits_rot90 = model(inputs_rot90)
            tta_logits.append(logits_rot90)

            # Average the logits over the TTA augmentations
            logits_avg = sum(tta_logits) / len(tta_logits)
            _, preds = torch.max(logits_avg, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = np.sum(all_preds == all_labels) / dataset_size
    print(f"Test Accuracy (Multi-TTA): {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

def main():
    # Set your test data folder.
    # For example, if your test folder is "Test" and it contains "Fake" and "Real" subfolders.
    data_dir = "Test"  # Adjust this path as needed.
    batch_size = 32

    # Create test DataLoader
    test_loader, dataset_size, class_names = get_test_loader(data_dir, batch_size, num_workers=4)
    print("Detected classes:", class_names)
    num_classes = len(class_names)
    
    # Initialize the model and load saved weights.
    model = initialize_model(num_classes)
    model.load_state_dict(torch.load("best_deepfake_model2.pth", map_location=device), strict=False)
    model = model.to(device)
    
    # Evaluate using multi-crop TTA.
    evaluate_model_multi_tta(model, test_loader, dataset_size, class_names)

if __name__ == '__main__':
    freeze_support()  # For Windows multiprocessing safety
    main()
