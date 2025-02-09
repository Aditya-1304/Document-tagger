import os
import time
import copy
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets
import timm  # For model creation
import torch.amp  # For AMP
from albumentations import (Compose, ImageCompression, GaussNoise, GaussianBlur,
                            HorizontalFlip, OneOf, PadIfNeeded, RandomBrightnessContrast,
                            FancyPCA, HueSaturationValue, ToGray, ShiftScaleRotate, CoarseDropout,
                            Resize, Normalize)
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import classification_report, confusion_matrix
# Enable cuDNN benchmark for fixed input sizes.
torch.backends.cudnn.benchmark = True
# Device configuration.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
# ---------------------------
# Albumentations Transforms
# ---------------------------
def create_train_transforms(size=300):
    """Create heavy augmentation transforms for training."""
    return Compose([
        # Remove quality_lower/quality_upper if not supported
        ImageCompression(p=0.5),
        GaussNoise(p=0.1),
        GaussianBlur(blur_limit=3, p=0.05),
        HorizontalFlip(p=0.5),
        OneOf([
            # Use Albumentations Resize (with explicit keyword arguments)
            Resize(height=size, width=size, interpolation=cv2.INTER_CUBIC, p=1.0),
            Resize(height=size, width=size, interpolation=cv2.INTER_LINEAR, p=1.0),
            Resize(height=size, width=size, interpolation=cv2.INTER_AREA, p=1.0)
        ], p=1.0),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),
        OneOf([
            RandomBrightnessContrast(p=0.7),
            FancyPCA(p=0.7),
            HueSaturationValue(p=0.7)
        ], p=0.7),
        ToGray(p=0.2),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10,
                         border_mode=cv2.BORDER_CONSTANT, p=0.5),
        CoarseDropout(max_holes=8, max_height=int(0.1 * size), max_width=int(0.1 * size), p=0.5),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
def get_val_transform(size=224):
    """Validation/test transform using Albumentations."""
    return Compose([
        Resize(height=size, width=size, interpolation=cv2.INTER_LINEAR),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
# ---------------------------
# Dataset and DataLoader Setup
# ---------------------------
# We'll define a simple wrapper so that Albumentations transforms can be used with ImageFolder.
class AlbumentationsTransform:
    def __init__(self, transform):
        self.transform = transform
    def __call__(self, image):
        # Convert PIL image to numpy array.
        image = np.array(image)
        augmented = self.transform(image=image)
        return augmented["image"]
def get_data_loaders(data_dir, batch_size=32, input_size=300, num_workers=16):
    """
    Create training, validation, and test DataLoaders.
    Expected folder structure:
       data_dir/
           Train/
               class1/
               class2/
           Validation/
               class1/
               class2/
           Test/
               class1/
               class2/
    """
    train_transform = AlbumentationsTransform(create_train_transforms(size=input_size))
    val_transform = AlbumentationsTransform(get_val_transform(size=input_size))
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x),
                                transform=(train_transform if x=='Train' else val_transform))
        for x in ['Train', 'Validation', 'Test']
    }
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x=='Train'),
                      num_workers=num_workers, pin_memory=True)
        for x in ['Train', 'Validation', 'Test']
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['Train', 'Validation', 'Test']}
    class_names = image_datasets['Train'].classes
    print("Classes:", class_names)
    print("Dataset sizes:", dataset_sizes)
    return dataloaders, dataset_sizes, class_names
# ---------------------------
# Model Initialization
# ---------------------------
def initialize_model(num_classes, fine_tune_strategy='full'):
    """
    Initialize an EfficientNetV2-B0 model using timm.
    Model name: 'tf_efficientnetv2_b0'
    """
    model = timm.create_model('tf_efficientnetv2_b0', pretrained=True, num_classes=num_classes)
    if fine_tune_strategy == 'classifier':
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
            else:
                print(f"Unfreezing parameter: {name}")
    elif fine_tune_strategy == 'full':
        for name, param in model.named_parameters():
            param.requires_grad = True
        print("All parameters are trainable.")
    return model.to(device)
# ---------------------------
# Training Function
# ---------------------------
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=10, scheduler=None, accumulation_steps=4):
    """
    Train the model using gradient accumulation and AMP.
    For a 4GB VRAM laptop, using a real batch size of 24 with accumulation_steps=4
    gives an effective batch size of 96.
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    scaler = torch.amp.GradScaler()  # Instantiate GradScaler.
    since = time.time()
    for epoch in range(num_epochs):
        print('-'*40)
        print(f"Epoch {epoch+1}/{num_epochs}")
        for phase in ['Train', 'Validation']:
            if phase=='Train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            optimizer.zero_grad()
            for step, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                with torch.set_grad_enabled(phase=='Train'):
                    with torch.amp.autocast(device_type='cuda'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss = loss / accumulation_steps  # Normalize loss for accumulation
                    _, preds = torch.max(outputs, 1)
                    if phase=='Train':
                        scaler.scale(loss).backward()
                        if (step+1) % accumulation_steps == 0:
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                running_loss += loss.item() * inputs.size(0) * accumulation_steps
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            if phase=='Validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        if scheduler is not None:
            scheduler.step()
        print()
    time_elapsed = time.time()-since
    print('-'*40)
    print(f"Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    return model
# ---------------------------
# Evaluation Function
# ---------------------------
def evaluate_model(model, dataloader, class_names):
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
    acc = np.sum(all_preds==all_labels)/len(all_labels)
    print(f"Test Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
# ---------------------------
# Main Function
# ---------------------------
def main():
    # Set parameters.
    data_dir = "Dataset"  # Replace with your dataset folder path
    num_epochs = 10
    batch_size = 32   # Real batch size for 4GB VRAM
    input_size = 300  # Using 300 to match augmentation settings
    num_classes = 2   # e.g., "fake" and "real"
    dataloaders, dataset_sizes, class_names = get_data_loaders(data_dir, batch_size, input_size, num_workers=16)
    model = initialize_model(num_classes, fine_tune_strategy='full')
    criterion = nn.CrossEntropyLoss()
    # Differential learning rates.
    classifier_params = list(model.get_classifier().parameters())
    classifier_ids = {id(p) for p in classifier_params}
    backbone_params = [param for param in model.parameters() if param.requires_grad and id(param) not in classifier_ids]
    optimizer = optim.Adam([
        {'params': backbone_params, 'lr': 5e-5},
        {'params': classifier_params, 'lr': 1e-3}
    ], weight_decay=1e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    # Use gradient accumulation with accumulation_steps=4.
    model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer,
                        num_epochs=num_epochs, scheduler=scheduler, accumulation_steps=4)
    torch.save(model.state_dict(), "best_deepfake_model2.pth")
    print("Saved best model weights to best_deepfake_model2.pth")
    print("\nEvaluating on Test Set:")
    evaluate_model(model, dataloaders['Test'], class_names)
if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
    
    
    