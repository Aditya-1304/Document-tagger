# main.py

import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Device configuration: use CUDA if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def get_data_loaders(data_dir, batch_size=32, input_size=224):
    """
    Create training, validation and test dataloaders using torchvision.datasets.ImageFolder.
    Assumes folder structure: 
      Dataset/Train/{Real,Fake}, Dataset/Validation/{Real,Fake}, Dataset/Test/{Real,Fake}
    """
    # Define transforms (data augmentation for training, normalization for all)
    data_transforms = {
        'Train': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],  # imagenet mean/std
                                 [0.229, 0.224, 0.225])
        ]),
        'Validation': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'Test': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['Train', 'Validation', 'Test']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x == 'Train'), num_workers=4)
                   for x in ['Train', 'Validation', 'Test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['Train', 'Validation', 'Test']}
    class_names = image_datasets['Train'].classes  # e.g., ['Fake', 'Real']

    print("Classes:", class_names)
    print("Dataset sizes:", dataset_sizes)

    return dataloaders, dataset_sizes, class_names


def initialize_model(num_classes, feature_extract=True, fine_tune_from='layer4'):
    """
    Load a pretrained ResNet18 model, freeze its parameters if feature_extract is True,
    then unfreeze the parameters for layers that contain the string `fine_tune_from`.
    Finally, replace the fully connected layer.
    """
    model = models.resnet18(pretrained=True)
    
    if feature_extract:
        # Freeze all parameters first.
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze parameters in the specified block (e.g., 'layer4').
        for name, param in model.named_parameters():
            if fine_tune_from in name:
                param.requires_grad = True
                print(f"Unfreezing parameter: {name}")

    # Replace the final fully connected layer.
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model.to(device)


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=10, scheduler=None):
    """
    Train the model and validate on the validation set.
    Returns the best model (deep copy).
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    since = time.time()
    for epoch in range(num_epochs):
        print('-' * 40)
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Each epoch has a training and validation phase.
        for phase in ['Train', 'Validation']:
            if phase == 'Train':
                model.train()  # Set model to training mode.
            else:
                model.eval()   # Set model to evaluation mode.

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Forward pass; track gradients only if in train phase.
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # Backward + optimize in train phase.
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Deep copy the model if validation accuracy improved.
            if phase == 'Validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        # Step the scheduler if provided.
        if scheduler is not None:
            scheduler.step()

        print()

    time_elapsed = time.time() - since
    print('-' * 40)
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f}")

    # Load best model weights.
    model.load_state_dict(best_model_wts)
    return model


def evaluate_model(model, dataloader, class_names):
    """
    Evaluate the model on the test set, printing out accuracy, classification report and confusion matrix.
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
    acc = np.sum(all_preds == all_labels) / len(all_labels)
    print("Test Accuracy: {:.4f}".format(acc))
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))


def main():
    # Set parameters.
    data_dir = "Dataset"  # path to the dataset folder
    num_epochs = 10
    batch_size = 32
    input_size = 224
    num_classes = 2  # e.g., "Fake" and "Real"

    # Get dataloaders.
    dataloaders, dataset_sizes, class_names = get_data_loaders(data_dir, batch_size, input_size)

    # Initialize model.
    # Here, we freeze most layers but unfreeze those in 'layer4' so the network can adapt more.
    model = initialize_model(num_classes, feature_extract=True, fine_tune_from='layer4')

    # Define loss function.
    criterion = nn.CrossEntropyLoss()

    # Only update parameters that require gradients.
    params_to_update = [param for param in model.parameters() if param.requires_grad]
    optimizer = optim.Adam(params_to_update, lr=0.001)

    # Define a learning rate scheduler.
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Train the model.
    model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs, scheduler)

    # Save the best model.
    torch.save(model.state_dict(), "best_deepfake_model.pth")
    print("Saved best model weights to best_deepfake_model.pth")

    # Evaluate on the test set.
    print("\nEvaluating on Test Set:")
    evaluate_model(model, dataloaders['Test'], class_names)


if __name__ == '__main__':
    main()
