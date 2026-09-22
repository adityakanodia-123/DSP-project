import os
import copy
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless plot generation
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

def main():
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'dataset'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=8, shuffle=True),
        'val': DataLoader(image_datasets['val'], batch_size=8, shuffle=False)
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device} | Diagnostic Classes: {class_names}", flush=True)

    # Transfer Learning: Pretrained ResNet-18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    EPOCHS = 10
    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    print("\nStarting ResNet-18 Training Loop...", flush=True)
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1:02d}/{EPOCHS} ---", flush=True)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = (running_corrects.double() / dataset_sizes[phase]).item()

            print(f"[{phase.upper()}] Loss: {epoch_loss:.4f} | Acc: {epoch_acc*100:.2f}%", flush=True)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_weights = copy.deepcopy(model.state_dict())

    # Load optimal weights and save model
    model.load_state_dict(best_weights)
    model_save_path = 'ocular_classifier_model.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"\nOptimal model saved to '{model_save_path}' with Validation Accuracy: {best_acc*100:.2f}%", flush=True)

    # Generate Validation Metrics & Confusion Matrix
    y_true = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print("\n" + "="*50, flush=True)
    print("FINAL PERFORMANCE EVALUATION (Validation Set)", flush=True)
    print("="*50, flush=True)
    print(classification_report(y_true, y_pred, target_names=class_names), flush=True)

    # Generate and save Confusion Matrix plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Diagnosis')
    plt.ylabel('Actual Diagnosis')
    plt.title('Clinical Confusion Matrix')
    plt.tight_layout()
    plot_save_path = 'confusion_matrix.png'
    plt.savefig(plot_save_path, dpi=300)
    plt.close()
    print(f"Saved evaluation plot as '{plot_save_path}'", flush=True)

if __name__ == '__main__':
    main()
