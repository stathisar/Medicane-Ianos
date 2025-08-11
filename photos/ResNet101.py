import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

# =============================
# Custom dataset to return (image, label, path)
# =============================
class CustomImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        path = self.imgs[index][0]
        return image, label, path

# =============================
# Transformations
# =============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),          
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])  
])

# =============================
# Dataset & Dataloaders
# =============================
dataset = CustomImageFolder(
    root='/media/stathis/StathisUSB/final_classification_march_9/training_data/vgg/train',
    transform=transform
)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

print(f"Training samples: {len(train_loader.dataset)}")
print(f"Validation samples: {len(val_loader.dataset)}")

# =============================
# Model setup (ResNet101)
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet101 = models.resnet101(pretrained=True)

# Freeze backbone
for param in resnet101.parameters():
    param.requires_grad = False

# Replace the final layer
num_ftrs = resnet101.fc.in_features
resnet101.fc = nn.Linear(num_ftrs, 1)

resnet101 = resnet101.to(device)

# Loss & optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(resnet101.fc.parameters(), lr=0.001)

# =============================
# Training
# =============================
val_last = 0
metrics = pd.DataFrame(columns=['epoch', 'accuracy', 'validation_accuracy', 'loss', 'val_loss'])
early_stopping = 0

for epoch in range(100):
    resnet101.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels, _ in train_loader:
        images, labels = images.to(device), labels.to(device).float()
        labels = labels.unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = resnet101(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        with torch.no_grad():
            preds = torch.sigmoid(outputs) > 0.5
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

    train_accuracy = 100 * correct_train / total_train
    avg_train_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}], Train Accuracy: {train_accuracy:.2f}%')

    # =============================
    # Validation
    # =============================
    resnet101.eval()
    correct_val = 0
    total_val = 0
    val_loss_sum = 0

    with torch.no_grad():
        for images, labels, _ in val_loader:
            images, labels = images.to(device), labels.to(device).float()
            if len(labels.shape) == 1:
                labels = labels.unsqueeze(1)

            outputs = resnet101(images)
            preds = torch.sigmoid(outputs) > 0.5
            correct_val += (preds == labels).sum().item()
            total_val += labels.size(0)
            val_loss_sum += criterion(outputs, labels).item()

    val_accuracy = 100 * correct_val / total_val
    avg_val_loss = val_loss_sum / len(val_loader)
    print(f'Validation Accuracy: {val_accuracy:.2f}%')

    # =============================
    # Save best model
    # =============================
    if abs(train_accuracy - val_accuracy) < 10 and val_accuracy > val_last:
        torch.save(resnet101, '/media/stathis/StathisUSB/final_classification_march_9/models/resnet101/resnet101_best_finetuned_iter1.pth')
        print(f'Condition is True, model saved. Validation Accuracy = {val_accuracy:.2f}%')
    else:
        print('Condition not met, model not saved.')

    # Early stopping
    if val_accuracy > val_last:
        val_last = val_accuracy
        early_stopping = 0
    else:
        early_stopping += 1

    metrics.loc[len(metrics)] = [epoch, train_accuracy, val_accuracy, avg_train_loss, avg_val_loss]
    if early_stopping == 100: #for stopping the loop in case there are constant worse results. Now it is set to 100 for completing the loop.
        break

# Save metrics & last model
metrics.to_csv('/media/stathis/StathisUSB/final_classification_march_9/models/resnet101/resnet101_metrics_august2025.csv', index=False)
torch.save(resnet101, '/media/stathis/StathisUSB/final_classification_march_9/models/resnet101/resnet101_finetuned_last_version_august2025.pth')

# =============================
# Prediction
# =============================
resnet101 = torch.load('/media/stathis/StathisUSB/final_classification_march_9/models/resnet101/resnet101_best_finetuned_iter1.pth', weights_only=False)

datasetPredict = CustomImageFolder(
    root='/media/stathis/StathisUSB/final_classification_march_9/photo_data/images/photos_ins_flickr/',
    transform=transform
)

pred_loader = DataLoader(datasetPredict, batch_size=128, shuffle=False)
print(f"Number of images in the dataset: {len(datasetPredict)}")

results = []
with torch.no_grad():
    for images, _, paths in pred_loader:
        images = images.to(device)
        outputs = resnet101(images)
        sigmoid_outputs = torch.sigmoid(outputs)
        preds = sigmoid_outputs > 0.5

        for img_path, pred, conf in zip(paths, preds, sigmoid_outputs):
            label = 'related_to_flood' if pred.item() == 1 else 'not_related_to_flood'
            results.append({
                'image_name': os.path.basename(img_path),
                'predicted_label': label,
                'confidence': conf.item()
            })

results_df = pd.DataFrame(results)
results_df.to_csv('/media/stathis/StathisUSB/final_classification_march_9/results/resnet101_predictions_august2025.csv', index=False)
results_df.sample(n=150).to_csv('/media/stathis/StathisUSB/final_classification_march_9/results/resnet101_predictions_august2025_sample_150.csv', index=False)

