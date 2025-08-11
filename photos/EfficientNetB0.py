from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import pandas as pd
import os

#custom class for importing data paths
class CustomImageFolder(datasets.ImageFolder):
	def __getitem__(self, index):
		# Taking the image and the label from Imagefolder
		original_tuple = super().__getitem__(index)
		# Taking image path
		path = self.imgs[index][0]
		# Returning image, label and path
		return original_tuple + (path,)
        
# Defining photo transformations
transform = transforms.Compose([
	transforms.Resize((224, 224)),  # Update size
	transforms.ToTensor(),          # Convert to tensor
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
])

# Import training pics
dataset = CustomImageFolder(root='/media/stathis/StathisUSB/final_classification_march_9/training_data/vgg/train', transform=transform)
# Estimate training and validation sizes
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

# Actual split of dataset
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoader for training and validation sets
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

#Print dataset sizes
print(f"Training samples: {len(train_loader.dataset)}")
print(f"Validation samples: {len(val_loader.dataset)}")
#==========================================================================
#TRAIN
# Import original efficientnetB0 model with pretrained weights
efficientnetb0 = models.efficientnetb0(pretrained=True)

for param in efficientnetb0.parameters():
    param.requires_grad = False
    # Αντικατάστησε την τελική ταξινόμηση
    num_ftrs = efficientnetb0.classifier[1].in_features
    efficientnetb0.classifier = nn.Sequential(
    nn.Linear(num_ftrs, 1)#,
  #  nn.Sigmoid()
    )
    
    
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
efficientnetb0=efficientnetb0.to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, efficientnetb0.parameters()), lr=0.001)

val_last = 0
array1 = {'epoch': [], 'accuracy': [], 'validation_accuracy': [], 'loss' : [], 'val_loss': []}
metrics = pd.DataFrame(data = array1)
index1 = 0
early_stopping = 0
for epoch in range(100):  # 100 epochs
	efficientnetb0.train()
	running_loss = 0.0
	correct_train = 0
	total_train = 0
	for images, labels, paths in train_loader:  # Import paths
		images, labels = images.to(device), labels.to(device).float()
		labels = labels.unsqueeze(1)  
		optimizer.zero_grad()
		outputs = efficientnetb0(images)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()
		running_loss += loss.item()
		with torch.no_grad():
			predictions = torch.sigmoid(outputs) > 0.5
			correct_train += (predictions == labels).sum().item()
			total_train += labels.size(0)
	train_accuracy = 100 * correct_train / total_train
	avg_train_loss = running_loss / len(train_loader)
	print(f'Train Accuracy: {train_accuracy:.2f}%')
	results = []
	efficientnetb0.eval()
	correct=0
	total=0
	val_loss = 0
	with torch.no_grad():
		for images, labels, paths in val_loader:  # val_loader returns among other the image paths as well
			images, labels = images.to(device), labels.to(device).float()
			outputs = efficientnetb0(images)
			predictions = torch.sigmoid(outputs) > 0.5
			total += labels.size(0)
			if len(labels.shape) == 1:
				labels = labels.unsqueeze(1)
			correct += (predictions == labels).sum().item()
			loss = criterion(outputs, labels)
			val_loss += loss.item()
		val_accuracy = 100 * correct/total
		avg_val_loss = val_loss / len(val_loader)
		print(f'Validation Accuracy: {val_accuracy:.2f}%')
	if abs(train_accuracy - val_accuracy) < 10 and val_accuracy > val_last:
		torch.save(efficientnetb0, '/media/stathis/StathisUSB/final_classification_march_9/models/efficientnetb0/efficientnetb0_least_overfitting_iter1.pth')
		print(f'Condition is True, model is saved. Validation Accuracy = {val_accuracy:}%, Train Accuracy = {train_accuracy}') 
	else: 
		print('Condition not met, model is not saved')
	if val_last < val_accuracy:
		val_last = val_accuracy
		early_stopping = 0
	else:
		early_stopping += 1
	metrics.loc[index1, 'epoch'] = epoch
	metrics.loc[index1, 'accuracy'] = train_accuracy
	metrics.loc[index1, 'validation_accuracy'] = val_accuracy
	metrics.loc[index1, 'loss'] = avg_train_loss
	metrics.loc[index1, 'val_loss'] = avg_val_loss
	index1 += 1
	if early_stopping == 100: #for stopping the loops in the event of having constant worse results in terms of overfitting. Now it is set to 100 for completing the loop.
		break



#save metrics
metrics.to_csv('/media/stathis/StathisUSB/final_classification_march_9/models/efficientnetb0/efficientnetb0_metrics_apr13.csv')
#save model
torch.save(efficientnetb0, '/media/stathis/StathisUSB/final_classification_march_9/models/efficientnetb0/efficientnetb0_finetuned_last_version__not_best_apr_13.pth')
#=========================================================
#model = torch.load('/media/stathis/StathisUSB/final_classification_march_9/models/vgg/vgg_best_finetuned.pth')
#VALIDATION
#results = []
#efficientnetb0.eval()
#correct=0
#total=0
#with torch.no_grad():
#	for images, labels, paths in val_loader:
#		images, labels = images.to(device), labels.to(device)
#		outputs = efficientnetb0(images)
#		predictions = torch.sigmoid(outputs) > 0.5
#		label = 'related_to_flood' if predictions.item() == 1 else 'not_related_to_flood'
#		total += labels.size(0)
#		labels = labels.unsqueeze(1)
#		correct += (predictions == labels).sum().item()
#	val_accuracy = 100 * correct/total
#	print(f'Validation Accuracy: {accuracy:.2f}%')


#accuracy
#accuracy = 100 * correct / total
#print(f'Validation Accuracy: {accuracy:.2f}%')

#=====================
#Predict
efficientnetb0 = torch.load('/media/stathis/StathisUSB/final_classification_march_9/models/efficientnetb0/efficientnetb0_least_overfitting_iter1.pth', weights_only=False)
datasetPredict = CustomImageFolder(root='/media/stathis/StathisUSB/final_classification_march_9/photo_data/images/photos_ins_flickr/', transform=transform)
pred_size = len(datasetPredict) 
pred_loader = DataLoader(datasetPredict, batch_size=128, shuffle=False)# Create dataloader
print(f"Number of images in the dataset: {pred_size}")


results=[]


with torch.no_grad():
    results = []  # Be assured that the results are initialized.
    for images, _, paths in pred_loader:  # Returning paths as well
        images = images.to(device)
        outputs = efficientnetb0(images)  # Outputs in a schema (batch_size, 2)
        #probabilities = torch.softmax(outputs, dim=1)  #softmax for returning probabilities
        sigmoid_outputs = torch.sigmoid(outputs)
        predictions = sigmoid_outputs > 0.5
        for img_path, pred, conf in zip(paths, predictions, sigmoid_outputs):
        	labels = 'related_to_flood' if pred.item() == 1 else 'not_related_to_flood'
        	confidence = conf.item()
        	results.append({'image_name': os.path.basename(img_path), 'predicted_label': labels, 'confidence': confidence})

# bind and export results
results_df = pd.DataFrame(results)
results_df.to_csv('/media/stathis/StathisUSB/final_classification_march_9/results/efficientnetb0_predictions.apr.13.5078_.csv', index=False)
results_df_sample = results_df.sample(n=150)
results_df_sample.to_csv('/media/stathis/StathisUSB/final_classification_march_9/results/efficientnetb0_predictions_apr_13_sample_150.csv', index=False)
