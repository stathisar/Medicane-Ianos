#=====================
#Predict
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import pandas as pd
import os
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


efficientnetb0 = torch.load('/home/stathis/Desktop/Research/revising_Ianos/data_scripts_models/efficientnetb0/efficientnetb0_less_overfitting.pth', weights_only=False)
datasetPredict = CustomImageFolder(root='/home/stathis/Desktop/Research/revising_Ianos/filtered/flickr/', transform=transform)
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
results_df.to_csv('/home/stathis/Desktop/Research/revising_Ianos/task-frames-predict-with-retrained-models/efficientnetb0_oct26_flickr.csv', index=False)
results_df_sample = results_df.sample(n=150)
results_df_sample.to_csv('/home/stathis/Desktop/Research/revising_Ianos/task-frames-predict-with-retrained-models/efficientnetb0_oct26_flickr_sample_150.csv', index=False)
