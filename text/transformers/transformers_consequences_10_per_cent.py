from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import os
# Dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Load dataset
df_cons = pd.read_csv("/media/stathis/StathisUSB/final_classification_march_9/training_data/training_792.csv")
texts = df_cons['translated'].values
labels = df_cons['consequences'].values

# Label encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Define tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Create DataLoader
train_dataset = TextDataset(X_train, y_train, tokenizer, max_length=128)
test_dataset = TextDataset(X_test, y_test, tokenizer, max_length=128)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Training parameters


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)
epochs = 100
loss_stats = pd.DataFrame()
loss_stats['epoch'] = []
loss_stats['train_loss'] = []
loss_stats['val_loss'] = []
loss_stats['val_accuracy'] = []
loss_stats['accuracy'] = []
patience = 10  # Number of epochs to wait for improvement before stopping
best_val_loss = float('inf')  # Initialize best validation loss
best_val_accuracy = 0.0  # Initialize best validation accuracy
patience_counter = 0  # To track how long validation loss hasn't improved
checkpoint_path = "/media/stathis/StathisUSB/final_classification_march_9/models/10/transformers_cons_best_model.pt"  # Filepath to save the best model
 

# Το μοντέλο εκπαιδεύεται κανονικά
for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    total_train_batches = 0
    total_accuracy = 0
    total_batches = 0
    # Training Loop
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)      
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        total_train_loss += loss.item()
        # Calculate accuracy
        train_predictions = torch.argmax(logits, dim=-1)
        train_acc = accuracy_score(labels.cpu().numpy(), train_predictions.cpu().numpy())
        total_accuracy += train_acc
        total_batches += 1
        # Backward pass
        loss.backward()
        optimizer.step()
        total_train_batches += 1
    
    # Compute average training loss and accuracy
    avg_train_loss = total_train_loss / total_train_batches
    avg_acc = total_accuracy / total_batches
    print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss}, Accuracy: {avg_acc}')
    
    # Validation Loop
    model.eval()
    total_val_loss = 0
    total_val_accuracy = 0
    total_val_batches = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            total_val_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            acc = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
            total_val_accuracy += acc
            total_val_batches += 1
    
    # Compute average validation loss and accuracy
    avg_val_loss = total_val_loss / total_val_batches
    avg_val_accuracy = total_val_accuracy / total_val_batches
    print(f'Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss}, Validation Accuracy: {avg_val_accuracy}')
    
    # Logging statistics for the epoch into the dataframe
    loss_stats.loc[epoch+1, 'epoch'] = epoch + 1
    loss_stats.loc[epoch+1, 'train_loss'] = avg_train_loss
    loss_stats.loc[epoch+1, 'val_loss'] = avg_val_loss
    loss_stats.loc[epoch+1, 'val_accuracy'] = avg_val_accuracy
    loss_stats.loc[epoch+1, 'accuracy'] = avg_acc
    
    # Check if this is the best model based on your new condition
    if abs(avg_acc - avg_val_accuracy) < 0.10 and avg_val_accuracy > best_val_accuracy:
        best_val_accuracy = avg_val_accuracy
        print(f"Validation accuracy improved and satisfies the condition (accuracy - val_accuracy < 0.10). Saving model at epoch {epoch + 1}")
        torch.save(model.state_dict(), checkpoint_path)  # Save the model if condition is met
    else:
        print(f"Model does not satisfy the condition (accuracy - val_accuracy < 0.10).")
    
    # Early stopping condition
    if patience_counter >= patience:
        print(f"Early stopping triggered after {patience_counter} epochs without improvement.")
        break




# After training is complete, you can load the best model for evaluation:
# model.load_state_dict(torch.load(checkpoint_path))

loss_stats.to_csv('/media/stathis/StathisUSB/final_classification_march_9/models/10/transformers_stats_consequences.csv')
torch.save(model.state_dict(), '/media/stathis/StathisUSB/final_classification_march_9/models/10/transformers_consequences.pth')



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Μεταφορά στη συσκευή
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Φόρτωση των αποθηκευμένων βαρών
checkpoint_path = "/media/stathis/StathisUSB/final_classification_march_9/models/10/transformers_consequences_best_model.pt"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))

# Βάζουμε το μοντέλο σε evaluation mode

#predict

# Φόρτωση του CSV αρχείου
data = pd.read_csv('/media/stathis/StathisUSB/final_classification_march_9/text_data/DataTextAllSources_March_15_Removed_duplicates.csv')
X = data['translated'].fillna('')

encoded_inputs = tokenizer(X.tolist(), padding=True, truncation=True, return_tensors="pt")
X_tensor = encoded_inputs['input_ids']  # Είσοδος για το μοντέλο
attention_mask = encoded_inputs['attention_mask']  # Προσθέτουμε την attention mask
#model=torch.load('/home/stathis/Desktop/transformer.identification.pth')
model.eval()  # Θέτουμε το μοντέλο σε inference mode

#dataloader:
dataset = TensorDataset(X_tensor, attention_mask)
dataloader=DataLoader(dataset, batch_size=128)

#with torch.no_grad():
#    outputs = model(X_tensor)
#    predictions = torch.argmax(outputs.logits, dim=-1)  # Αντιστοιχίζουμε την κλάση με την μεγαλύτερη πιθανότητα
predictions = []
with torch.no_grad():  # Disable gradient calculations for inference
    for batch in dataloader:
        input_ids = batch[0].to(device)  # Get the input_ids from the batch
        attention_mask = batch[1].to(device) if len(batch) > 1 else None
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        batch_preds = torch.argmax(outputs.logits, dim=-1)
        predictions.extend(batch_preds.cpu().numpy())  # Collect predictions

data['prediction'] = predictions

# Αποθήκευση του DataFrame με τις προβλέψεις σε νέο CSV
data.to_csv('/media/stathis/StathisUSB/final_classification_march_9/results/10/transformers_consequences.csv', index=False)
