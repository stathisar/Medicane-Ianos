from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.optim as optim
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import os
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
#transformers dm
df_dm = pd.read_csv('/home/stathis/Desktop/Research/revising_Ianos/training_data_text/training_766.csv')
texts = df_dm['translated2'].values
labels = df_dm['dm'].values
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
optimizer = optim.Adam(model.parameters(), lr=5e-5)
epochs = 100
loss_stats = pd.DataFrame()
loss_stats['epoch'] = []
loss_stats['train_loss'] = []
loss_stats['val_loss'] = []
loss_stats['val_accuracy'] = []
loss_stats['accuracy'] = []

patience = 100  # Number of epochs to wait for improvement before stopping
best_val_loss = float('inf')  # Initialize best validation loss
best_val_accuracy = 0.0  # Initialize best validation accuracy
patience_counter = 0  # To track how long validation loss hasn't improved
checkpoint_path = "/home/stathis/Desktop/Research/revising_Ianos/text_and_ler_results/text/transformers_dm_best_model.pt"  # Filepath to save the best model
 

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
loss_stats.to_csv('/home/stathis/Desktop/Research/revising_Ianos/text_and_ler_results/text/transformers_stats_dm.csv')
torch.save(model.state_dict(), '/home/stathis/Desktop/Research/revising_Ianos/text_and_ler_results/text/transformers_dm_last_model.pth')







#transformers identification
df_ident = pd.read_csv('/home/stathis/Desktop/Research/revising_Ianos/training_data_text/training_766.csv')
texts = df_ident['translated2'].values
labels = df_ident['identification'].values
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
optimizer = optim.Adam(model.parameters(), lr=5e-5)
epochs = 100
loss_stats = pd.DataFrame()
loss_stats['epoch'] = []
loss_stats['train_loss'] = []
loss_stats['val_loss'] = []
loss_stats['val_accuracy'] = []
loss_stats['accuracy'] = []

patience = 100  # Number of epochs to wait for improvement before stopping
best_val_loss = float('inf')  # Initialize best validation loss
best_val_accuracy = 0.0  # Initialize best validation accuracy
patience_counter = 0  # To track how long validation loss hasn't improved
checkpoint_path = "/home/stathis/Desktop/Research/revising_Ianos/text_and_ler_results/text/transformers_ident_best_model.pt"  # Filepath to save the best model
 

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
loss_stats.to_csv('/home/stathis/Desktop/Research/revising_Ianos/text_and_ler_results/text/transformers_stats_ident.csv')
torch.save(model.state_dict(), '/home/stathis/Desktop/Research/revising_Ianos/text_and_ler_results/text/transformers_ident.pth')










#transformers consequences
df_cons = pd.read_csv('/home/stathis/Desktop/Research/revising_Ianos/training_data_text/training_766.csv')
texts = df_cons['translated2'].values
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
optimizer = optim.Adam(model.parameters(), lr=5e-5)
epochs = 100
loss_stats = pd.DataFrame()
loss_stats['epoch'] = []
loss_stats['train_loss'] = []
loss_stats['val_loss'] = []
loss_stats['val_accuracy'] = []
loss_stats['accuracy'] = []

patience = 100  # Number of epochs to wait for improvement before stopping
best_val_loss = float('inf')  # Initialize best validation loss
best_val_accuracy = 0.0  # Initialize best validation accuracy
patience_counter = 0  # To track how long validation loss hasn't improved
checkpoint_path = "/home/stathis/Desktop/Research/revising_Ianos/text_and_ler_results/text/transformers_cons_best_model.pt"  # Filepath to save the best model
 

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
loss_stats.to_csv('/home/stathis/Desktop/Research/revising_Ianos/text_and_ler_results/text/transformers_stats_consequences.csv')
torch.save(model.state_dict(), '/home/stathis/Desktop/Research/revising_Ianos/text_and_ler_results/text/transformers_consequences.pth')




#transformers weather
df_weather = pd.read_csv('/home/stathis/Desktop/Research/revising_Ianos/training_data_text/training_766.csv')
texts = df_weather['translated2'].values
labels = df_weather['weather'].values
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
optimizer = optim.Adam(model.parameters(), lr=5e-5)
epochs = 100
loss_stats = pd.DataFrame()
loss_stats['epoch'] = []
loss_stats['train_loss'] = []
loss_stats['val_loss'] = []
loss_stats['val_accuracy'] = []
loss_stats['accuracy'] = []

patience = 100  # Number of epochs to wait for improvement before stopping
best_val_loss = float('inf')  # Initialize best validation loss
best_val_accuracy = 0.0  # Initialize best validation accuracy
patience_counter = 0  # To track how long validation loss hasn't improved
checkpoint_path = "/home/stathis/Desktop/Research/revising_Ianos/text_and_ler_results/text/transformers_weather_best_model.pt"  # Filepath to save the best model
 

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
loss_stats.to_csv('/home/stathis/Desktop/Research/revising_Ianos/text_and_ler_results/text/transformers_stats_weather.csv')
torch.save(model.state_dict(), '/home/stathis/Desktop/Research/revising_Ianos/text_and_ler_results/text/transformers_weather.pth')





#transformers opinions_emotions_etc
df_ident = pd.read_csv('/home/stathis/Desktop/Research/revising_Ianos/training_data_text/training_766.csv')
texts = df_ident['translated2'].values
labels = df_ident['emotions_opinions_etc'].values
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
optimizer = optim.Adam(model.parameters(), lr=5e-5)
epochs = 100
loss_stats = pd.DataFrame()
loss_stats['epoch'] = []
loss_stats['train_loss'] = []
loss_stats['val_loss'] = []
loss_stats['val_accuracy'] = []
loss_stats['accuracy'] = []

patience = 100  # Number of epochs to wait for improvement before stopping
best_val_loss = float('inf')  # Initialize best validation loss
best_val_accuracy = 0.0  # Initialize best validation accuracy
patience_counter = 0  # To track how long validation loss hasn't improved
checkpoint_path = "/home/stathis/Desktop/Research/revising_Ianos/text_and_ler_results/text/transformers_emotions_opinions_etc_best_model.pt"  # Filepath to save the best model
 

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
loss_stats.to_csv('/home/stathis/Desktop/Research/revising_Ianos/text_and_ler_results/text/transformers_stats_emotions_opinions_etc.csv')
torch.save(model.state_dict(), '/home/stathis/Desktop/Research/revising_Ianos/text_and_ler_results/text/transformers_emotions_opinions_etc.pth')

