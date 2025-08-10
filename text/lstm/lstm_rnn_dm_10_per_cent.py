import numpy as np
import pandas as pd
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, Callback
# Load and preprocess your data
df = pd.read_csv('/media/stathis/StathisUSB/final_classification_march_9/training_data/training_792.csv')

# Preprocessing function
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove multiple spaces
    text = text.strip()
    return text

df['processed'] = df['translated'].apply(preprocess_text)

# Tokenize and pad the text
MAX_NB_WORDS = 5000
MAX_SEQUENCE_LENGTH = 50
EMBEDDING_DIM = 100

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True)
tokenizer.fit_on_texts(df['processed'].values)
X = tokenizer.texts_to_sequences(df['processed'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

# Prepare labels
df['label'] = df['dm']
Y = to_categorical(df['label'])

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))  # Binary classification

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Create the custom callback for value_limit
# Custom callback to monitor and prevent saving if accuracy difference is too large
class CustomCheckpoint(Callback):
    def __init__(self, filepath, monitor='val_accuracy', threshold=0.10, verbose=1):
        super(CustomCheckpoint, self).__init__()
        self.filepath = filepath
        self.monitor = monitor  # What to monitor (usually 'val_accuracy')
        self.threshold = threshold  # Max allowed difference between acc and val_acc
        self.verbose = verbose
        self.best_val_acc = -np.Inf  # Initialize to track best validation accuracy
    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get('val_accuracy')
        acc = logs.get('accuracy')
        val_loss = logs.get('val_loss')
        # Check if difference between accuracy and val_accuracy exceeds the threshold
        if (acc - val_acc) <= self.threshold:  # Only save if the difference is small
            if val_acc > self.best_val_acc:
                # Update the best validation accuracy
                self.best_val_acc = val_acc
                # Save the model
                if self.verbose > 0:
                    print(f"\nEpoch {epoch + 1}: val_accuracy improved to {val_acc:.4f}, saving model to {self.filepath}")
                self.model.save(self.filepath)
            else:
                if self.verbose > 0:
                    print(f"\nEpoch {epoch + 1}: val_accuracy did not improve from {self.best_val_acc:.4f}")
        else:
            # If the difference between acc and val_acc is too large, don't save
            if self.verbose > 0:
                print(f"\nEpoch {epoch + 1}: Difference between acc ({acc:.4f}) and val_acc ({val_acc:.4f}) is too large. Model not saved.")

# Use CustomCheckpoint instead of ModelCheckpoint
custom_checkpoint = CustomCheckpoint(
    filepath='/media/stathis/StathisUSB/final_classification_march_9/models/10/lstm_dm_best_model.h5',
    monitor='val_accuracy',
    threshold=0.10,  # Threshold for accuracy difference
    verbose=1
)

# Callbacks
csv_logger = CSVLogger('/media/stathis/StathisUSB/final_classification_march_9/models/10/dm_model_log.csv', append=False)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=30, mode='max', verbose=1, restore_best_weights=True)



# Callbacks
#csv_logger = CSVLogger('/media/stathis/KINGSTON/ML-Water/filtered_data/final_classification_march_9/models/10/consequences_model_log.csv', append=False)
#checkpoint = ModelCheckpoint('/media/stathis/KINGSTON/ML-Water/filtered_data/final_classification_march_9/models/10/lstm_conseq_best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
#early_stopping = EarlyStopping(monitor='val_accuracy', patience=30, mode='max', verbose=1, restore_best_weights=True)
history = model.fit(
    X_train, Y_train,
    epochs=100,
    batch_size=1024,
    validation_data=(X_test, Y_test),
    verbose=2,
    callbacks=[csv_logger, custom_checkpoint]
)
# Evaluate the model
score, acc = model.evaluate(X_test, Y_test, verbose=2)
print(f'Test score: {score}')
print(f'Test accuracy: {acc}')

# Save predictions
data_predict = pd.read_csv('/media/stathis/StathisUSB/final_classification_march_9/text_data/DataTextAllSources_March_15_Removed_duplicates.csv')
data_predict['processed'] = data_predict['translated'].apply(preprocess_text)
X_new = tokenizer.texts_to_sequences(data_predict['processed'])
X_new = pad_sequences(X_new, maxlen=MAX_SEQUENCE_LENGTH)


# Αν θέλεις να φορτώσεις το καλύτερο μοντέλο


best_model = load_model('/media/stathis/StathisUSB/final_classification_march_9/models/10/lstm_dm_best_model.h5')

# Και να κάνεις την πρόβλεψη με το καλύτερο μοντέλο
y_pred = best_model.predict(X_new)
y_pred = model.predict(X_new)
predicted_label = np.argmax(y_pred, axis=1)
df = pd.concat([data_predict, pd.Series(predicted_label, name='predicted')], axis=1)

# Save predictions to CSV
df.to_csv('/media/stathis/StathisUSB/final_classification_march_9/results/10/dm_lstm.csv'

