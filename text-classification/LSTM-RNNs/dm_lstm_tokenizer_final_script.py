import os
import numpy as np
import pandas as pd
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, Callback
import pickle

#set wd
df = pd.read_csv('/home/stathis/Desktop/text_classification/dm/training_766.csv')

# Preprocessing function
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

df['processed'] = df['translated'].apply(preprocess_text)

# Tokenize and pad
MAX_NB_WORDS = 380
MAX_SEQUENCE_LENGTH = 50
EMBEDDING_DIM = 100

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True)
tokenizer.fit_on_texts(df['processed'].values)
X = tokenizer.texts_to_sequences(df['processed'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

# Labels
df['label'] = df['dm']
Y = to_categorical(df['label'])

# Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Build model
model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
model.add(SpatialDropout1D(0.1))
model.add(LSTM(300, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Custom checkpoint
class CustomCheckpoint(Callback):
    def __init__(self, filepath, monitor='val_accuracy', threshold=0.10, verbose=1):
        super(CustomCheckpoint, self).__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.threshold = threshold
        self.verbose = verbose
        self.best_val_acc = -np.inf
    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get('val_accuracy')
        acc = logs.get('accuracy')
        if (acc - val_acc) <= self.threshold:
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                if self.verbose > 0:
                    print(f"\nEpoch {epoch + 1}: val_accuracy improved to {val_acc:.4f}, saving model to {self.filepath}")
                self.model.save(self.filepath)
            else:
                if self.verbose > 0:
                    print(f"\nEpoch {epoch + 1}: val_accuracy did not improve from {self.best_val_acc:.4f}")
        else:
            if self.verbose > 0:
                print(f"\nEpoch {epoch + 1}: Difference between acc ({acc:.4f}) and val_acc ({val_acc:.4f}) too large. Model not saved.")

custom_checkpoint = CustomCheckpoint(
    filepath='/home/stathis/Desktop/text_classification/dm/models/lstm_dm_best_model.h5',
    monitor='val_accuracy',
    threshold=0.10,
    verbose=1
)

csv_logger = CSVLogger('/home/stathis/Desktop/text_classification/dm/models/dm_lstm_model_log.csv', append=False)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=100, mode='max', verbose=1, restore_best_weights=True)

# Train
history = model.fit(
    X_train, Y_train,
    epochs=100,
    batch_size=512,
    validation_data=(X_test, Y_test),
    verbose=2,
    callbacks=[csv_logger, custom_checkpoint]
)

# Save last model
model.save('/home/stathis/Desktop/text_classification/dm/models/lstm_dm_last_model.h5')

# Save tokenizer
with open('/home/stathis/Desktop/text_classification/dm/models/tokenizer_dm.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Load prediction data
data_predict = pd.read_csv('/home/stathis/Desktop/text_classification/dm/DataTextAllSources_7058_manual_cleaning.csv')

# Preprocess text
data_predict['processed'] = data_predict['translated'].apply(preprocess_text)

# Load saved tokenizer
with open('/home/stathis/Desktop/text_classification/dm/models/tokenizer_dm.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Convert texts to sequences and pad
X_new = tokenizer.texts_to_sequences(data_predict['processed'])
X_new = pad_sequences(X_new, maxlen=MAX_SEQUENCE_LENGTH)

# Load best model
best_model_cons = load_model('/home/stathis/Desktop/text_classification/dm/models/lstm_dm_best_model.h5')

# Make predictions
y_pred_cons = best_model_cons.predict(X_new)
predicted_label_cons = np.argmax(y_pred_cons, axis=1)

# Merge predictions with original data
df_dm = pd.concat([data_predict, pd.Series(predicted_label_cons, name='predicted')], axis=1)

# Save predictions
df_dm.to_csv('/home/stathis/Desktop/text_classification/dm/lstm_dm_predicted.csv', index=False)

# Export random sample
df_dm_sample = df_dm.sample(350)
df_dm_sample.to_csv('/home/stathis/Desktop/text_classification/dm/models/lstm_dm_predicted_sample350.csv', index=False)

