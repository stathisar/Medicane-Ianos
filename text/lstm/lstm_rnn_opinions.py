import numpy as np
import pandas as pd
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
# Δείγμα δεδομένων (Μπορείς να φορτώσεις ένα πραγματικό dataset)

 
#df = pd.DataFrame(data)

df = pd.read_csv('/media/stathis/StathisUSB/final_classification_march_9/training_data/training_792.csv')

# Προεπεξεργασία του κειμένου
def preprocess_text(text):
    textstr = str(text)  # Μικρά γράμματα
    text = textstr.lower()
    text = re.sub(r'\W', ' ', text)  # Αφαίρεση ειδικών χαρακτήρων
    text = re.sub(r'\s+', ' ', text)  # Αφαίρεση πολλαπλών κενών
    text = text.strip()
    return text

df['processed'] = df['translated'].apply(preprocess_text)
# Χρήση Tokenizer για τη μετατροπή των λέξεων σε ακολουθίες αριθμών
MAX_NB_WORDS = 5000  # Μέγιστος αριθμός μοναδικών λέξεων
MAX_SEQUENCE_LENGTH = 50  # Μέγιστο μήκος ακολουθίας
EMBEDDING_DIM = 100  # Διάσταση embedding

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True)
tokenizer.fit_on_texts(df['processed'].values)
word_index = tokenizer.word_index
print(f"Found {len(word_index)} unique tokens.")
#df = df.rename(columns={"consequence score":"label"})
X = tokenizer.texts_to_sequences(df['processed'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

#df['label'] = df['identification'].apply(lambda x: re.sub('*', '', str(x)))
# Μετατροπή των ετικετών σε κατηγορίες
df['label'] = df['emotions_opinions_etc']
Y = to_categorical(df['label'])

# Χωρισμός των δεδομένων σε training και validation set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Κατασκευή του μοντέλου LSTM-RNN
model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))  # 2 κατηγορίες (positive, negative)

csv_logger = CSVLogger('/media/stathis/StathisUSB/final_classification_march_9/models/10/emotions_opinion_etc.csv', append=False)


# Σύνθεση του μοντέλου
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Εκπαίδευση του μοντέλου
epochs = 100
batch_size = 1024
checkpoint = ModelCheckpoint('lstm_emotions_best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=30, mode='max', verbose=1, restore_best_weights=True)

history = model.fit(X_train, 
			Y_train, 
			epochs=epochs, 
			batch_size=batch_size, 
			validation_data=(X_test, Y_test), 
			verbose=2, 
			callbacks=[csv_logger,checkpoint, early_stopping])



# Evaluate the model
score, acc = model.evaluate(X_test, Y_test, verbose=2)
print(f'Test score: {score}')
print(f'Test accuracy: {acc}')


#import best model in terms of OvF
best_model = load_model('/media/stathis/StathisUSB/final_classification_march_9/models/10/lstm_opinions_best_model.h5')


# Load predictions
data_predict = pd.read_csv('/media/stathis/StathisUSB/final_classification_march_9/text_data/DataTextAllSources_March_15_Removed_duplicates.csv')
data_predict['processed'] = data_predict['translated'].apply(preprocess_text)
X_new = tokenizer.texts_to_sequences(data_predict['processed'])
X_new = pad_sequences(X_new, maxlen=MAX_SEQUENCE_LENGTH)


#Prediction with best model in terms of OvF
y_pred = best_model.predict(X_new)
y_pred = model.predict(X_new)
predicted_label = np.argmax(y_pred, axis=1)
df = pd.concat([data_predict, pd.Series(predicted_label, name='predicted')], axis=1)

# Save predictions to CSV
df.to_csv('/media/stathis/StathisUSB/final_classification_march_9/results/10/opinions_lstm.csv'
#model.save('/media/stathis/KINGSTON/ML-Water/filtered_data/final_classification_march_9/models/10/emotions_opinion_etc.h5')

#with open('/media/stathis/KINGSTON/ML-Water/filtered_data/final_classification_march_9/models/10/emotions_opinion_etc', 'wb') #as file_pi:
#    pickle.dump(history.history, file_pi)


# Αξιολόγηση του μοντέλου
score, acc = model.evaluate(X_test, Y_test, verbose=2)
print(f'Test score: {score}')
print(f'Test accuracy: {acc}')


data_predict = pd.read_csv('/media/stathis/StathisUSB/final_classification_march_9/text_data/DataTextAllSources_March_15_Removed_duplicates.csv')
data_predict['processed'] = data_predict['translated'].apply(preprocess_text)
print(data_predict.head())

X_new = tokenizer.texts_to_sequences(data_predict['processed'])
X_new = pad_sequences(X_new, maxlen=MAX_SEQUENCE_LENGTH)
y_pred = model.predict(X_new)
print(y_pred)


predicted_label = np.argmax(y_pred, axis=1)
df = pd.concat([data_predict, pd.Series(predicted_label, name='predicted')], axis=1)

df.to_csv('/media/stathis/StathisUSB/final_classification_march_9/results/10/emotions_opinion_etc_lstm.csv')

