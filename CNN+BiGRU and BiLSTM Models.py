import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import time
from datetime import date
import plotly.graph_objects as go
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import Counter
from sklearn.model_selection import train_test_split
import cufflinks as cf
import pandas as pd
cf.go_offline()

data_df = pd.read_csv("data.csv")
print(f"Data shape {data_df.shape}")
data = data_df.iloc[:700000,:]
print(f"Data shape {data.shape}")

df_spam = data[data['target']==0]

df_ham = data[data['target']==1]

df_ham_downsampled = df_ham.sample(df_spam.shape[0])

df_balanced = pd.concat([df_ham_downsampled, df_spam])

pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

def remove_url(text):
    no_html= pattern.sub('',text)
    return no_html
df_balanced.text = df_balanced.text.apply(lambda x: remove_url(x))

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [w for w in text if not w in set(stopwords.words('english'))]
    text = ' '.join(text)
    return text

df_balanced.text = df_balanced.text.apply(lambda x : clean_text(x))

def counter_word(text):
    count = Counter()
    for i in text.values:
        for word in i.split():
            count[word] += 1
    return count

text_values = df_balanced.text
counter = counter_word(text_values)
print(f"The len of words is: {len(counter)}")

train, test, train_category, test_category = train_test_split(df_balanced.text, df_balanced.target, test_size=0.30, random_state=42, stratify=df_balanced.target)

vocab_size = len(counter)
embedding_dim = 32

max_length = 20
trunc_type = 'post'
padding_type = 'pre'

oov_tok = "<XXX>"
training_size = int(len(train) * 0.8)
seq_len = 12

training_sentences = train[0:training_size]
training_labels = train_category[0:training_size]

valid_sentences = train[training_size:]
valid_labels = train_category[training_size:]

print('The Shape of training ',training_sentences.shape)
print('The Shape of testing',valid_sentences.shape)

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index


training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
T = training_padded.shape[0]
print("The shape of training data is: ",training_padded.shape, T)


valid_sequences = tokenizer.texts_to_sequences(valid_sentences)
valid_padded = pad_sequences(valid_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
print("The shape of validation data is: ",valid_padded.shape)



#Model Definition with LSTM

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length),
#     tf.keras.layers.LSTM(64, return_sequences=True),
#     tf.keras.layers.LSTM(64),
#     tf.keras.layers.Dense(32, activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid') # binary classification
# ])


# #using BERT
# text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
# preprocessed_text = bert_preprocess(text_input)
# outputs = bert_encoder(preprocessed_text)

# # Neural network layers
# l = tf.keras.layers.Dropout(0.1, name="dropout")(outputs['pooled_output'])
# l = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(l)

# # Use inputs and outputs to construct a final model
# model = tf.keras.Model(inputs=[text_input], outputs = [l])



# # ##working model
# i = tf.keras.layers.Input(shape=(max_length,))
# x= tf.keras.layers.Embedding(vocab_size+1, embedding_dim)(i)
# x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)
# x = tf.keras.layers.Dropout(0.05)(x)    #optional might be better to use dropout
# x = tf.keras.layers.GlobalMaxPooling1D()(x)
# x = tf.keras.layers.Dense(1, activation='sigmoid')(x)


# #Ensemble CNN-BiGRU
def ensemble_CNN_BiGRU(filters = 100, kernel_size = 3, activation='relu', max_length = max_length):
  
    # Channel 1D CNN
    input1 = tf.keras.layers.Input(shape=(max_length,))
    embeddding1 = tf.keras.layers.Embedding(vocab_size+1, embedding_dim)(input1)
    conv1 = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', 
                   kernel_constraint= tf.keras.constraints.MaxNorm( max_value=3, axis=[0,1]))(embeddding1)
    pool1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(conv1)
    flat1 = tf.keras.layers.Flatten()(pool1)
    drop1 = tf.keras.layers.Dropout(0.25)(flat1)
    dense1 = tf.keras.layers.Dense(40, activation='relu')(drop1)
    dense1 = tf.keras.layers.Dense(10, activation='relu')(dense1)
    drop1 = tf.keras.layers.Dropout(0.25)(dense1)
    out1 = tf.keras.layers.Dense(1, activation='sigmoid')(drop1)
    
    # Channel BiGRU
    input2 = tf.keras.layers.Input(shape=(max_length,))
    embeddding2 = tf.keras.layers.Embedding(vocab_size+1, embedding_dim, mask_zero=True)(input2)
    gru2 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64))(embeddding2)
    drop2 = tf.keras.layers.Dropout(0.25)(gru2)
    out2 = tf.keras.layers.Dense(1, activation='sigmoid')(drop2)
    
    merged = tf.keras.layers.concatenate([out1, out2])
    
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(merged)
    model = tf.keras.models.Model(inputs=[input1, input2], outputs=outputs)
    
    model.compile( loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

model = ensemble_CNN_BiGRU()

#CNN with better config
# i = tf.keras.layers.Input(shape=(max_length,))
# x = tf.keras.layers.Embedding(vocab_size+1, embedding_dim)(i)
# x = tf.keras.layers.Conv1D(32, 3, activation='relu')(x)
# x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.layers.Dropout(0.20)(x)
# x = tf.keras.layers.MaxPooling1D(3)(x)
# x = tf.keras.layers.Conv1D(64, 3, activation='relu')(x)
# x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.layers.Dropout(0.20)(x)
# x = tf.keras.layers.Conv1D(128, 3, activation='relu')(x)
# x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.layers.Dropout(0.20)(x)
# x = tf.keras.layers.GlobalMaxPooling1D()(x)
# x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

# model = tf.keras.models.Model(inputs=i, outputs=x)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss= 'binary_crossentropy' , metrics=['accuracy'])

# ENsemble CNN BiGRU


model.summary()


num_epochs = 2
history = model.fit([training_padded, training_padded], training_labels,batch_size =10,  epochs=num_epochs, validation_data=([valid_padded, valid_padded], valid_labels))

# num_epochs = 2
# history = model.fit(training_padded, training_labels,batch_size =32,  epochs=num_epochs, validation_data=(valid_padded, valid_labels))

model_loss = pd.DataFrame(history.history)
model_loss.plot()
model_loss[['accuracy','val_accuracy']].plot(xlabel= 'epochs', ylabel = 'accuracy', ylim=[0.75,1])