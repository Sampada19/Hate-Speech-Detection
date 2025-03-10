pip install tensorflow pandas matplotlib scikit-learn

import tensorflow as tf
print(tf.__version__)

import os
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding
from tensorflow.keras.saving import register_keras_serializable



df=pd.read_csv("C:\\Users\\sampa\\Downloads\\DTL\\train.csv")

df.iloc[2]['comment_text']

df[df.columns[2:]].iloc[5]

from tensorflow.keras.layers import TextVectorization

X=df['comment_text']
y=df[df.columns[2:]].values

df[df.columns[2:]].values


MAX_FEATURES=2000000


vectorizer=TextVectorization(max_tokens=MAX_FEATURES,output_sequence_length=1800,output_mode='int')

vectorizer.adapt(X.values)

vectorizer.get_vocabulary()

vectorized_text=vectorizer(X.values)

vectorized_text


dataset=tf.data.Dataset.from_tensor_slices((vectorized_text,y))
dataset=dataset.cache()
dataset=dataset.shuffle(160000)
dataset=dataset.batch(16)
dataset=dataset.prefetch(8)

batch_X,batch_y=dataset.as_numpy_iterator().next()

train=dataset.take(int(len(dataset)*0.7))
val=dataset.skip(int(len(dataset)*0.7)).take(int(len(dataset)*0.2))
test=dataset.skip(int(len(dataset)*0.9)).take(int(len(dataset)*0.1))

train_generator=train.as_numpy_iterator()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding

model = Sequential()
# Create the embedding layer 
model.add(Embedding(MAX_FEATURES+1, 32))
# Bidirectional LSTM Layer
model.add(Bidirectional(LSTM(32, activation='tanh')))
# Feature extractor Fully connected layers
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
# Final layer 
model.add(Dense(6, activation='sigmoid'))

model.compile(loss='BinaryCrossentropy', optimizer='Adam')

history = model.fit(train, epochs=1, validation_data=val)


history.history

from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy

pre=Precision()
re=Recall()
acc=CategoricalAccuracy()

for batch in test.as_numpy_iterator(): 
    # Unpack the batch 
    X_true, y_true = batch
    # Make a prediction 
    yhat = model.predict(X_true)
    
    # Flatten the predictions
    y_true = y_true.flatten()
    yhat = yhat.flatten()
    
    pre.update_state(y_true, yhat)
    re.update_state(y_true, yhat)
    acc.update_state(y_true, yhat)


print(f'Precision: {pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}')

print(f'Precision: {pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}')

model.summary()


