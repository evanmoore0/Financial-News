from tensorflow import keras

import numpy as np
from preprocess import tokenizer, X
from keras_preprocessing.sequence import pad_sequences



model = keras.models.load_model('./saved_models')

message = ["Is Arista Networks a Buy as Wall Street Analysts Look Optimistic?"]

seq = tokenizer.texts_to_sequences(message)


padded = pad_sequences(seq, maxlen=50)

pred = model.predict(padded)

labels = ["0", "1", "2"]

print(pred, labels[np.argmax(pred)])

#Convert sting to numeric
sentiment  = {'positive': 0,'neutral': 1,'negative':2} 