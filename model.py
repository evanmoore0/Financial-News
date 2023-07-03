import pandas as pd
import numpy as np

# Matplot
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Bidirectional, MaxPooling1D
from sklearn.model_selection import train_test_split

from preprocess import X, d2v_model, embedding_matrix, df, X_train, X_test, Y_train, Y_test



model = Sequential([    
    Embedding(len(d2v_model.wv)+1,20,input_length=X.shape[1],weights=[embedding_matrix],trainable=True),
    MaxPooling1D(pool_size=2),
    Bidirectional(LSTM(50)),
    Dense(3, activation='sigmoid')

])


model.summary()
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=['acc'])

Y = pd.get_dummies(df['sentiment']).values



history=model.fit(X_train, Y_train, epochs = 50, batch_size=32, verbose = 2)

model.save("./saved_models")

plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('acc')
plt.xlabel('epochs')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('model_accuracy.png')

# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('model_loss.png')


# evaluate the model
_, train_acc = model.evaluate(X_train, Y_train, verbose=2)
_, test_acc = model.evaluate(X_test, Y_test, verbose=2)
print('Train: %.3f, Test: %.4f' % (train_acc, test_acc))



# predict probabilities for test set
yhat_probs = model.predict(X_test, verbose=0)

# predict crisp classes for test set
predict_x=model.predict(X_test) 
yhat_classes=np.argmax(predict_x,axis=1)

# reduce to 1d array
rounded_labels=np.argmax(Y_test, axis=1)


cm = confusion_matrix(rounded_labels, yhat_classes)


lstm_val = confusion_matrix(rounded_labels, yhat_classes)
_, ax = plt.subplots(figsize=(5,5))
sns.heatmap(lstm_val, annot=True, linewidth=0.7, linecolor='cyan', fmt='g', ax=ax, cmap="BuPu")
plt.title('LSTM Classification Confusion Matrix')
plt.xlabel('Y predict')
plt.ylabel('Y test')
plt.show()

