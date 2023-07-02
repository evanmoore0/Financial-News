from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding


def create_model():
    model = Sequential([

    # InputLayer(MAX_SEQUENCE_LENGTH),
    
    Embedding(len(d2v_model.wv)+1,20,input_length=X.shape[1],weights=[embedding_matrix],trainable=True),
    # SpatialDropout1D(0.2),
    # Conv1D(100, kernel_size=3, strides = 2, padding='same', activation='relu'),
    MaxPooling1D(pool_size=2),
    Bidirectional(LSTM(100)),
    Dense(3, activation='sigmoid')


])