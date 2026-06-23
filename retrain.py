import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

print("TensorFlow version:", tf.__version__)

MAX_FEATURES = 10000
MAXLEN = 500
EMBEDDING_DIM = 128
RNN_UNITS = 128
EPOCHS = 5
BATCH_SIZE = 64

print("Loading IMDB dataset...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_FEATURES)

print("Padding sequences...")
x_train = pad_sequences(x_train, maxlen=MAXLEN)
x_test  = pad_sequences(x_test,  maxlen=MAXLEN)

model = Sequential([
    Embedding(input_dim=MAX_FEATURES, output_dim=EMBEDDING_DIM),
    SimpleRNN(RNN_UNITS),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

print("Training...")
model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)

loss, acc = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {acc:.4f}")

model.save("model.h5")
print("\nDone! model.h5 saved.")