import tensorflow as tf

model = tf.keras.models.load_model('model.keras', compile=False)
model.save('model.h5')
print("Done! model.h5 created.")
