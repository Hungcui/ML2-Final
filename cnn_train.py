import numpy as np
from tensorflow import keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()  # (60000,28,28)    

# Normalize to [0,1] and add channel dim
x_train = (x_train.astype("float32") / 255.0)[..., np.newaxis]  # (60000,28,28,1)
x_test  = (x_test.astype("float32") / 255.0)[..., np.newaxis]   # (10000,28,28,1)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3,3), activation="relu"),   # learns local patterns  
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",  # labels are 0..9 ints
    metrics=["accuracy"]
)

history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("Test accuracy:", test_acc)

save_path = r"D:\ML2-Final\saved_models\mnist_cnn.keras"
model.save(save_path)
print("Saved to:", save_path)

y_pred = np.argmax(model.predict(x_test), axis=1)

cm = confusion_matrix(y_test, y_pred, labels=range(10))
disp = ConfusionMatrixDisplay(cm, display_labels=range(10))
disp.plot(cmap="Blues", xticks_rotation="vertical")
plt.title("MNIST CNN Confusion Matrix")
plt.show()

