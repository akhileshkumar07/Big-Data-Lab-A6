import keras
from keras import layers

# Function to load the MNIST dataset and preprocess it
def load_dataset():    
    num_classes = 10
    # Load the MNIST dataset
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()
    # Reshape the input data
    x_train = X_train.reshape(60000, -1)
    x_test = X_test.reshape(10000, -1)
    # Normalize pixel values to range between 0 and 1
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    # Convert labels to one-hot encoding
    y_train = keras.utils.to_categorical(Y_train, num_classes)
    y_test = keras.utils.to_categorical(Y_test, num_classes)

    return x_train, y_train, x_test, y_test

# Function to define and train the neural network model
def train_model():
    # Load and preprocess the dataset
    x_train, y_train, x_test, y_test = load_dataset()
    # Define the model architecture
    model = keras.Sequential()
    model.add(layers.Dense(256, activation='relu', input_shape=(784,)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Train the model
    history = model.fit(x_train, y_train, epochs=15, validation_data=(x_test, y_test))
    # Save the trained model
    model.save("mnist_model.keras")
    # Print the final training and validation accuracy
    print('train_accuracy: ', history.history['accuracy'][-1])
    print('validation_accuracy: ', history.history['val_accuracy'][-1])

if __name__ == "__main__":
    # Call the train_model function when the script is run
    train_model()