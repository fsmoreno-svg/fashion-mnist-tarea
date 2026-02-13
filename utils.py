import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

def cargar_y_preparar_fashion_mnist():
    """Carga y prepara los datos de Fashion MNIST"""
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    # Normalizar y aplanar
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    x_train_flat = x_train.reshape(-1, 784)
    x_test_flat = x_test.reshape(-1, 784)

    return (x_train_flat, y_train), (x_test_flat, y_test), (x_train, y_train), (x_test, y_test)
def visualizar_reconstruccion(autoencoder, x_test, x_test_flat, n=10):
    """Visualiza reconstrucciones del autoencoder"""
    reconstruidas = autoencoder.predict(x_test_flat[:n])

    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        plt.title("Original")
        plt.axis('off')

        # Reconstruida
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstruidas[i].reshape(28, 28), cmap='gray')
        plt.title("Reconstruida")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('reconstrucciones_autoencoder.png')
    plt.show()