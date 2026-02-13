"""
ETAPA 1: Autoencoder para Fashion MNIST
- Extracción de características
- Entrenamiento del encoder
- Guardar encoder para etapa 2
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt



print("ETAPA 1: ENTRENAMIENTO DE AUTOENCODER")


# 1. Cargar datos
(x_train_flat, y_train), (x_test_flat, y_test), (x_train_img, y_train_img), (x_test_img, y_test_img) = cargar_y_preparar_fashion_mnist()

print(f"Forma datos entrenamiento: {x_train_flat.shape}")
print(f"Forma datos prueba: {x_test_flat.shape}")

# 2. Definir dimensiones
input_dim = 784
latent_dim = 32  # Dimensión del espacio latente

# 3. Construir ENCODER
print("\n--- Construyendo Encoder ---")
encoder_input = layers.Input(shape=(input_dim,), name='encoder_input')
x = layers.Dense(256, activation='relu', name='encoder_dense_1')(encoder_input)
x = layers.BatchNormalization(name='encoder_bn_1')(x)
x = layers.Dropout(0.2, name='encoder_dropout_1')(x)
x = layers.Dense(128, activation='relu', name='encoder_dense_2')(x)
x = layers.BatchNormalization(name='encoder_bn_2')(x)
x = layers.Dropout(0.2, name='encoder_dropout_2')(x)
x = layers.Dense(64, activation='relu', name='encoder_dense_3')(x)
latent = layers.Dense(latent_dim, activation='relu', name='espacio_latente')(x)

encoder = models.Model(encoder_input, latent, name='encoder')
encoder.summary()

# 4. Construir DECODER
print("\n--- Construyendo Decoder ---")
decoder_input = layers.Input(shape=(latent_dim,), name='decoder_input')
x = layers.Dense(64, activation='relu', name='decoder_dense_1')(decoder_input)
x = layers.Dense(128, activation='relu', name='decoder_dense_2')(x)
x = layers.Dense(256, activation='relu', name='decoder_dense_3')(x)
decoder_output = layers.Dense(input_dim, activation='sigmoid', name='decoder_output')(x)

decoder = models.Model(decoder_input, decoder_output, name='decoder')
decoder.summary()

# 5. Construir AUTOENCODER
print("\n--- Construyendo Autoencoder ---")
autoencoder_input = layers.Input(shape=(input_dim,), name='autoencoder_input')
encoded = encoder(autoencoder_input)
decoded = decoder(encoded)
autoencoder = models.Model(autoencoder_input, decoded, name='autoencoder')
autoencoder.summary()

# 6. Compilar y entrenar
autoencoder.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print("\n--- Entrenando Autoencoder ---")
history = autoencoder.fit(
    x_train_flat, x_train_flat,
    epochs=50,
    batch_size=128,
    validation_data=(x_test_flat, x_test_flat),
    verbose=1
)

# 7. Visualizar reconstrucciones
visualizar_reconstruccion(autoencoder, x_test_img, x_test_flat)

# 8. Guardar modelos
print("\n--- Guardando modelos ---")
encoder.save('modelos/encoder.keras')
autoencoder.save('modelos/autoencoder_completo.keras')
print("✓ Modelos guardados exitosamente")

# 9. Evaluar reconstrucción
loss = autoencoder.evaluate(x_test_flat, x_test_flat, verbose=0)
print(f"\nError de reconstrucción en test set: {loss[0]:.4f}")
print(f"MAE en test set: {loss[1]:.4f}")

# 10. Visualizar pérdida
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida del Autoencoder')
plt.xlabel('Época')
plt.ylabel('Loss (MSE)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Entrenamiento')
plt.plot(history.history['val_mae'], label='Validación')
plt.title('MAE del Autoencoder')
plt.xlabel('Época')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.show()

print("etapa 1 completada")