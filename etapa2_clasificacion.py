"""
ETAPA 2: Clasificación con Transfer Learning
- Cargar encoder pre-entrenado (congelado)
- Agregar capas de clasificación
- Entrenar solo las nuevas capas
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical


print("ETAPA 2: CLASIFICACIÓN CON TRANSFER LEARNING")


# 1. Cargar datos
(x_train_flat, y_train), (x_test_flat, y_test), _, _ = cargar_y_preparar_fashion_mnist()

# Convertir etiquetas a one-hot encoding
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

print(f"Forma datos entrenamiento: {x_train_flat.shape}")
print(f"Forma etiquetas entrenamiento: {y_train_cat.shape}")

# 2. Cargar encoder pre-entrenado de la Etapa 1
print("\n--- Cargando encoder pre-entrenado ---")
encoder_preentrenado = keras.models.load_model('/content/drive/MyDrive/fashion_mnist_modelos/encoder.h5')
encoder_preentrenado.summary()

# 3. Congelar el encoder
encoder_preentrenado.trainable = False
print("\n✓ Encoder congelado (trainable = False)")

# 4. Construir modelo de clasificación completo
print("\n--- Construyendo modelo de clasificación ---")
clasificador_input = layers.Input(shape=(784,), name='clasificador_input')

# Encoder pre-entrenado (congelado)
encoded_features = encoder_preentrenado(clasificador_input)

# 1. Verificar etiquetas 
print(np.unique(y_train))  # Debe ser [0 1 2 3 4 5 6 7 8 9]

# Nuevas capas para clasificación - AUMENTADA CAPACIDAD
x = layers.Dense(128, activation='relu', name='clasificador_dense_1')(encoded_features)
x = layers.Dropout(0.3, name='clasificador_dropout_1')(x)
x = layers.Dense(64, activation='relu', name='clasificador_dense_2')(x)
x = layers.Dropout(0.3, name='clasificador_dropout_2')(x)
x = layers.Dense(32, activation='relu', name='clasificador_dense_3')(x)
clasificador_output = layers.Dense(10, activation='softmax', name='clasificador_output')(x)

clasificador = models.Model(clasificador_input, clasificador_output, name='clasificador_fashion_mnist')
clasificador.summary()

# 5. Compilar modelo
clasificador.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 6. Entrenar solo las nuevas capas
print("\n--- Entrenando capas de clasificación (encoder congelado) ---")
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=0.00001
)

history = clasificador.fit(
    x_train_flat, y_train_cat,
    epochs=30,
    batch_size=64,
    validation_data=(x_test_flat, y_test_cat),
    class_weight=class_weight_dict,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# 7. Evaluar modelo
print("\n--- Evaluando modelo final ---")
test_loss, test_accuracy = clasificador.evaluate(x_test_flat, y_test_cat, verbose=0)
print(f"✓ Exactitud en test set: {test_accuracy:.4f}")
print(f"✓ Pérdida en test set: {test_loss:.4f}")

# 8. Descongelar y fine-tuning
print("\n--- Fine-tuning (descongelar encoder) ---")
encoder_preentrenado.trainable = True
clasificador.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # Learning rate más bajo
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_finetune = clasificador.fit(
    x_train_flat, y_train_cat,
    epochs=10,
    batch_size=64,
    validation_data=(x_test_flat, y_test_cat),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# 9. Evaluar después de fine-tuning
test_loss_ft, test_accuracy_ft = clasificador.evaluate(x_test_flat, y_test_cat, verbose=0)
print(f"\n--- Resultados finales ---")
print(f"✓ Exactitud después de fine-tuning: {test_accuracy_ft:.4f}")
print(f"✓ Mejora: +{test_accuracy_ft - test_accuracy:.4f}")

# 10. Guardar modelo final
clasificador.save('modelos/clasificador_final.h5')
print("\n✓ Modelo clasificador guardado en 'modelos/clasificador_final.h5'")

# 11. Visualizar resultados
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento (etapa 2)')
plt.plot(history.history['val_accuracy'], label='Validación (etapa 2)')
if history_finetune.history['accuracy']:
    plt.plot(range(30, 30+len(history_finetune.history['accuracy'])),
             history_finetune.history['accuracy'], label='Fine-tuning train')
    plt.plot(range(30, 30+len(history_finetune.history['val_accuracy'])),
             history_finetune.history['val_accuracy'], label='Fine-tuning val')
plt.title('Exactitud del Clasificador')
plt.xlabel('Época')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento (etapa 2)')
plt.plot(history.history['val_loss'], label='Validación (etapa 2)')
if history_finetune.history['loss']:
    plt.plot(range(30, 30+len(history_finetune.history['loss'])),
             history_finetune.history['loss'], label='Fine-tuning train')
    plt.plot(range(30, 30+len(history_finetune.history['val_loss'])),
             history_finetune.history['val_loss'], label='Fine-tuning val')
plt.title('Pérdida del Clasificador')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('clasificador_training_history.png')
plt.show()

print("\n✓ ETAPA 2 COMPLETADA EXITOSAMENTE")
print(f"✓ Modelo final: Exactitud {test_accuracy_ft:.4f}")
