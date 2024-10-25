import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# Caminho do conjunto de dados
train_dir = 'dataset/train'
validation_dir = 'dataset/validation'

# Carregar o conjunto de dados usando `image_dataset_from_directory`
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(150, 150),
    batch_size=32,
    label_mode='categorical'
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    image_size=(150, 150),
    batch_size=32,
    label_mode='categorical'
)

# Construindo o modelo CNN com a camada Input
model = Sequential([
    tf.keras.layers.Input(shape=(150, 150, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(train_dataset.class_names), activation='softmax')
])

# Compilando o modelo
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Treinando o modelo
history = model.fit(
    train_dataset,
    epochs=15,
    validation_data=validation_dataset
)

# Função para plotar a acurácia e a perda
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Acurácia de Treinamento')
    plt.plot(epochs, val_acc, 'r', label='Acurácia de Validação')
    plt.title('Acurácia de Treinamento e Validação')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Perda de Treinamento')
    plt.plot(epochs, val_loss, 'r', label='Perda de Validação')
    plt.title('Perda de Treinamento e Validação')
    plt.legend()

    plt.show()

# Plotando os gráficos de treinamento
plot_history(history)
