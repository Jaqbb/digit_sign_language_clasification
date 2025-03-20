import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Relatywna ścieżka do danych względem katalogu projektu
data_dir = os.path.join(os.getcwd(), 'American Sign Language Digits Dataset')


# Pobranie listy wszystkich obrazów i etykiet klas
all_images = []
all_labels = []

for class_dir in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_dir)
    if os.path.isdir(class_path):
        for img in os.listdir(class_path):
            all_images.append(os.path.join(class_path, img))
            all_labels.append(class_dir)

# Konwersja do DataFrame
df = pd.DataFrame({
    'filename': all_images,
    'class': all_labels
})

# Podział danych na 70% treningowe i 30% testowe
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['class'], random_state=42)  # 70% treningowe
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['class'], random_state=42)  # 15% walidacyjne i testowe

# Generator danych
data_gen = ImageDataGenerator(rescale=1./255)

# Wczytywanie danych treningowych
train_data = data_gen.flow_from_dataframe(
    train_df,
    x_col='filename',
    y_col='class',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)
# Wczytywanie danych testowych
test_data = data_gen.flow_from_dataframe(
    test_df,
    x_col='filename',
    y_col='class',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Kolejność ma znaczenie do oceny wyników
)
val_data = data_gen.flow_from_dataframe(
    dataframe=val_df,  # Dane walidacyjne
    x_col='filename',  # Kolumna z nazwami plików obrazów
    y_col='class',  # Kolumna z etykietami klas
    target_size=(64, 64),  # Rozmiar do którego skalowane są obrazy
    batch_size=32,  # Liczba próbek na batch
    class_mode='categorical',  # Dane są skategoryzowane (jeden obraz -> jedna klasa)
    shuffle=False  # Kolejność próbek jest zachowana (ważne do ewaluacji)
)

# Liczba klas
num_classes = len(train_data.class_indices)

# Definicja modelu CNN
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(64, 64, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')  # Użycie num_classes
])

# Kompilacja modelu
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Trenowanie modelu
history = model.fit(
    train_data,
    validation_data=test_data,  # Używamy zbioru testowego do walidacji
    epochs=3
)

# Ocena modelu na danych testowych
test_loss, test_acc = model.evaluate(test_data)
print(f"Test accuracy: {test_acc * 100:.2f}%")

# Uzyskanie predykcji z modelu
y_true = test_data.classes  # Prawdziwe etykiety
y_pred = model.predict(test_data)
y_pred_classes = np.argmax(y_pred, axis=1)  # Konwersja do klas


# Macierz konfuzji
cm = confusion_matrix(y_true, y_pred_classes)
print("Confusion Matrix:\n", cm)

# Obliczanie miar dla klasyfikacji
accuracy = accuracy_score(y_true, y_pred_classes)
print(f"Overall Accuracy: {accuracy * 100:.2f}%")

class_report = classification_report(y_true, y_pred_classes, target_names=test_data.class_indices.keys())
print("Classification Report:\n", class_report)

# Specyficzność (specificity) i czułość (sensitivity) dla każdej klasy
for i in range(len(cm)):
    tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
    fp = cm[:, i].sum() - cm[i, i]
    fn = cm[i, :].sum() - cm[i, i]
    tp = cm[i, i]

    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    print(f"Class {i} - Specificity: {specificity:.2f}, Sensitivity: {sensitivity:.2f}")

# Dokładność dla każdej klasy
class_accuracy = cm.diagonal() / cm.sum(axis=1)  # Prawidłowe predykcje podzielone przez liczbę wszystkich próbek dla każdej klasy

# Wyświetlenie wyników
for i, acc in enumerate(class_accuracy):
    print(f"Accuracy for class {i}: {acc * 100:.2f}%")


# Wizualizacja macierzy konfuzji
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_data.class_indices.keys(), yticklabels=test_data.class_indices.keys())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Wizualizacja krzywych uczenia
plt.figure(figsize=(12, 5))

# Wykres dokładności
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Wykres funkcji straty
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Zapisanie modelu do pliku
model.save('sign_language_model.h5')
