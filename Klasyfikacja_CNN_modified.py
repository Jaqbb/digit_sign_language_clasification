import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


# Relatywna ścieżka do danych względem katalogu projektu
data_dir = os.path.join(os.getcwd(), 'American Sign Language Digits Dataset')

# Pobranie listy wszystkich obrazów i etykiet klas
all_images = []  # Lista przechowująca ścieżki do obrazów
all_labels = []  # Lista przechowująca etykiety klas

for class_dir in os.listdir(data_dir):  # Iteracja po katalogach klas
    class_path = os.path.join(data_dir, class_dir)  # Pełna ścieżka do katalogu klasy
    if os.path.isdir(class_path):  # Sprawdzenie, czy to katalog
        for img in os.listdir(class_path):  # Iteracja po plikach w katalogu
            all_images.append(os.path.join(class_path, img))  # Dodanie pełnej ścieżki obrazu
            all_labels.append(class_dir)  # Dodanie etykiety klasy

# Konwersja do DataFrame
df = pd.DataFrame({  # Tworzenie DataFrame z obrazami i etykietami
    'filename': all_images,  # Kolumna z nazwami plików
    'class': all_labels  # Kolumna z etykietami klas
})

# Podział danych na 70% treningowe, 15% walidacyjne i 15% testowe
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['class'], random_state=42)  # 70% treningowe
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['class'], random_state=42)  # 15% walidacyjne i testowe

# Generator danych
data_gen = ImageDataGenerator(rescale=1./255)  # Skaluje wartości pikseli obrazów na zakres [0, 1]

train_data = data_gen.flow_from_dataframe(
    dataframe=train_df,  # Dane treningowe
    x_col='filename',  # Kolumna z nazwami plików obrazów
    y_col='class',  # Kolumna z etykietami klas
    target_size=(64, 64),  # Rozmiar do którego skalowane są obrazy
    batch_size=32,  # Liczba próbek na batch
    class_mode='categorical',  # Dane są skategoryzowane (jeden obraz -> jedna klasa)
    shuffle=True  # Losowe mieszanie próbek
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

test_data = data_gen.flow_from_dataframe(
    dataframe=test_df,  # Dane testowe
    x_col='filename',  # Kolumna z nazwami plików obrazów
    y_col='class',  # Kolumna z etykietami klas
    target_size=(64, 64),  # Rozmiar do którego skalowane są obrazy
    batch_size=32,  # Liczba próbek na batch
    class_mode='categorical',  # Dane są skategoryzowane (jeden obraz -> jedna klasa)
    shuffle=False  # Kolejność próbek jest zachowana (ważne do ewaluacji)
)

# Liczba klas
num_classes = len(train_data.class_indices)  # Oblicza liczbę unikalnych klas na podstawie indeksów

# Definicja drugiego modelu CNN
model2 = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(64, 64, 3)),  # Warstwa wejściowa dla obrazów 64x64x3 (RGB)
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),  # Warstwa konwolucyjna z 32 filtrami
    tf.keras.layers.BatchNormalization(),  # Normalizacja wsadowa dla stabilizacji uczenia
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),  # Warstwa poolingowa (zmniejsza wymiary)
    tf.keras.layers.Dropout(0.25),  # Dropout dla redukcji przeuczenia

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),  # Kolejna warstwa konwolucyjna z 64 filtrami
    tf.keras.layers.BatchNormalization(),  # Normalizacja wsadowa
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),  # Pooling
    tf.keras.layers.Dropout(0.25),  # Dropout

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),  # Trzecia warstwa konwolucyjna z 128 filtrami
    tf.keras.layers.BatchNormalization(),  # Normalizacja wsadowa
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),  # Pooling
    tf.keras.layers.Dropout(0.5),  # Dropout z większym współczynnikiem

    tf.keras.layers.GlobalAveragePooling2D(),  # Warstwa GlobalAveragePooling (zmniejsza wymiary w sposób globalny)
    tf.keras.layers.Dense(128, activation='relu'),  # Warstwa gęsta z 128 neuronami
    tf.keras.layers.Dropout(0.5),  # Dropout
    tf.keras.layers.Dense(num_classes, activation='softmax')  # Warstwa wyjściowa z softmax dla klasyfikacji
])

# Kompilacja modelu
model2.compile(optimizer='adam',  # Optymalizator Adam
               loss='categorical_crossentropy',  # Funkcja strat dla klasyfikacji wieloklasowej
               metrics=['accuracy'])  # Metryka oceny modelu

# Trenowanie modelu
history2 = model2.fit(
    train_data,  # Dane treningowe
    validation_data=val_data,  # Dane walidacyjne
    epochs=20  # Liczba epok (można dostosować)
)

# Ocena modelu na danych testowych
test_loss2, test_acc2 = model2.evaluate(test_data)  # Ocena strat i dokładności na zbiorze testowym
print(f"Test accuracy for Model 2: {test_acc2 * 100:.2f}%")

# Uzyskanie predykcji z modelu
y_true = test_data.classes  # Prawdziwe etykiety klas
y_pred2 = model2.predict(test_data)  # Predykcje modelu (prawdopodobieństwa dla każdej klasy)
y_pred_classes2 = np.argmax(y_pred2, axis=1)  # Przekształcenie predykcji na indeksy klas

# Macierz konfuzji
cm2 = confusion_matrix(y_true, y_pred_classes2)  # Obliczanie macierzy konfuzji

# Obliczanie specyficzności i czułości dla każdej klasy
print("\nClass-wise Specificity and Sensitivity for Model 2:")
for i in range(len(cm2)):
    tn = cm2.sum() - (cm2[i, :].sum() + cm2[:, i].sum() - cm2[i, i])  # True negatives
    fp = cm2[:, i].sum() - cm2[i, i]  # False positives
    fn = cm2[i, :].sum() - cm2[i, i]  # False negatives
    tp = cm2[i, i]  # True positives

    specificity = tn / (tn + fp)  # Obliczanie specyficzności
    sensitivity = tp / (tp + fn)  # Obliczanie czułości
    print(f"Class {i} - Specificity: {specificity:.2f}, Sensitivity: {sensitivity:.2f}")

# Dokładność dla każdej klasy
class_accuracy = cm2.diagonal() / cm2.sum(axis=1)  # Prawidłowe predykcje podzielone przez liczbę wszystkich próbek dla każdej klasy

# Wyświetlenie wyników
for i, acc in enumerate(class_accuracy):
    print(f"Accuracy for class {i}: {acc * 100:.2f}%")


# Wizualizacja macierzy konfuzji
plt.figure(figsize=(10, 8))  # Ustawienia rozmiaru wykresu
sns.heatmap(cm2, annot=True, fmt='d', cmap='Greens', xticklabels=test_data.class_indices.keys(), yticklabels=test_data.class_indices.keys())  # Tworzenie mapy cieplnej
plt.xlabel('Predicted')  # Etykieta osi X
plt.ylabel('True')  # Etykieta osi Y
plt.title('Confusion Matrix for Model 2')  # Tytuł wykresu
plt.show()

# Wizualizacja krzywych uczenia
plt.figure(figsize=(12, 5))

# Wykres dokładności
plt.subplot(1, 2, 1)
plt.plot(history2.history['accuracy'], label='Train Accuracy')
plt.plot(history2.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Wykres funkcji straty
plt.subplot(1, 2, 2)
plt.plot(history2.history['loss'], label='Train Loss')
plt.plot(history2.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()


# Zapisanie modelu do pliku
model2.save('sign_language_model2.h5')  # Zapis wytrenowanego modelu do pliku
