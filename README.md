
# Klasyfikacja cyfr w języku migowym z wykorzystaniem CNN

## 📝 **Opis projektu**

Celem projektu jest stworzenie modeli sieci konwolucyjnych (CNN) do klasyfikacji cyfr w języku migowym. Modele te mają na celu wspomaganie komunikacji między osobami głuchymi a słyszącymi poprzez dokładne rozpoznawanie znaków języka migowego. Projekt wykorzystuje **American Sign Language Digits Dataset**, który zawiera obrazy przedstawiające cyfry od 0 do 9 w języku migowym.

## 📂 **Zawartość repozytorium**

- **Klasyfikacja_CNN.py** – Skrypt implementujący pierwszy model klasyfikacyjny oparty na CNN.
- **Klasyfikacja_CNN_modified.py** – Skrypt implementujący zmodyfikowaną wersję modelu CNN.
- **sign_language_model.h5** – Wytrenowany pierwszy model CNN zapisany w formacie H5.
- **sign_language_model2.h5** – Wytrenowany drugi model CNN zapisany w formacie H5.
- **American Sign Language Digits Dataset** – Folder zawierający dane treningowe i testowe z obrazami cyfr w języku migowym.

## ⚙️ **Wymagania**

Aby uruchomić projekt, musisz mieć zainstalowane poniższe oprogramowanie i biblioteki:

- **Python (>=3.8)**
- **TensorFlow**
- **Keras**
- **NumPy**
- **Matplotlib**
- **Scikit-learn**

Możesz zainstalować wymagane biblioteki za pomocą poniższego polecenia:

```bash
pip install tensorflow keras numpy matplotlib scikit-learn
```

## 📊 **Dane wejściowe**

Dane pochodzą z zestawu **American Sign Language Digits Dataset**, który zawiera obrazy przedstawiające cyfry w języku migowym. Struktura folderu danych wygląda następująco:

```
American Sign Language Digits Dataset/
├── 0/
├── 1/
├── 2/
...
├── 9/
```

Każdy folder zawiera **500 obrazów** przedstawiających daną cyfrę, co daje łącznie **5000 obrazów**.

## 🛠️ **Instrukcja działania**

### 1️⃣ **Przygotowanie danych**

Upewnij się, że folder **American Sign Language Digits Dataset** znajduje się w tej samej lokalizacji co skrypty. Dane będą automatycznie ładowane i przetwarzane przez skrypty.

### 2️⃣ **Uruchomienie modelu 1**

Model 1 można uruchomić za pomocą skryptu `Klasyfikacja_CNN.py`:

```bash
python Klasyfikacja_CNN.py
```

Skrypt:

- Wczytuje dane z folderu datasetu.
- Trenuje model CNN.
- Zapisuje wytrenowany model do pliku `sign_language_model.h5`.

### 3️⃣ **Uruchomienie modelu 2**

Model 2 można uruchomić za pomocą skryptu `Klasyfikacja_CNN_modified.py`:

```bash
python Klasyfikacja_CNN_modified.py
```

Podobnie jak model 1, ten skrypt:

- Wczytuje dane z folderu datasetu.
- Trenuje alternatywną wersję modelu CNN.
- Zapisuje wytrenowany model do pliku `sign_language_model2.h5`.

## 📈 **Wyniki**

Oba modele generują wyniki dokładności klasyfikacji na danych testowych. Wyniki te są wyświetlane zarówno na wykresach, jak i w konsoli, co pozwala na łatwą ocenę skuteczności modelu.

## ⚠️ **Uwaga**

Przed uruchomieniem projektu upewnij się, że:

- Wszystkie pliki znajdują się w odpowiednich lokalizacjach.
- Wymagane biblioteki zostały poprawnie zainstalowane.
