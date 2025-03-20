README

Projekt: Klasyfikacja cyfr w języku migowym z wykorzystaniem CNN

Opis projektu

Celem projektu jest stworzenie modeli sieci konwolucyjnych (CNN) do klasyfikacji cyfr w języku migowym. Modele te mają na celu wspomaganie komunikacji między osobami głuchymi a słyszącymi poprzez dokładne rozpoznawanie znaków języka migowego.

Zawartość repozytorium

Klasyfikacja_CNN.py - Skrypt implementujący pierwszy model klasyfikacyjny.

Klasyfikacja_CNN_modified.py - Skrypt implementujący drugi model klasyfikacyjny.

sign_language_model.h5 - Wytrenowany pierwszy model CNN.

sign_language_model2.h5 - Wytrenowany drugi model CNN.

American Sign Language Digits Dataset - Folder zawierający dane treningowe i testowe.

Wymagania

Aby uruchomić projekt, należy mieć zainstalowane:

Python (>=3.8)

Biblioteki:

TensorFlow

Keras

NumPy

Matplotlib

Scikit-learn

Możesz zainstalować wymagane biblioteki za pomocą poniższego polecenia:

pip install tensorflow keras numpy matplotlib scikit-learn

Dane wejściowe

Dane pochodzą z zestawu American Sign Language Digits Dataset, który zawiera:

10 folderów odpowiadających cyfrom od 0 do 9.

Każdy folder zawiera 500 obrazów przedstawiających daną cyfrę.

Struktura folderu danych:

American Sign Language Digits Dataset/
├── 0/
├── 1/
├── 2/
...
├── 9/

Instrukcja działania

1. Przygotowanie danych

Upewnij się, że folder American Sign Language Digits Dataset znajduje się w tej samej lokalizacji co skrypty.

Dane mogą być automatycznie ładowane i przetwarzane przez skrypty.

2. Uruchomienie modelu 1

Model 1 można uruchomić za pomocą skryptu Klasyfikacja_CNN.py:

python Klasyfikacja_CNN.py

Skrypt:

Wczytuje dane z folderu datasetu.

Trenuje model CNN.

Zapisuje wytrenowany model do pliku sign_language_model.h5.

3. Uruchomienie modelu 2

Model 2 można uruchomić za pomocą skryptu Klasyfikacja_CNN_modified.py:

python Klasyfikacja_CNN_modified.py

Podobnie jak model 1, ten skrypt:

Wczytuje dane z folderu datasetu.

Trenuje alternatywną wersję modelu CNN.

Zapisuje wytrenowany model do pliku sign_language_model2.h5.

Wyniki

Oba modele generują wyniki dokładności klasyfikacji na danych testowych.

Wyniki są wyświetlane na wykresach oraz w konsoli.

Uwaga

Przed uruchomieniem upewnij się, że wszystkie pliki są w odpowiednich lokalizacjach i że wymagane biblioteki są poprawnie zainstalowane.