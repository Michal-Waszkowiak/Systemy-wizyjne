# Systemy-wizyjne
Rozwiązanie zadań z laboratorium + projekt wykrywania tablic

# Projekt

## Temat: Wykrywanie znaków na tablicy rejestracyjnej

### 1. Opis zastosowanej metody:

#### 1.1. Wstępne Przetwarzanie Obrazu

- **Zmiana Rozmiaru**: Obraz jest skalowany do mniejszego rozmiaru.
- **Konwersja do Odcieni Szarości**: Obraz jest konwertowany do skali szarości.
- **Filtracja Gaussowska**: Zastosowanie filtru Gaussowskiego w celu wygładzenia obrazu.
- **Progowanie**: Wykorzystanie metody Otsu do binarnego progowania obrazu.
- **Detekcja Krawędzi**: Algorytm Canny’ego do detekcji krawędzi.
- **Operacje Morfologiczne**: Zamykanie (closing) i dylacja (dilate) w celu wzmocnienia krawędzi.

#### 1.2. Wykrywanie Konturów

- **Znajdowanie Konturów**: Wykrycie konturów na przetworzonym obrazie.
- **Sortowanie Konturów**: Sortowanie konturów według obszaru, wybór pięciu największych.

#### 1.3. Wybór Konturu Tablicy Rejestracyjnej

- **Filtrowanie Konturów**: Sprawdzanie proporcji boków (aspect ratio) i powierzchni konturu.
- **Wyznaczanie Przybliżonego Konturu**: Algorytm aproksymacji kształtu konturu (Douglas-Peucker).

#### 1.4. Transformacja Perspektywy

- Przekształcenie perspektywy do prostokąta o ustalonym rozmiarze (400x100).

#### 1.5. Segmentacja Znaków

- Progowanie obrazu i ponowne wykrycie konturów, filtrowanie konturów odpowiadających znakom.

#### 1.6. Predykcja Znaków

- **Wstępne Przetwarzanie Znaków**: Konwersja do skali szarości, filtrowanie Gaussowskie, progowanie, zmiana rozmiaru.
- **Klasyfikacja Znaków**: Model SVM (Support Vector Machine) do klasyfikacji wykrytych znaków.

### 2. Model Uczenia Maszynowego

#### 2.1. Model SVC

- **Typ Modelu**: SVM z jądrem liniowym.
- **Trening Modelu**: Model trenowany na zbiorze zdjęć znaków zapisanych w katalogu train. Zdjęcia są konwertowane do skali szarości, zmieniane na rozmiar (70x100) i spłaszczone do wektora.

#### 2.2. Wstępne Przetwarzanie Danych

- **Konwersja i Normalizacja**: Zdjęcia znaków są przetwarzane do postaci wektorów wartości pikseli.
- **Etykietowanie**: Zdjęcia są oznaczone odpowiednimi etykietami znaków (0-9, A-Z).

### 3. Schemat blokowy
