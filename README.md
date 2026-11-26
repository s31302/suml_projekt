# Analiza cen ofert Airbnb

Projekt analizuje dane ofert Airbnb i buduje modele regresji przewidujące cenę noclegu na podstawie cech mieszkania, lokalizacji i informacji o gospodarzu.

## Zakres projektu

- kontrola cen, identyfikatorów ofert i braków danych,
- porównanie cen według typu pokoju i miasta,
- inżynieria i analiza cech,
- preprocessing danych numerycznych i kategorycznych,
- trenowanie i porównanie modeli:
  - model bazowy przewidujący medianę,
  - regresja liniowa,
  - Random Forest,
  - XGBoost na logarytmie ceny,
  - XGBoost z ważonym błędem bezwzględnym.

Wyniki MAPE, MAE i R² są liczone na tym samym losowo wydzielonym zbiorze testowym (20% ofert, ziarno 42). Trening i ocena obejmują także najtańsze i najdroższe oferty. W treningu pominięto szacowany przychód i obłożenie, ponieważ ich iloraz odtwarza cenę.

Regresja liniowa, Random Forest i oba warianty XGBoost korzystają z tych samych 20 cech wejściowych. Model bazowy przewiduje medianę cen ze zbioru treningowego. Pełne porównanie znajduje się w sekcji 8 notebooka. Ostatnia komórka zapisuje metryki i wspólną listę cech do [wyniki_modeli.txt](wyniki_modeli.txt).

`airbnb_price_model.pkl` oraz `airbnb_train.csv` w google drive (ze wzgledu na wielkosc) -> https://drive.google.com/drive/u/0/folders/1sAYIrRYvIIAzBHzRSFXEY_8CqgRNsVDB

## Dane

Lokalnie notebook szuka pliku `data/airbnb_train.csv`, a następnie `airbnb_train.csv` obok notebooka. W Colab może go też odczytać z Google Drive.

```text
/content/drive/MyDrive/data/airbnb_train.csv
```

## Uruchomienie

### Lokalnie

1. Utwórz i aktywuj środowisko wirtualne:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Zainstaluj zależności:

   ```bash
   python -m pip install -r requirements.txt
   ```

   Jeśli import XGBoost na macOS zgłasza brak `libomp`, zainstaluj tę bibliotekę przez `brew install libomp`. [Instrukcja instalacji XGBoost](https://xgboost.readthedocs.io/en/stable/install.html)

3. Umieść plik w `data/airbnb_train.csv` i uruchom notebook:

   ```bash
   jupyter notebook suml.ipynb
   ```

### Google Colab

1. Otwórz `suml.ipynb` w Google Colab.
2. W razie potrzeby zainstaluj XGBoost poleceniem `%pip install "xgboost>=3.1"`.
3. Umieść `airbnb_train.csv` we wskazanym katalogu Google Drive.
4. Uruchom wszystkie komórki i autoryzuj dostęp do Google Drive, gdy notebook o to poprosi.

## Autorzy

- Małgosia Świderek
- Jakub Kowalczyk
- Mikołaj Stępowski
