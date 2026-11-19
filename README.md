# Analiza cen ofert Airbnb

Projekt analizuje dane ofert Airbnb i buduje modele regresji przewidujące cenę noclegu na podstawie cech mieszkania, lokalizacji i informacji o gospodarzu.

## Zakres projektu

- analiza rozkładu cen i braków danych,
- porównanie cen według typu pokoju i miasta,
- inżynieria i analiza cech,
- preprocessing danych numerycznych i kategorycznych,
- trenowanie i porównanie modeli:
  - Linear Regression,
  - Random Forest,
  - XGBoost.

## Dane

Notebook oczekuje pliku `airbnb_train.csv` w katalogu Google Drive:

```text
/content/drive/MyDrive/data/airbnb_train.csv
```

Po uruchomieniu notebooku należy autoryzować dostęp do Google Drive.

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

3. Umieść plik `airbnb_train.csv` obok `suml.ipynb` i uruchom notebook:

   ```bash
   jupyter notebook suml.ipynb
   ```

Jeśli notebook uruchamiany jest lokalnie, komórka wczytująca dane odczyta plik z bieżącego katalogu. Jeśli wczytanie z bieżącego katalogu się nie powiedzie, należy wskazać inną lokalną ścieżkę.

### Google Colab

1. Otwórz `suml.ipynb` w Google Colab.
2. Zamontuj Google Drive.
3. Umieść `airbnb_train.csv` we wskazanym katalogu.
4. Uruchom wszystkie komórki notatnika.

## Autorzy

- Małgosia Świderek
- Jakub Kowalczyk
- Mikołaj Stępowski
