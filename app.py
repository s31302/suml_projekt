import os
import gdown
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Airbnb EDA", layout="wide")
sns.set_style("whitegrid")
st.markdown(
    """
    <style>
        .stAppDeployButton, .stDeployButton { display: none; }
        #MainMenu { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data(show_spinner="Wczytywanie danych...")
def load_from_gdrive(file_id: str) -> pd.DataFrame:
    output = "airbnb_train.csv"
    if not os.path.exists(output):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output, quiet=False)
    return pd.read_csv(output)

@st.cache_data(show_spinner="Wczytywanie danych...")
def load_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner="Wczytywanie pliku...")
def load_from_upload(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)

FEATURES_TO_KEEP = [
    "room_type", "city", "accommodates", "bathrooms", "bedrooms", "beds",
    "latitude", "longitude", "host_is_superhost", "host_identity_verified",
    "has_availability", "availability_30", "availability_60", "availability_90",
    "availability_365", "number_of_reviews", "number_of_reviews_ltm",
    "number_of_reviews_l30d", "review_scores_rating", "reviews_per_month",
    "price",
]


@st.cache_data
def preprocess_data(df: pd.DataFrame, apply_clean: bool) -> pd.DataFrame:
    df = df.copy()

    if apply_clean:
        df = df[FEATURES_TO_KEEP].copy()

        low = df["price"].quantile(0.01)
        high = df["price"].quantile(0.95)
        df = df[(df["price"] > low) & (df["price"] <= high)]

        num_cols = df.select_dtypes(include=["int64", "float64"]).columns
        cat_cols = df.select_dtypes(include=["object", "bool"]).columns

        for col in num_cols:
            df[col] = df[col].fillna(df[col].median())
        for col in cat_cols:
            df[col] = df[col].fillna(df[col].mode()[0])

    df["beds_per_bedroom"] = df["beds"] / (df["bedrooms"] + 0.1)
    df["has_reviews"] = df["number_of_reviews"] > 0
    return df


# --- PANEL BOCZNY: ŹRÓDŁO DANYCH ---
st.sidebar.header("1. Źródło danych")
source = st.sidebar.radio(
    "Skąd pobrać airbnb_train.csv?",
    ["Google Drive (ID pliku)", "Ścieżka lokalna / Colab", "Wgraj plik CSV"],
)

#1xqsfMebm22S27J7MW_bYxD2ZRBidTM5F
housing = None
if source == "Google Drive (ID pliku)":
    file_id = st.sidebar.text_input("ID pliku z Google Drive -> \n1xqsfMebm22S27J7MW_bYxD2ZRBidTM5F", placeholder="np. 1AbC2dEf3GhI...")
    if file_id:
        housing = load_from_gdrive(file_id.strip())
elif source == "Ścieżka lokalna / Colab":
    path = st.sidebar.text_input("Ścieżka do pliku", value="/content/drive/MyDrive/data/airbnb_train.csv")
    if path and os.path.exists(path):
        housing = load_from_path(path)
    elif path:
        st.sidebar.warning("Nie znaleziono pliku pod tą ścieżką.")
else:
    uploaded = st.sidebar.file_uploader("Wybierz plik CSV", type=["csv"])
    if uploaded:
        housing = load_from_upload(uploaded)

st.title("Analiza ofert Airbnb")

if housing is None:
    st.info("Wybierz źródło danych w panelu bocznym po lewej, aby załadować zbiór.")
    st.stop()

st.sidebar.header("2. Preprocessing")
use_clean = st.sidebar.checkbox("Pokaż dane po preprocessingu (bez braków i outlierów)", value=True)
housing = preprocess_data(housing, apply_clean=use_clean)

# --- PANEL BOCZNY: FILTRY ---
st.sidebar.header("2. Filtry")
min_p, max_p = float(housing["price"].min()), float(housing["price"].max())
price_range = st.sidebar.slider("Zakres cen", min_p, max_p, (min_p, max_p))

all_rooms = sorted(housing["room_type"].dropna().unique())
selected_rooms = st.sidebar.multiselect("Typ pokoju", all_rooms, default=all_rooms)

all_cities = sorted(housing["city"].dropna().unique())
selected_cities = st.sidebar.multiselect("Miasta (puste = wszystkie)", all_cities, default=[])

filtered = housing[
    (housing["price"].between(price_range[0], price_range[1]))
    & (housing["room_type"].isin(selected_rooms))
]
if selected_cities:
    filtered = filtered[filtered["city"].isin(selected_cities)]

if filtered.empty:
    st.warning("Brak danych dla wybranych filtrów.")
    st.stop()

# --- KPI NA GÓRZE ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Liczba ofert", f"{len(filtered):,}".replace(",", " "))
c2.metric("Liczba kolumn", filtered.shape[1])
c3.metric("Mediana ceny", f"{filtered['price'].median():.2f}")
c4.metric("Liczba miast", filtered["city"].nunique())

# --- ZAKŁADKI ---
tab1, tab2, tab3, tab4 = st.tabs([
    "Przegląd i braki",
    "Ceny i typy pokoi",
    "Miasta",
    "Nowe cechy i korelacje",
])

with tab1: #Przegląd i braki
    st.subheader("Podgląd danych")
    st.dataframe(filtered.head(50), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Top 15 kolumn z brakami")
        missing = filtered.isnull().sum().sort_values(ascending=False).head(15)
        st.dataframe(missing.rename("Braki"), use_container_width=True)
    with col_b:
        st.subheader("Statystyki opisowe (numeryczne)")
        st.dataframe(filtered.describe().T, use_container_width=True)

with tab2:
    st.subheader("Rozkład cen")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(filtered["price"], bins=50, kde=True, ax=ax)
    ax.set_xlabel("Cena")
    ax.set_ylabel("Częstość")
    st.pyplot(fig)
    plt.close(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Liczba ofert wg typu pokoju")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=filtered, x="room_type", order=filtered["room_type"].value_counts().index, ax=ax)
        ax.tick_params(axis="x", rotation=30)
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Wpływ typu pokoju na cenę")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=filtered, x="room_type", y="price", ax=ax)
        ax.tick_params(axis="x", rotation=30)
        st.pyplot(fig)
        plt.close(fig)

with tab3:
    top_n = st.slider("Ile najpopularniejszych miast pokazać?", 5, 25, 10)
    top_cities = filtered["city"].value_counts().head(top_n).index
    df_cities = filtered[filtered["city"].isin(top_cities)]

    st.subheader(f"Rozkład cen w top {top_n} miastach")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df_cities, x="city", y="price", order=top_cities, ax=ax)
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)
    plt.close(fig)

with tab4:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Czy oferta ma opinie?")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(data=filtered, x="has_reviews", ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Podgląd nowych cech")
        st.dataframe(
            filtered[["beds", "bedrooms", "beds_per_bedroom", "number_of_reviews", "has_reviews"]].head(20),
            use_container_width=True,
        )

    st.subheader("Macierz korelacji cech numerycznych")
    num_cols = filtered.select_dtypes(include="number").columns.tolist()
    default_corr = [
        c for c in ["price", "accommodates", "bathrooms", "bedrooms", "beds", "beds_per_bedroom", "number_of_reviews", "review_scores_rating"]
        if c in num_cols
    ]
    chosen_cols = st.multiselect("Kolumny do korelacji", num_cols, default=default_corr)

    if len(chosen_cols) >= 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(filtered[chosen_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        plt.close(fig)

#tuta jeszcze dodoac interfejs do przewidywania ceny noclegua a do tego musze miec plk