import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

np.random.seed(42)

n = 500

klienci = [
    "Anna Kowalska",
    "Jan Nowak",
    "Anna Nowak",
    "Piotr Wiśniewski",
    "katarzyna lewandowska",
    "Tomasz Zieliński ",
    "Marta Wójcik",
    "anna kowalska ",
    "Krzysztof Kamiński",
    " Magdalena Dąbrowska"
]

produkty = [
    "Laptop",
    "Mysz",
    "Klawiatura",
    "Monitor",
    "laptop",
    "MYSZ",
    "Słuchawki",
    "Pendrive",
    "monitor",
    "Webcam"
]

kategorie = [
    "Elektronika",
    "elektronika",
    "ELEKTRONIKA",
    "Akcesoria",
    "akcesoria",
    "Akcesoria "
]

miasta = [
    "Warszawa",
    "Kraków",
    "warszawa",
    "Gdańsk",
    "Wrocław",
    "Poznań",
    "Łódź ",
    " Warszawa",
    "kraków"
]

start_date = datetime(2025, 1, 1)

daty_iso = [
    (start_date + timedelta(days=int(d))).strftime("%Y-%m-%d")
    for d in np.random.randint(0, 300, n // 2)
]

daty_pl = [
    (start_date + timedelta(days=int(d))).strftime("%d.%m.%Y")
    for d in np.random.randint(0, 300, n // 2)
]

daty = daty_iso + daty_pl
np.random.shuffle(daty)

df = pd.DataFrame({
    "order_id": range(1001, 1001 + n),
    "klient": np.random.choice(klienci, n),
    "produkt": np.random.choice(produkty, n),
    "kategoria": np.random.choice(kategorie, n),
    "miasto": np.random.choice(miasta, n),
    "ilosc": np.random.choice(
        [1, 2, 3, 5, -1, 0],
        n,
        p=[0.5, 0.2, 0.15, 0.1, 0.025, 0.025]
    ),
    "cena_jednostkowa": np.random.choice(
        ["199.99", "299,99", "1 499.00", "89.50", "2999", "399.00 zł", None, "abc"],
        n
    ),
    "data_zamowienia": daty,
    "email": np.random.choice(
        [
            "anna@gmail.com",
            "JAN@WP.PL",
            "piotr.w@onet",
            "marta@gmail.com",
            "tomasz@interia.pl",
            None,
            "krzysztof.k@gmail.com",
            "brak"
        ],
        n
    )
})

# Braki danych
for col in ["miasto", "kategoria", "data_zamowienia"]:
    df.loc[df.sample(frac=0.05, random_state=1).index, col] = np.nan

# Duplikaty
df = pd.concat([df, df.sample(20, random_state=2)], ignore_index=True)

# Zapis brudnych danych
df.to_csv("zamowienia_messy.csv", index=False)

print("Wygenerowano plik 'zamowienia_messy.csv'")
print(f"Liczba wierszy w brudnym zbiorze: {len(df)}")

print("CZĘŚĆ 1 — EKSPLORACJA")

print("\nStan:")
print(df.shape)

print("\nInfo:")
df.info()

print("\nOpis dla kolumn liczbowych:")
print(df.describe())

print("\nOpis dla wszystkich kolumn:")
print(df.describe(include="all"))

print("\nLiczba braków danych:")
print(df.isnull().sum())

kolumny_kategoryczne = [
    "klient",
    "produkt",
    "kategoria",
    "miasto",
    "cena_jednostkowa",
    "email"
]

for col in kolumny_kategoryczne:
    print(f"\nValue counts dla kolumny: {col}")
    print(df[col].value_counts(dropna=False))


"""
Zidentyfikowane problemy z jakością danych:

1. W zbiorze występują duplikaty całych wierszy.
2. W kolumnach tekstowych są niepotrzebne spacje, np. '  Jan Nowak', 'Łódź ', ' Warszawa'.
3. Występuje niespójna wielkość liter, np. 'Laptop' i 'laptop', 'MYSZ', 'warszawa', 'WROCŁAW'.
4. Występują braki danych w kolumnach: miasto, kategoria, data_zamowienia, cena_jednostkowa i email.
5. Daty mają dwa różne formaty: YYYY-MM-DD oraz DD.MM.YYYY.
6. Ceny są zapisane jako tekst i mają różne formaty, np. '299,99', '1 499.00', '399.00 zł'.
7. W kolumnie cena_jednostkowa występują wartości niemożliwe do konwersji, np. 'abc'.
8. W kolumnie ilosc występują błędne wartości: 0 oraz -1.
9. W kolumnie email występują niepoprawne adresy, np. 'piotr.w@onet' oraz 'brak'.
"""

print("CZĘŚĆ 2 — CZYSZCZENIE")

df_clean = df.copy()

df_clean = df_clean.drop_duplicates()

# Standaryzacja kolumn tekstowych
df_clean["klient"] = df_clean["klient"].astype("string").str.strip().str.lower().str.title()
df_clean["produkt"] = df_clean["produkt"].astype("string").str.strip().str.lower().str.title()
df_clean["kategoria"] = df_clean["kategoria"].astype("string").str.strip().str.lower()
df_clean["miasto"] = df_clean["miasto"].astype("string").str.strip().str.lower().str.title()

# Konwersja daty
df_clean["data_zamowienia"] = pd.to_datetime(
    df_clean["data_zamowienia"],
    format="mixed",
    dayfirst=True,
    errors="coerce"
)

def wyczysc_cene(wartosc):
    if pd.isna(wartosc):
        return np.nan

    wartosc = str(wartosc)
    wartosc = wartosc.strip()
    wartosc = wartosc.lower()
    wartosc = wartosc.replace("zł", "")
    wartosc = wartosc.replace(" ", "")
    wartosc = wartosc.replace(",", ".")

    return pd.to_numeric(wartosc, errors="coerce")


df_clean["cena_jednostkowa"] = df_clean["cena_jednostkowa"].apply(wyczysc_cene)

df_clean = df_clean.dropna(subset=["cena_jednostkowa", "data_zamowienia"])

# Uzupełnienie braków w pozostałych kolumnach
df_clean["miasto"] = df_clean["miasto"].fillna("unknown")
df_clean["kategoria"] = df_clean["kategoria"].fillna("unknown")
df_clean["email"] = df_clean["email"].fillna("brak_emaila")

Usunięcie błędnych ilości
df_clean = df_clean[df_clean["ilosc"] > 0].copy()

print("\nStan po czyszczeniu:")
print(df_clean.shape)

print("\nBraki danych po czyszczeniu:")
print(df_clean.isnull().sum())

print("\nTypy danych po czyszczeniu:")
print(df_clean.dtypes)

print("CZĘŚĆ 3 — TRANSFORMACJE")

# Wartość zamówienia
df_clean["wartosc_zamowienia"] = df_clean["ilosc"] * df_clean["cena_jednostkowa"]

# Kolumny pochodne z daty
df_clean["rok"] = df_clean["data_zamowienia"].dt.year
df_clean["miesiac"] = df_clean["data_zamowienia"].dt.to_period("M").astype(str)
df_clean["nazwa_dnia"] = df_clean["data_zamowienia"].dt.day_name()

# Poprawność emaila
df_clean["email_poprawny"] = df_clean["email"].str.match(
    r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
    na=False
)

print("\nPierwsze wiersze po transformacjach:")
print(df_clean.head())

print("CZĘŚĆ 4 — ANALIZA SQL-STYLE")

wartosc_miesiac = (
    df_clean
    .groupby("miesiac", as_index=False)["wartosc_zamowienia"]
    .sum()
    .sort_values("miesiac")
)

print("\nŁączna wartość zamówień w każdym miesiącu:")
print(wartosc_miesiac)

top_5_klientow = (
    df_clean
    .groupby("klient", as_index=False)["wartosc_zamowienia"]
    .sum()
    .sort_values("wartosc_zamowienia", ascending=False)
    .head(5)
)

print("\nTop 5 klientów pod względem łącznej wartości zamówień:")
print(top_5_klientow)

srednia_kategoria = (
    df_clean
    .groupby("kategoria", as_index=False)["wartosc_zamowienia"]
    .mean()
    .sort_values("wartosc_zamowienia", ascending=False)
)

print("\nŚrednia wartość zamówienia w każdej kategorii:")
print(srednia_kategoria)

print("CZĘŚĆ 5 — WIZUALIZACJA")

plt.figure(figsize=(10, 5))
plt.bar(wartosc_miesiac["miesiac"], wartosc_miesiac["wartosc_zamowienia"])
plt.title("Łączna wartość zamówień w każdym miesiącu")
plt.xlabel("Miesiąc")
plt.ylabel("Łączna wartość zamówień")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("wartosc_zamowien_miesiac.png", dpi=150)
plt.close()

print("Zapisano wykres: wartosc_zamowien_miesiac.png")

print("\n" + "=" * 60)
print("CZĘŚĆ 6 — ZAPIS")
print("=" * 60)

df_clean.to_csv("zamowienia_clean.csv", index=False)

print("Zapisano oczyszczony plik: zamowienia_clean.csv")

print("\nProjekt zakończony poprawnie.")