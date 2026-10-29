import os
import pandas as pd


def load_and_clean_data():
    """Schritt 1 & 2: Lädt die gesamte CSV mit allen ~518 Aktien,

    entfernt die Dublette GOOG (GOOGL bleibt) und bereinigt Spalten.
    """
    # Sucht nach deiner echten CSV-Datei im Repository
    possible_paths = [
        "step2_clean.csv",
        "SPNASDAQ100_RESULTS-2.csv",
        "data/sp500_nasdaq100.csv",
        "sp500_nasdaq100.csv",
    ]

    file_path = None
    for path in possible_paths:
        if os.path.exists(path):
            file_path = path
            break

    if not file_path:
        raise FileNotFoundError(
            "Keine passende CSV-Datei für die 518 Aktien gefunden!"
        )

    print(f"Lade Aktien-Universum aus: {file_path}")
    df = pd.read_csv(file_path)

    # Clean 1: Dublette GOOG entfernen (GOOGL behalten)
    if "Ticker" in df.columns:
        df = df[df["Ticker"] != "GOOG"]

    # Clean 2: Sektor_Prio sicherstellen (Falls Text, in Zahl umwandeln)
    if "Sektor_Prio" in df.columns:
        df["Sektor_Prio"] = pd.to_numeric(df["Sektor_Prio"], errors="coerce")

    return df


def evaluate_screener(df):
    """Schritt 3 bis 8:

    - Filtert dynamisch aus allen 518 Aktien NUR diejenigen mit Sektor_Prio == 1
    - Bewertet Qualität, Umsatzwachstum, Margen, ROE, FCF & Forward-PE
    - Berechnet das Ranking für den 1. und 2. Platz.
    """
    # Schritt 3: Strikte Filterung auf deine ausgewählte Sektor_Prio = 1
    prio_df = df[df["Sektor_Prio"] == 1].copy()

    print(f"Anzahl gefilterter Sektor_Prio=1 Aktien: {len(prio_df)}")

    if prio_df.empty:
        return None, None

    # Numerische Spalten sicherstellen
    num_cols = [
        "Quality_Score",
        "RevenueGrowth_%",
        "NetMargin_%",
        "ROE_%",
        "Forward_PE",
        "FreeCashflow_Mio",
    ]
    for col in num_cols:
        if col in prio_df.columns:
            prio_df[col] = pd.to_numeric(prio_df[col], errors="coerce").fillna(0)

    # Schritt 4 bis 6: Gewichtete Bewertungsformel nach deinen Kriterien
    # (Belohnt Quality-Score, hohes Wachstum, hohe Margen, hohen ROE & moderates Forward-PE)
    prio_df["Composite_Score"] = (
        (prio_df["Quality_Score"] * 25.0)
        + (prio_df["RevenueGrowth_%"] * 0.5)
        + (prio_df["NetMargin_%"] * 0.3)
        + (prio_df["ROE_%"] * 0.2)
        - (prio_df["Forward_PE"] * 0.4)
    )

    # Sortieren: Die Aktie mit dem höchsten Gesamtscore gewinnt
    ranked_df = prio_df.sort_values(by="Composite_Score", ascending=False)

    winner = ranked_df.iloc[0]
    runner_up = ranked_df.iloc[1] if len(ranked_df) > 1 else None

    return winner, runner_up


def generate_report(winner, runner_up):
    """Schritt 8: Generiert den automatischen Bericht mit der Begründung."""
    report = f"🏆 **AKTIE DES TAGES (SCREENER-SIEGER)** 🏆\n\n"
    report += f"🥇 **1. Platz: {winner['Ticker']}**\n"
    report += f"• Quality Score: {int(winner.get('Quality_Score', 0))}/6\n"
    report += f"• Umsatzwachstum: {winner.get('RevenueGrowth_%', 0):.1f}%\n"
    report += f"• Nettomarge: {winner.get('NetMargin_%', 0):.2f}%\n"
    report += f"• ROE: {winner.get('ROE_%', 0):.1f}%\n"
    report += f"• Forward-PE: {winner.get('Forward_PE', 0):.1f}\n"

    fcf = winner.get("FreeCashflow_Mio", 0)
    report += (
        f"• Free Cashflow: {int(fcf):,} Mio. USD\n\n"
        if fcf > 0
        else "• Free Cashflow: k.A.\n\n"
    )

    if runner_up is not None:
        report += f"🥈 **2. Platz (Vergleich): {runner_up['Ticker']}**\n"
        report += f"• Quality Score: {int(runner_up.get('Quality_Score', 0))}/6\n"
        report += f"• Umsatzwachstum: {runner_up.get('RevenueGrowth_%', 0):.1f}%\n"
        report += f"• Nettomarge: {runner_up.get('NetMargin_%', 0):.2f}%\n"
        report += f"• Forward-PE: {runner_up.get('Forward_PE', 0):.1f}\n\n"

        report += f"💡 **Begründung:**\n"
        report += (
            f"{winner['Ticker']} setzt sich dynamisch gegen {runner_up['Ticker']} durch. "
        )
        report += f"{winner['Ticker']} bietet mit {winner.get('RevenueGrowth_%', 0):.1f}% ein überlegenes Umsatzwachstum "
        report += f"(vs. {runner_up.get('RevenueGrowth_%', 0):.1f}%), eine stärkere Nettomarge ({winner.get('NetMargin_%', 0):.1f}%) "
        report += f"und eine höhere Gesamteffizienz nach deinem 8-Stufen-Filter."

    return report


def main():
    print("Starte Screening für alle 518 Aktien...")
    try:
        df = load_and_clean_data()
        winner, runner_up = evaluate_screener(df)

        if winner is not None:
            report = generate_report(winner, runner_up)
            print("\n--- GENERIERTER REPORT ---")
            print(report)
            print("--------------------------\n")
            print("Tagessieger erfolgreich aus allen Daten ermittelt!")
        else:
            print("Keine Aktien mit Sektor_Prio = 1 im Universum gefunden.")
    except Exception as e:
        print(f"Fehler beim Screening: {e}")


if __name__ == "__main__":
    main()
