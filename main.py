import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import requests
import yfinance as yf

# ==========================================
# CONFIG & SCHWELLENWERTE
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILENAME = os.path.join(SCRIPT_DIR, "SPNASDAQ100_RESULTS.csv")

BATCH_SIZE = 15
MAX_WORKERS = 2

# FILTER-EINSTELLUNGEN
USE_5Y_PROFIT_FILTER = True
REQUIRED_YEARS = 5       
MIN_AVAILABLE_YEARS = 4   

USE_FCF_FILTER = True
FCF_MIN_YEARS = 4

USE_QUALITY_FILTER = False
SCENARIO_HORIZON_YEARS = 2

csv_lock = threading.Lock()


def fetch_combined_tickers():
    """Lädt und kombiniert die Ticker-Listen von S&P 500 und Nasdaq 100."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML,"
            " like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    all_tickers = []

    # 1. S&P 500 laden
    try:
        url_sp = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        df_sp = pd.read_html(url_sp, storage_options=headers)[0]
        sp_tickers = df_sp["Symbol"].str.replace(".", "-", regex=False).tolist()
        all_tickers.extend(sp_tickers)
        print(f"-> {len(sp_tickers)} S&P 500 Ticker geladen.")
    except Exception as e:
        print(f"Fehler beim Laden der S&P-500-Liste: {e}")

    # 2. Nasdaq 100 laden
    ndx_tickers = []
    try:
        url_ndx = "https://www.slickcharts.com/nasdaq100"
        df_list = pd.read_html(url_ndx, storage_options=headers)
        for df in df_list:
            if "Symbol" in df.columns:
                ndx_tickers = (
                    df["Symbol"]
                    .dropna()
                    .astype(str)
                    .str.replace(".", "-", regex=False)
                    .tolist()
                )
                if len(ndx_tickers) > 50:
                    break
    except Exception as e:
        print(f"Hinweis bei Slickcharts: {e}")

    if len(ndx_tickers) < 50:
        try:
            api_url = "https://api.nasdaq.com/api/quote/list-type/nasdaq100"
            response = requests.get(api_url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                rows = data.get("data", {}).get("data", {}).get("rows", [])
                ndx_tickers = [
                    row["symbol"].strip().replace(".", "-")
                    for row in rows
                    if "symbol" in row
                ]
        except Exception as e:
            print(f"Hinweis bei Nasdaq API: {e}")

    if ndx_tickers:
        all_tickers.extend(ndx_tickers)
        print(f"-> {len(ndx_tickers)} Nasdaq-100 Ticker erfolgreich geladen.")

    unique_tickers = list(set(all_tickers))
    print(f"-> Insgesamt {len(unique_tickers)} einzigartige Unternehmen nach Bereinigung.")
    return unique_tickers


def check_5y_profits(t_obj):
    try:
        income = t_obj.income_stmt
        if income is None or income.empty or "Net Income" not in income.index:
            return False, "nicht pruefbar"
        net_income = income.loc["Net Income"].dropna()
        if len(net_income) < MIN_AVAILABLE_YEARS:
            return False, "wenige Jahre"
        last_years = net_income.sort_index(ascending=False).head(REQUIRED_YEARS)
        for _, value in last_years.items():
            if value <= 0:
                return False, "Verlustjahr"
        return True, "profitabel"
    except Exception:
        return False, "Fehler"


def check_multi_year_fcf(t_obj, sector):
    if sector == "Financial Services":
        return True, "N/A"
    try:
        cf = t_obj.cashflow
        if cf is None or cf.empty or "Free Cash Flow" not in cf.index:
            return False, "keine CF-Daten"
        fcf = cf.loc["Free Cash Flow"].dropna()
        if len(fcf) < FCF_MIN_YEARS:
            return False, "wenige FCF-Jahre"
        if any(v <= 0 for v in fcf.values):
            return False, "negativer FCF"
        return True, "positiver FCF"
    except Exception:
        return False, "Fehler"


def calculate_scenario_returns(info, current_price):
    forward_eps = info.get("forwardEps")
    if not forward_eps or forward_eps <= 0 or not current_price:
        return {"KGV25_Rendite_pa_%": 0.0}
    growth = info.get("earningsGrowth") or info.get("revenueGrowth") or 0.0
    growth = min(max(growth, 0.0), 1.0)
    eps_2y = forward_eps * (1 + growth)
    model_price = eps_2y * 25
    ann_return = (model_price / current_price) ** (1 / SCENARIO_HORIZON_YEARS) - 1
    return {"KGV25_Rendite_pa_%": round(ann_return * 100, 1)}


def process_batch(ticker_batch):
    results = []
    for attempt in range(3):
        try:
            time.sleep(0.8)
            tickers_obj = yf.Tickers(" ".join(ticker_batch))
            for ticker in ticker_batch:
                try:
                    t_obj = tickers_obj.tickers.get(ticker)
                    if not t_obj: continue
                    info = getattr(t_obj, "info", None)
                    if not info or not isinstance(info, dict): continue

                    current_price = info.get("currentPrice", info.get("regularMarketPrice", 0))
                    market_cap = info.get("marketCap", 0)
                    if not market_cap or not current_price: continue

                    sector = info.get("sector", "")
                    
                    if USE_5Y_PROFIT_FILTER and not check_5y_profits(t_obj)[0]: continue
                    if USE_FCF_FILTER and not check_multi_year_fcf(t_obj, sector)[0]: continue

                    rev_growth = info.get("revenueGrowth", 0.0) or 0.0
                    profit_margin = info.get("profitMargins", 0.0) or 0.0
                    forward_pe = info.get("forwardPE", 0.0) or 0.0
                    roe = info.get("returnOnEquity") or 0.0
                    payout = info.get("payoutRatio")
                    dte = info.get("debtToEquity")

                    quality_score = 0
                    if roe >= 0.15: quality_score += 1
                    if info.get("freeCashflow", 0) > 0: quality_score += 1
                    if profit_margin >= 0.15: quality_score += 1
                    if payout is not None and 0 < payout <= 0.60: quality_score += 1
                    if sector != "Financial Services" and dte is not None and dte <= 100: quality_score += 1
                    if rev_growth >= 0.08: quality_score += 1

                    scen = calculate_scenario_returns(info, current_price)

                    results.append({
                        "Ticker": ticker,
                        "Name": info.get("shortName", ticker),
                        "Sector": sector,
                        "Price": current_price,
                        "RevenueGrowth_%": round(rev_growth * 100, 2),
                        "NetMargin_%": round(profit_margin * 100, 2),
                        "Forward_PE": round(forward_pe, 1) if forward_pe else 0,
                        "ROE_%": round(roe * 100, 1),
                        "FreeCashflow_Mio": round(info.get("freeCashflow", 0) / 1_000_000, 0) if info.get("freeCashflow") else 0,
                        "Quality_Score": quality_score,
                        "KGV25_Rendite_pa_%": scen["KGV25_Rendite_pa_%"],
                        "Sektor_Prio": 1
                    })
                except Exception:
                    continue
            break
        except Exception:
            time.sleep(2)

    if results:
        df_batch = pd.DataFrame(results)
        with csv_lock:
            header_flag = not os.path.exists(CSV_FILENAME)
            df_batch.to_csv(CSV_FILENAME, mode="a", header=header_flag, index=False)

    return len(results)


def send_telegram_message(message):
    """Sendet die Nachricht via Telegram Bot."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("Telegram-Zugangsdaten nicht gefunden (Secrets fehlen).")
        return
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            print("-> Telegram-Nachricht erfolgreich gesendet!")
        else:
            print(f"-> Fehler bei Telegram: {response.text}")
    except Exception as e:
        print(f"-> Telegram-Ausnahme: {e}")


def send_whatsapp_message(message):
    """Sendet die Nachricht via CallMeBot / WhatsApp."""
    phone = os.getenv("WHATSAPP_PHONE")
    apikey = os.getenv("WHATSAPP_API_KEY")
    if not phone or not apikey:
        print("WhatsApp-Zugangsdaten nicht gefunden (Secrets fehlen).")
        return

    url = f"https://api.callmebot.com/whatsapp.php?phone={phone}&text={requests.utils.quote(message)}&apikey={apikey}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            print("-> WhatsApp-Nachricht erfolgreich gesendet!")
        else:
            print(f"-> Fehler bei WhatsApp: {response.text}")
    except Exception as e:
        print(f"-> WhatsApp-Ausnahme: {e}")


def evaluate_and_report():
    """Analysiert die Live-Daten, ermittelt Sieger & Zweitsieger, gibt Report aus und sendet Benachrichtigungen."""
    if not os.path.exists(CSV_FILENAME):
        print("Keine Ergebnisse gefunden.")
        return

    df = pd.read_csv(CSV_FILENAME)
    if df.empty:
        print("Datensatz ist leer.")
        return

    if "Ticker" in df.columns:
        df = df[df["Ticker"] != "GOOG"]

    df["Composite_Score"] = (
        (df["Quality_Score"] * 25.0) +
        (df["RevenueGrowth_%"] * 0.5) +
        (df["NetMargin_%"] * 0.3) +
        (df["ROE_%"] * 0.2) -
        (df["Forward_PE"] * 0.4)
    )

    ranked_df = df.sort_values(by="Composite_Score", ascending=False)
    winner = ranked_df.iloc[0]
    runner_up = ranked_df.iloc[1] if len(ranked_df) > 1 else None

    report = f"🏆 *AKTIE DES TAGES (SCREENER-SIEGER)* 🏆\n\n"
    report += f"🥇 *1. Platz: {winner['Ticker']}* ({winner['Name']})\n"
    report += f"• Quality Score: {int(winner['Quality_Score'])}/6\n"
    report += f"• Umsatzwachstum: {winner['RevenueGrowth_%']:.1f}%\n"
    report += f"• Nettomarge: {winner['NetMargin_%']:.2f}%\n"
    report += f"• ROE: {winner['ROE_%']:.1f}%\n"
    report += f"• Forward-PE: {winner['Forward_PE']}\n"
    report += f"• Free Cashflow: {int(winner['FreeCashflow_Mio']):,} Mio. USD\n\n"

    if runner_up is not None:
        report += f"🥈 *2. Platz (Vergleich): {runner_up['Ticker']}* ({runner_up['Name']})\n"
        report += f"• Quality Score: {int(runner_up['Quality_Score'])}/6\n"
        report += f"• Umsatzwachstum: {runner_up['RevenueGrowth_%']:.1f}%\n"
        report += f"• Forward-PE: {runner_up['Forward_PE']}\n\n"

        report += f"💡 *Begründung:*\n"
        report += f"{winner['Ticker']} setzt sich gegen {runner_up['Ticker']} durch, da {winner['Ticker']} "
        report += f"ein überlegenes Umsatzwachstum ({winner['RevenueGrowth_%']:.1f}% vs {runner_up['RevenueGrowth_%']:.1f}%), "
        report += f"stärkere Nettomargen ({winner['NetMargin_%']:.1f}%) und eine herausragende Gesamteffizienz aufweist."

    print("\n" + "="*40)
    print(report)
    print("="*40 + "\n")

    # Benachrichtigungen ausführen
    print("Sende Benachrichtigungen...")
    send_telegram_message(report)
    send_whatsapp_message(report)


def main():
    print("1. Lade kombinierte Ticker-Liste (S&P 500 & Nasdaq 100)...")
    tickers = fetch_combined_tickers()
    total_tickers = len(tickers)

    if total_tickers == 0:
        print("Fehler: Keine Ticker geladen.")
        return

    if os.path.exists(CSV_FILENAME):
        try:
            os.remove(CSV_FILENAME)
        except PermissionError:
            pass

    batches = [tickers[i : i + BATCH_SIZE] for i in range(0, total_tickers, BATCH_SIZE)]
    print(f"2. Starte Live-Analyse von {len(batches)} Batches ({total_tickers} Titel)...")

    start_time = time.time()
    processed_batches = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_batch, batch) for batch in batches]
        for future in as_completed(futures):
            processed_batches += 1
            if processed_batches % 10 == 0 or processed_batches == len(batches):
                elapsed = time.time() - start_time
                avg = elapsed / processed_batches
                rem = (len(batches) - processed_batches) * avg
                print(f"Fortschritt: {processed_batches}/{len(batches)} | Restzeit: ca. {int(rem // 60)}m {int(rem % 60)}s")

    print("\n3. Ermittle Tagessieger, Vergleiche und versende Benachrichtigungen...")
    evaluate_and_report()
    print("Fertig!")


if __name__ == "__main__":
    main()
