import os
import time
import requests
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# CONFIG & SCHWELLENWERTE
# ==========================================
BATCH_SIZE = 15
MAX_WORKERS = 2

USE_5Y_PROFIT_FILTER = True
REQUIRED_YEARS = 5
MIN_AVAILABLE_YEARS = 4

USE_FCF_FILTER = True
FCF_MIN_YEARS = 4

USE_QUALITY_FILTER = False
SCENARIO_HORIZON_YEARS = 2

# ==========================================
# BENACHRICHTIGUNGEN (Telegram & WhatsApp)
# ==========================================
def send_telegram(message):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    requests.post(url, json=payload)

def send_whatsapp_callmebot(message):
    phone = os.getenv("WHATSAPP_PHONE")
    apikey = os.getenv("WHATSAPP_API_KEY")
    if not phone or not apikey:
        return
    import urllib.parse
    encoded_message = urllib.parse.quote(message)
    url = f"https://api.callmebot.com/whatsapp.php?phone={phone}&text={encoded_message}&apikey={apikey}"
    requests.get(url)

# ==========================================
# TICKER LADEN
# ==========================================
def fetch_combined_tickers():
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    all_tickers = []

    # S&P 500
    try:
        url_sp = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        df_sp = pd.read_html(url_sp, storage_options=headers)[0]
        sp_tickers = df_sp["Symbol"].str.replace(".", "-", regex=False).tolist()
        all_tickers.extend(sp_tickers)
    except Exception as e:
        print(f"Fehler S&P 500: {e}")

    # Nasdaq 100
    ndx_tickers = []
    try:
        url_ndx = "https://www.slickcharts.com/nasdaq100"
        df_list = pd.read_html(url_ndx, storage_options=headers)
        for df in df_list:
            if "Symbol" in df.columns:
                ndx_tickers = df["Symbol"].dropna().astype(str).str.replace(".", "-", regex=False).tolist()
                if len(ndx_tickers) > 50:
                    break
    except Exception as e:
        print(f"Hinweis Slickcharts: {e}")

    if ndx_tickers:
        all_tickers.extend(ndx_tickers)

    return list(set(all_tickers))

# ==========================================
# ANALYSE & FILTER
# ==========================================
def check_5y_profits(t_obj):
    try:
        income = t_obj.income_stmt
        if income is None or income.empty or "Net Income" not in income.index:
            return False
        net_income = income.loc["Net Income"].dropna()
        if len(net_income) < MIN_AVAILABLE_YEARS:
            return False
        last_years = net_income.sort_index(ascending=False).head(REQUIRED_YEARS)
        for _, value in last_years.items():
            if value <= 0:
                return False
        return True
    except Exception:
        return False

def check_multi_year_fcf(t_obj, sector):
    if sector == "Financial Services":
        return True
    try:
        cf = t_obj.cashflow
        if cf is None or cf.empty or "Free Cash Flow" not in cf.index:
            return False
        fcf = cf.loc["Free Cash Flow"].dropna()
        if len(fcf) < FCF_MIN_YEARS:
            return False
        if any(v <= 0 for v in fcf.values):
            return False
        return True
    except Exception:
        return False

def calculate_scenario_returns(info, current_price):
    forward_eps = info.get("forwardEps")
    if not forward_eps or forward_eps <= 0 or not current_price:
        return 0.0
    growth = info.get("earningsGrowth") or info.get("revenueGrowth") or 0.0
    growth = min(max(growth, 0.0), 1.0)
    eps_2y = forward_eps * (1 + growth)
    model_price = eps_2y * 25  # KGV-25 Basis für das Ranking
    return round((model_price / current_price) ** (1 / SCENARIO_HORIZON_YEARS) - 1, 4) * 100

def process_batch(ticker_batch):
    results = []
    for _ in range(3):
        try:
            time.sleep(0.5)
            tickers_obj = yf.Tickers(" ".join(ticker_batch))
            for ticker in ticker_batch:
                try:
                    t_obj = tickers_obj.tickers.get(ticker)
                    if not t_obj:
                        continue
                    info = getattr(t_obj, "info", None)
                    if not info or not isinstance(info, dict):
                        continue

                    current_price = info.get("currentPrice", info.get("regularMarketPrice", 0))
                    market_cap = info.get("marketCap", 0)
                    if not market_cap or not current_price:
                        continue

                    sector = info.get("sector", "")
                    if USE_5Y_PROFIT_FILTER and not check_5y_profits(t_obj):
                        continue
                    if USE_FCF_FILTER and not check_multi_year_fcf(t_obj, sector):
                        continue

                    rev_growth = info.get("revenueGrowth", 0.0) or 0.0
                    profit_margin = info.get("profitMargins", 0.0) or 0.0
                    roe = info.get("returnOnEquity", 0.0) or 0.0
                    
                    # Quality Score Berechnung (0 bis 6)
                    q_score = 0
                    if roe >= 0.15: q_score += 1
                    if info.get("freeCashflow", 0) > 0: q_score += 1
                    if profit_margin >= 0.15: q_score += 1
                    payout = info.get("payoutRatio")
                    if payout is not None and 0 < payout <= 0.60: q_score += 1
                    dte = info.get("debtToEquity")
                    if sector != "Financial Services" and dte is not None and dte <= 100: q_score += 1
                    if rev_growth >= 0.08: q_score += 1

                    kgv25_return = calculate_scenario_returns(info, current_price)

                    results.append({
                        "Ticker": ticker,
                        "Name": info.get("shortName", ticker),
                        "Sector": sector,
                        "Price": current_price,
                        "MarketCap_Brd": round(market_cap / 1_000_000_000, 2),
                        "RevenueGrowth_%": round(rev_growth * 100, 2),
                        "NetMargin_%": round(profit_margin * 100, 2),
                        "ROE_%": round(roe * 100, 1),
                        "KGV25_Rendite": kgv25_return,
                        "Quality_Score": q_score
                    })
                except Exception:
                    continue
            break
        except Exception:
            time.sleep(2)
    return results

def main():
    print("Starte Screening für Tagessieger-Ermittlung...")
    tickers = fetch_combined_tickers()
    if not tickers:
        print("Keine Ticker gefunden.")
        return

    batches = [tickers[i:i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    all_results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_batch, batch) for batch in batches]
        for future in as_completed(futures):
            res = future.result()
            if res:
                all_results.extend(res)

    if not all_results:
        print("Keine Aktien haben alle Filter passiert.")
        return

    df = pd.DataFrame(all_results)
    
    # Sortierung nach Gesamt-Score (Primär Quality Score, Sekundär KGV-25 Rendite & Gewinnwachstum)
    df = df.sort_values(by=["Quality_Score", "KGV25_Rendite", "RevenueGrowth_%"], ascending=[False, False, False]).reset_index(drop=True)

    winner = df.iloc[0]
    runner_up = df.iloc[1] if len(df) > 1 else None

    # Nachricht zusammenbauen
    msg_lines = [
        "🏆 *AKTIE DES TAGES (SCREENER-SIEGER)* 🏆\n",
        f"🥇 *1. Platz: {winner['Ticker']} ({winner['Name']})*",
        f"• Sektor: {winner['Sector']}",
        f"• Quality Score: {winner['Quality_Score']}/6",
        f"• KGV-25 Rendite p.a.: {winner['KGV25_Rendite']}%",
        f"• Umsatzwachstum: {winner['RevenueGrowth_%']}%",
        f"• Nettomarge: {winner['NetMargin_%']}%",
        f"• ROE: {winner['ROE_%']}%\n"
    ]

    if runner_up is not None:
        msg_lines.extend([
            f"🥈 *2. Platz (Vergleich): {runner_up['Ticker']} ({runner_up['Name']})*",
            f"• Quality Score: {runner_up['Quality_Score']}/6",
            f"• KGV-25 Rendite p.a.: {runner_up['KGV25_Rendite']}%",
            f"• Umsatzwachstum: {runner_up['RevenueGrowth_%']}%\n",
            f"💡 *Begründung:* {winner['Ticker']} setzt sich gegen {runner_up['Ticker']} durch, da sie einen überlegenen Quality-Score ({winner['Quality_Score']} vs {runner_up['Quality_Score']}) und eine stärkere modellierte Jahresrendite aufweisen."
        ])

    final_message = "\n".join(msg_lines)
    
    # Senden über bestehende Kanäle
    send_telegram(final_message)
    send_whatsapp_callmebot(final_message)
    print("Tagessieger erfolgreich ermittelt und versendet!")

if __name__ == "__main__":
    main()