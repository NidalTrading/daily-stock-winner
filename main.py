import os
import json
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
HISTORY_FILENAME = os.path.join(SCRIPT_DIR, "TOP5_HISTORY.json")

BATCH_SIZE = 15
MAX_WORKERS = 2

# FILTER-EINSTELLUNGEN
MIN_MARKET_CAP = 2_000_000_000   # 2 Mrd. USD
MAX_DIST_FROM_HIGH = 0.20        # Maximal 20% unter dem 52-Wochen-Hoch

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
    """
    Berechnet zukünftige Erträge und bewertet das zukünftige PE (Forward / Scenario P/E) 
    angelehnt an das methodische Vorgehen bei Wachstumsanalysen.
    """
    forward_eps = info.get("forwardEps")
    if not forward_eps or forward_eps <= 0 or not current_price:
        return {"KGV25_Rendite_pa_%": 0.0, "Future_PE_Implied": 0.0}
    
    growth = info.get("earningsGrowth") or info.get("revenueGrowth") or 0.0
    growth = min(max(growth, 0.0), 1.0)
    eps_2y = forward_eps * (1 + growth)
    
    model_price = eps_2y * 25
    ann_return = (model_price / current_price) ** (1 / SCENARIO_HORIZON_YEARS) - 1
    
    future_pe = current_price / eps_2y if eps_2y > 0 else 0.0

    return {
        "KGV25_Rendite_pa_%": round(ann_return * 100, 1),
        "Future_PE_Implied": round(future_pe, 1)
    }


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
                    fifty_two_high = info.get("fiftyTwoWeekHigh", 0)

                    if not market_cap or not current_price: continue

                    # 1. Market Cap Filter (> 2 Mrd. USD)
                    if market_cap < MIN_MARKET_CAP:
                        continue

                    # 2. 52-Wochen-Hoch Filter (Maximal 20% darunter)
                    if fifty_two_high and fifty_two_high > 0:
                        if current_price < (fifty_two_high * (1.0 - MAX_DIST_FROM_HIGH)):
                            continue

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
