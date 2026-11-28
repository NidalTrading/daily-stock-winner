# -*- coding: utf-8 -*-
"""
Optimierter Fundamentaler Aktien-Screener (Maximal-Paket: Fadi + Qussai)
- Fadi: 5J Profitabilität, Equity-Trend, FCF, PEG
- Qussai: Rating-Labels, Peer-Vergleich, Forward-Estimates, Quartals-Check, Markdown-Reports
"""

import os
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd
import requests
import yfinance as yf

# ==========================================
# CONFIG & SCHWELLENWERTE
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
REPORTS_DIR = os.path.join(SCRIPT_DIR, "reports")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

TODAY = datetime.now().strftime("%Y-%m-%d")
CSV_FILENAME = os.path.join(RESULTS_DIR, f"SPNASDAQ100_RESULTS_{TODAY}.csv")
LOG_FILENAME = os.path.join(RESULTS_DIR, f"screener_{TODAY}.log")

BATCH_SIZE = 15
MAX_WORKERS = 2
REQUEST_DELAY = 0.8

MIN_MARKET_CAP = 2_000_000_000
MAX_DIST_FROM_HIGH = 0.60

USE_5Y_PROFIT_FILTER = True
REQUIRED_YEARS = 5
MIN_AVAILABLE_YEARS = 4

USE_FCF_FILTER = True
FCF_MIN_YEARS = 3
FCF_SUM_POSITIVE = True

USE_EQUITY_TREND_FILTER = True
EQUITY_MIN_YEARS = 3

SCENARIO_HORIZON_YEARS = 2
EPS_GROWTH_CAP = 0.25

SECTOR_TARGET_PE = {
    "Technology": 30, "Communication Services": 25, "Healthcare": 22,
    "Consumer Cyclical": 20, "Consumer Defensive": 20, "Industrials": 20,
    "Energy": 15, "Financial Services": 15, "Utilities": 18, "Real Estate": 18, "Basic Materials": 18,
}
DEFAULT_TARGET_PE = 22

EXCLUDE_TICKERS = {"GOOG"}
MAX_PER_SECTOR_TOP5 = 2
TOP5_SIZE = 5
MIN_UPSIDE_TOP5 = -10.0

BASE_COLUMNS = [
    "Ticker", "Name", "Sector", "Price", "Rating", "RevenueGrowth_%", "NetMargin_%", 
    "ROE_%", "Forward_PE", "PEG", "Equity_B", "Quality_Score", "Composite_Score", 
    "Fair_Value_Base", "Target_PE", "Upside_Base_%"
]
REV_COLUMNS = [f"Rev_{year}" for year in range(2016, 2028)]
ALL_COLUMNS = BASE_COLUMNS + REV_COLUMNS

csv_lock = threading.Lock()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILENAME, encoding="utf-8"), logging.StreamHandler()]
)
log = logging.getLogger("screener")

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
WIKI_HEADERS = {"User-Agent": "AktienScreener/1.0 (https://github.com/NidalTrading; nidal.ali@gmail.com) Python-requests/2.31"}

# ==========================================
# HILFSFUNKTIONEN
# ==========================================
def safe_float(value, default=None):
    try:
        if value is None or pd.isna(value): return default
        res = float(value)
        if res != res or res in (float("inf"), float("-inf")): return default
        return res
    except (TypeError, ValueError):
        return default

def row_value(frame, labels):
    if frame is None or frame.empty: return pd.Series(dtype=float)
    for label in labels:
        if label in frame.index:
            return frame.loc[label].dropna().sort_index(ascending=True)
    return pd.Series(dtype=float)

def growth_trend_ok(values, min_positive_ratio=0.6):
    if len(values) < 2: return False
    diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    positive = sum(1 for d in diffs if d > 0)
    return positive >= len(diffs) * min_positive_ratio

def format_rev_string(row):
    rev_cols = [c for c in row.index if str(c).startswith("Rev_")]
    if not rev_cols: return "N/A"
    parts = []
    for c in sorted(rev_cols):
        if pd.notna(row[c]) and str(row[c]).lower() != "nan":
            parts.append(f"{c.replace('Rev_', '')}: {row[c]}")
    return " ➔ ".join(parts) if parts else "N/A"

def get_rating(comp_score, upside, peg):
    if comp_score >= 160 and upside >= 20 and (peg is None or peg < 1.2): return "STRONG BUY"
    elif comp_score >= 130 and upside >= 5: return "BUY"
    elif comp_score >= 100 and upside >= -15: return "HOLD"
    else: return "AVOID"

# ==========================================
# TICKER-BESCHAFFUNG
# ==========================================
def fetch_combined_tickers():
    all_tickers, sp_tickers = [], []
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
        df = pd.read_csv(url)
        sp_tickers = df["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
    except Exception:
        pass

    if not sp_tickers:
        try:
            df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", storage_options=WIKI_HEADERS)[0]
            sp_tickers = df["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
        except Exception:
            pass
    all_tickers.extend(sp_tickers)

    ndx_tickers = []
    try:
        for df in pd.read_html("https://www.slickcharts.com/nasdaq100", storage_options={"User-Agent": USER_AGENT}):
            if "Symbol" in df.columns:
                ndx_tickers = df["Symbol"].dropna().astype(str).str.replace(".", "-", regex=False).tolist()
                if len(ndx_tickers) > 50: break
    except Exception: pass

    all_tickers.extend(ndx_tickers)
    return sorted(set(all_tickers) - EXCLUDE_TICKERS)

def get_fear_and_greed_index():
    try:
        res = requests.get("https://production.dataviz.cnn.io/index/fearandgreed/graphdata", headers={"User-Agent": USER_AGENT}, timeout=10)
        if res.status_code == 200:
            data = res.json().get("fear_and_greed", {})
            return round(float(data.get("score")), 1), str(data.get("rating")).replace("_", " ").title()
    except Exception: pass
    return None, "N/A"

# ==========================================
# FUNDAMENTALE PRÜFUNGEN & ANALYSE
# ==========================================
def check_5y_profit_and_growth(t_obj):
    try:
        income = getattr(t_obj, "income_stmt", None)
        if income is None or income.empty: return False, {}
        net_income = row_value(income, ["Net Income", "Net Income Common Stockholders"])
        revenue = row_value(income, ["Total Revenue", "Operating Revenue"])
        if len(net_income) < MIN_AVAILABLE_YEARS or len(revenue) < MIN_AVAILABLE_YEARS: return False, {}
        
        ni_tail, rev_tail = net_income.tail(REQUIRED_YEARS), revenue.tail(REQUIRED_YEARS)
        if any(v <= 0 for v in ni_tail.values): return False, {}
        if not growth_trend_ok(ni_tail.values) or not growth_trend_ok(rev_tail.values): return False, {}
        
        rev_dict = {}
        for idx, val in rev_tail.items():
            year_str = str(idx.year) if hasattr(idx, "year") else str(idx)[:4]
            rev_dict[f"Rev_{year_str}"] = val
        return True, rev_dict
    except Exception: return False, {}

def check_multi_year_fcf(t_obj, sector):
    if sector == "Financial Services": return True
    try:
        cf = getattr(t_obj, "cashflow", None)
        fcf_series = row_value(cf, ["Free Cash Flow"])
        if fcf_series.empty or len(fcf_series) < FCF_MIN_YEARS: return False
        if fcf_series.tail(FCF_MIN_YEARS).sum() <= 0: return False
        return True
    except Exception: return False

def check_equity_trend(t_obj):
    try:
        balance = getattr(t_obj, "balance_sheet", None)
        equity = row_value(balance, ["Stockholders Equity", "Total Stockholder Equity"])
        if equity.empty or len(equity) < EQUITY_MIN_YEARS: return False, 0.0
        eq_vals = equity.tail(EQUITY_MIN_YEARS).values
        if eq_vals[-1] <= eq_vals[0] or not growth_trend_ok(eq_vals, 0.5): return False, float(eq_vals[-1])
        return True, float(eq_vals[-1])
    except Exception: return False, 0.0

def calculate_scenarios(info, current_price, forward_eps, sector):
    growth = min(max(safe_float(info.get("earningsGrowth") or info.get("revenueGrowth"), 0.08), -0.20), EPS_GROWTH_CAP)
    if not forward_eps or forward_eps <= 0 or not current_price: return {k: 0 for k in ["Target_PE", "Fair_Value_Base", "Upside_Base_%"]} | {"PEG": None}
    
    eps_2y = forward_eps * (1 + growth) ** 2
    target_pe = max(min(SECTOR_TARGET_PE.get(sector, DEFAULT_TARGET_PE), max(safe_float(info.get("forwardPE"), 22), 5.0)), 5.0)
    
    fair_base = eps_2y * target_pe
    upside_base = (fair_base / current_price - 1) * 100
    
    earnings_growth = min(max(safe_float(info.get("earningsGrowth"), 0.10), 0.05), 0.25)
    fwd_pe = safe_float(info.get("forwardPE"), 0.0)
    peg = round(fwd_pe / (earnings_growth * 100), 2) if fwd_pe > 0 else None

    return {"Target_PE": round(target_pe, 1), "Fair_Value_Base": round(fair_base, 2), "Upside_Base_%": round(upside_base, 1), "PEG": peg}

def analyze_ticker(ticker):
    try:
        t_obj = yf.Ticker(ticker)
        info = t_obj.info
        if not info or not isinstance(info, dict): return None, "keine Info"

        current_price = safe_float(info.get("currentPrice", info.get("regularMarketPrice")))
        market_cap = safe_float(info.get("marketCap"))
        fifty_two_high = safe_float(info.get("fiftyTwoWeekHigh"))

        if not current_price or not market_cap or market_cap < MIN_MARKET_CAP: return None, "zu klein"
        if fifty_two_high and current_price < (fifty_two_high * (1.0 - MAX_DIST_FROM_HIGH)): return None, "zu weit unten"
        sector = info.get("sector", "") or ""

        eq_ok, eq_val = check_equity_trend(t_obj)
        if USE_EQUITY_TREND_FILTER and not eq_ok: return None, "Equity fail"

        p_ok, rev_dict = check_5y_profit_and_growth(t_obj)
        if USE_5Y_PROFIT_FILTER and not p_ok: return None, "Profit fail"

        if USE_FCF_FILTER and not check_multi_year_fcf(t_obj, sector): return None, "FCF fail"

        rev_growth, margin, fwd_pe, roe = safe_float(info.get("revenueGrowth"), 0), safe_float(info.get("profitMargins"), 0), safe_float(info.get("forwardPE"), 0), safe_float(info.get("returnOnEquity"), 0)
        
        quality_score = sum([roe >= 0.15, safe_float(info.get("freeCashflow"), 0) > 0, margin >= 0.15, 0 < safe_float(info.get("payoutRatio"), 0) <= 0.60, sector != "Financial Services" and 0 < safe_float(info.get("debtToEquity"), 0) <= 100, rev_growth >= 0.08, 0 < fwd_pe < 30, eq_val > 0])
        
        scen = calculate_scenarios(info, current_price, safe_float(info.get("forwardEps")), sector)
        comp_score = (quality_score * 15.0) + (max(rev_growth, 0)*40) + (max(margin, 0)*40) + (max(roe, 0)*20) + (min(max(scen["Upside_Base_%"], -50), 100)*0.2)
        rating = get_rating(comp_score, scen["Upside_Base_%"], scen["PEG"])

        res = {
            "Ticker": ticker, "Name": info.get("shortName", ticker), "Sector": sector, "Price": current_price, "Rating": rating,
            "RevenueGrowth_%": round(rev_growth*100, 2), "NetMargin_%": round(margin*100, 2), "ROE_%": round(roe*100, 1),
            "Forward_PE": round(fwd_pe, 1), "PEG": scen["PEG"], "Equity_B": round(eq_val/1e9, 2) if eq_val else None,
            "Quality_Score": quality_score, "Composite_Score": round(comp_score, 1), "Fair_Value_Base": scen["Fair_Value_Base"],
            "Target_PE": scen["Target_PE"], "Upside_Base_%": scen["Upside_Base_%"]
        }
        for k, v in rev_dict.items(): res[k] = f"${v/1e9:.2f}B" if v else "N/A"
        return res, "OK"
    except Exception as e:
        return None, f"Err: {e}"

def process_batch(batch):
    results = []
    for t in batch:
        time.sleep(REQUEST_DELAY)
        res, _ = analyze_ticker(t)
        if res:
            for col in ALL_COLUMNS: res.setdefault(col, None)
            results.append({col: res.get(col) for col in ALL_COLUMNS})
    if results:
        df = pd.DataFrame(results, columns=ALL_COLUMNS)
        with csv_lock:
            df.to_csv(CSV_FILENAME, mode="a", header=not os.path.exists(CSV_FILENAME), index=False, encoding="utf-8-sig")
    return len(results)

# ==========================================
# TOP 5, REPORTS & BENACHRICHTIGUNGEN
# ==========================================
def select_top5(df):
    selected, sectors = [], {}
    for _, row in df.iterrows():
        if safe_float(row.get("Upside_Base_%"), 0) < MIN_UPSIDE_TOP5: continue
        sec = row.get("Sector", "Unknown")
        if sectors.get(sec, 0) < MAX_PER_SECTOR_TOP5:
            selected.append(row); sectors[sec] = sectors.get(sec, 0) + 1
        if len(selected) == TOP5_SIZE: break
    return pd.DataFrame(selected)

def generate_qussai_report(row, df):
    ticker, sector = row["Ticker"], row["Sector"]
    peers = df[(df["Sector"] == sector) & (df["Forward_PE"] > 0) & (df["Forward_PE"] < 100)]
    median_pe = peers["Forward_PE"].median() if not peers.empty else None
    premium = ((row["Forward_PE"] / median_pe - 1) * 100) if median_pe else 0
    
    try:
        rev_est = yf.Ticker(ticker).revenue_estimate
        fwd_rev = rev_est.loc["+1y", "growth"] * 100 if not rev_est.empty else 0
    except Exception:
        fwd_rev = 0

    hist = format_rev_string(row).replace(' ➔ ', '\n- ')
    median_pe_str = f"{median_pe:.1f}" if median_pe else 'N/A'

    md = f"""# {ticker} – Tiefenanalyse ({TODAY})

**Rating: {row['Rating']}** | Sektor: {sector}

## 1. Unternehmensprofil & Trends
- **Name:** {row['Name']}
- **Kurs:** ${row['Price']:.2f} | **Fair Value:** ${row['Fair_Value_Base']:.2f} (Upside: {row['Upside_Base_%']:.1f}%)
- **Umsatzwachstum (TTM):** {row['RevenueGrowth_%']:.1f}% | **Forward Revenue (Konsens):** {fwd_rev:.1f}%
- **Nettomarge:** {row['NetMargin_%']:.1f}% | **ROE:** {row['ROE_%']:.1f}%

## 2. Bewertung & Peer-Vergleich
- **Forward KGV:** {row['Forward_PE']:.1f} | **Sektor-Median:** {median_pe_str}
- **Premium/Discount zum Sektor:** {premium:+.1f}%
- **PEG Ratio:** {row['PEG'] if pd.notna(row['PEG']) else 'N/A'}

## 3. Umsatzhistorie (letzte Jahre)
- {hist}

## 4. Fadi's Qualitäts-Check
- **Quality Score:** {int(row['Quality_Score'])}/8
- **Equity:** ${row['Equity_B']} Mrd. (Trend steigend)

---
*Automatisch generiert vom Aktien-Screener (Fadi + Qussai Methodik). Keine Anlageberatung.*
"""
    with open(os.path.join(REPORTS_DIR, f"{ticker}.md"), "w", encoding="utf-8") as f:
        f.write(md)

def evaluate_and_report():
    if not os.path.exists(CSV_FILENAME): return
    df = pd.read_csv(CSV_FILENAME)
    df.sort_values(by="Composite_Score", ascending=False, inplace=True)
    top_5 = select_top5(df)
    
    if top_5.empty: return

    for _, row in top_5.iterrows():
        generate_qussai_report(row, df)

    fg_score, fg_rating = get_fear_and_greed_index()
    report = f"🏆 *TOP 5 AKTIEN (Fadi + Qussai)* 🏆\n🌐 F&G: {fg_score} ({fg_rating})\n\n"
    
    for i, row in top_5.iterrows():
        report += f"{i+1}. *{row['Ticker']}* ({row['Rating']}) | Q:{int(row['Quality_Score'])}/8\n"
        report += f"   • {row['Sector']} | Upside: {min(row['Upside_Base_%'], 100):.0f}%\n"
        report += f"   • PEG: {row['PEG']} | Umsätze: {format_rev_string(row)[:50]}...\n\n"

    print(report)
    
    # Telegram & WhatsApp (läuft nur, wenn Secrets existieren)
    def send_telegram(msg):
        token, chat_id = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
        if token and chat_id:
            try: requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}, timeout=10)
            except Exception as e: log.error(f"Telegram Fehler: {e}")

    def send_whatsapp(msg):
        phone, apikey = os.getenv("WHATSAPP_PHONE"), os.getenv("WHATSAPP_API_KEY")
        if phone and apikey:
            try: requests.get(f"https://api.callmebot.com/whatsapp.php?phone={phone}&text={requests.utils.quote(msg)}&apikey={apikey}", timeout=10)
            except Exception as e: log.error(f"WhatsApp Fehler: {e}")

    send_telegram(report)
    send_whatsapp(report)

# ==========================================
# MAIN
# ==========================================
def main():
    log.info("Lade Ticker...")
    tickers = fetch_combined_tickers()
    if os.path.exists(CSV_FILENAME): os.remove(CSV_FILENAME)
    
    log.info(f"Analysiere {len(tickers)} Ticker...")
    batches = [tickers[i:i+BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(process_batch, b) for b in batches]
        for i, f in enumerate(as_completed(futures)):
            log.info(f"Fortschritt: {i+1}/{len(batches)}")

    log.info("Erstelle Reports...")
    evaluate_and_report()
    log.info("Fertig!")

if __name__ == "__main__":
    main()
