
# -*- coding: utf-8 -*-
"""
Optimierter Fundamentaler Aktien-Screener (S&P 500 & Nasdaq 100)
Basierend auf Fadi's Maturity Model & Video-Regeln.

Option B+ – finale Version mit robuster Ticker-Beschaffung:
- S&P 500 primär von GitHub (kein Rate-Limit)
- Fallback 1: Wikipedia mit Retry
- Fallback 2: nur Nasdaq 100
- EPS_GROWTH_CAP = 0.25
- PEG nutzt earningsGrowth (5–25% gekappt)
- forwardPE-Untergrenze 5.0
- Upside im Report auf 100% gekappt mit Warnhinweis
- nan-Filter im Report
- Sektor-Cap in Top 5 (max. 2 pro Sektor)
- Mindest-Upside-Filter (-10%) in Top-5-Auswahl
- Sektor in WhatsApp-Nachricht
- REV_COLUMNS dynamisch 2020–2027
"""

import os
import json
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
os.makedirs(RESULTS_DIR, exist_ok=True)

TODAY = datetime.now().strftime("%Y-%m-%d")
CSV_FILENAME = os.path.join(RESULTS_DIR, f"SPNASDAQ100_RESULTS_{TODAY}.csv")
HISTORY_FILENAME = os.path.join(SCRIPT_DIR, "TOP5_HISTORY.json")
LOG_FILENAME = os.path.join(RESULTS_DIR, f"screener_{TODAY}.log")

BATCH_SIZE = 15
MAX_WORKERS = 2
REQUEST_DELAY = 0.8
RETRIES = 3

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

USE_NET_CASH_FILTER = False
NET_CASH_EXEMPT_SECTORS = {"Financial Services", "Utilities", "Real Estate"}

SCENARIO_HORIZON_YEARS = 2
EPS_GROWTH_CAP = 0.25

SECTOR_TARGET_PE = {
    "Technology": 30,
    "Communication Services": 25,
    "Healthcare": 22,
    "Consumer Cyclical": 20,
    "Consumer Defensive": 20,
    "Industrials": 20,
    "Energy": 15,
    "Financial Services": 15,
    "Utilities": 18,
    "Real Estate": 18,
    "Basic Materials": 18,
}
DEFAULT_TARGET_PE = 22

EXCLUDE_TICKERS = {"GOOG"}

MAX_PER_SECTOR_TOP5 = 2
TOP5_SIZE = 5
MIN_UPSIDE_TOP5 = -10.0

BASE_COLUMNS = [
    "Ticker", "Name", "Sector", "Price",
    "RevenueGrowth_%", "NetMargin_%", "ROE_%", "Forward_PE", "PEG",
    "Equity_B", "Net_Cash_B", "Quality_Score", "Composite_Score",
    "Fair_Value_Base", "Target_PE", "Upside_Base_%", "Future_PE"
]
REV_COLUMNS = [f"Rev_{year}" for year in range(2020, 2028)]
ALL_COLUMNS = BASE_COLUMNS + REV_COLUMNS

csv_lock = threading.Lock()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILENAME, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("screener")

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Custom User-Agent für Wikipedia (mit Kontakt laut Richtlinie)
WIKI_HEADERS = {
    "User-Agent": (
        "AktienScreener/1.0 "
        "(https://github.com/alin/aktien-screener; alin@example.com) "
        "Python-requests/2.31"
    )
}


# ==========================================
# HILFSFUNKTIONEN
# ==========================================
def safe_float(value, default=None):
    try:
        if value is None or pd.isna(value):
            return default
        res = float(value)
        if res != res or res in (float("inf"), float("-inf")):
            return default
        return res
    except (TypeError, ValueError):
        return default


def row_value(frame, labels):
    if frame is None or frame.empty:
        return pd.Series(dtype=float)
    for label in labels:
        if label in frame.index:
            return frame.loc[label].dropna().sort_index(ascending=True)
    return pd.Series(dtype=float)


def growth_trend_ok(values, min_positive_ratio=0.6):
    if len(values) < 2:
        return False
    diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    positive = sum(1 for d in diffs if d > 0)
    return positive >= len(diffs) * min_positive_ratio


def format_rev_string(row):
    rev_cols = [c for c in row.index if str(c).startswith("Rev_")]
    if not rev_cols:
        return "N/A"
    parts = []
    for c in sorted(rev_cols):
        if pd.notna(row[c]) and str(row[c]).lower() != "nan":
            parts.append(f"{c.replace('Rev_', '')}: {row[c]}")
    return " ➔ ".join(parts) if parts else "N/A"


# ==========================================
# TICKER-BESCHAFFUNG (ROBUST)
# ==========================================
def fetch_sp500_from_github():
    """Primärquelle: GitHub-Spiegel der S&P 500 Liste (kein Rate-Limit)."""
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
        df = pd.read_csv(url)
        tickers = df["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
        log.info(f"S&P 500 von GitHub geladen: {len(tickers)}")
        return tickers
    except Exception as e:
        log.warning(f"GitHub-Fallback fehlgeschlagen: {e}")
        return []


def fetch_sp500_from_wikipedia():
    """Fallback: Wikipedia mit Retry und realistischem User-Agent."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    for attempt in range(3):
        try:
            df = pd.read_html(url, storage_options=WIKI_HEADERS)[0]
            tickers = df["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
            log.info(f"S&P 500 von Wikipedia geladen: {len(tickers)}")
            return tickers
        except Exception as e:
            wait = 5 * (attempt + 1)
            log.warning(f"Wikipedia-Versuch {attempt+1}/3 fehlgeschlagen ({e}). Warte {wait}s...")
            time.sleep(wait)
    return []


def fetch_nasdaq100():
    """Nasdaq 100: Slickcharts → Nasdaq-API."""
    ndx_tickers = []
    try:
        for df in pd.read_html("https://www.slickcharts.com/nasdaq100",
                                storage_options={"User-Agent": USER_AGENT}):
            if "Symbol" in df.columns:
                ndx_tickers = df["Symbol"].dropna().astype(str).str.replace(".", "-", regex=False).tolist()
                if len(ndx_tickers) > 50:
                    log.info(f"Nasdaq 100 von Slickcharts geladen: {len(ndx_tickers)}")
                    break
    except Exception as e:
        log.warning(f"Slickcharts Hinweis: {e}")

    if len(ndx_tickers) < 50:
        try:
            api_url = "https://api.nasdaq.com/api/quote/list-type/nasdaq100"
            res = requests.get(api_url, headers={"User-Agent": USER_AGENT}, timeout=10)
            if res.status_code == 200:
                rows = res.json().get("data", {}).get("data", {}).get("rows", [])
                ndx_tickers = [r["symbol"].strip().replace(".", "-") for r in rows if "symbol" in r]
                log.info(f"Nasdaq 100 von API geladen: {len(ndx_tickers)}")
        except Exception as e:
            log.warning(f"Nasdaq API Hinweis: {e}")

    return ndx_tickers


def fetch_combined_tickers():
    all_tickers = []

    # 1. S&P 500: primär GitHub, Fallback Wikipedia
    sp_tickers = fetch_sp500_from_github()
    if not sp_tickers:
        log.info("GitHub nicht erreichbar – versuche Wikipedia...")
        sp_tickers = fetch_sp500_from_wikipedia()
    if not sp_tickers:
        log.error("S&P 500 konnte weder von GitHub noch Wikipedia geladen werden – nutze nur Nasdaq 100")
    else:
        all_tickers.extend(sp_tickers)

    # 2. Nasdaq 100
    ndx_tickers = fetch_nasdaq100()
    all_tickers.extend(ndx_tickers)

    unique_tickers = sorted(set(all_tickers) - EXCLUDE_TICKERS)
    log.info(f"Einzigartige Ticker nach Bereinigung: {len(unique_tickers)}")
    return unique_tickers


def get_fear_and_greed_index():
    try:
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json().get("fear_and_greed", {})
            score = data.get("score")
            rating = data.get("rating")
            if score is not None:
                return round(float(score), 1), str(rating).replace("_", " ").title()
    except Exception as e:
        log.warning(f"Fear & Greed Index nicht ladbar: {e}")
    return None, "N/A"


# ==========================================
# FUNDAMENTALE PRÜFUNGEN
# ==========================================
def check_5y_profit_and_growth(t_obj):
    try:
        income = getattr(t_obj, "income_stmt", None)
        if income is None or income.empty:
            return False, "keine GuV-Daten", {}

        net_income = row_value(income, ["Net Income", "Net Income Common Stockholders"])
        revenue = row_value(income, ["Total Revenue", "Operating Revenue"])

        if len(net_income) < MIN_AVAILABLE_YEARS or len(revenue) < MIN_AVAILABLE_YEARS:
            return False, "zu wenig Historie", {}

        ni_tail = net_income.tail(REQUIRED_YEARS)
        rev_tail = revenue.tail(REQUIRED_YEARS)

        ni_vals = ni_tail.values
        rev_vals = rev_tail.values

        if any(v <= 0 for v in ni_vals):
            return False, "Verlustjahr in Historie", {}

        if not growth_trend_ok(ni_vals, 0.6):
            return False, "Gewinn-Trend nicht steigend", {}
        if not growth_trend_ok(rev_vals, 0.6):
            return False, "Umsatz-Trend nicht steigend", {}

        rev_dict = {}
        for idx, val in rev_tail.items():
            year_str = str(idx.year) if hasattr(idx, "year") else str(idx)[:4]
            rev_dict[f"Rev_{year_str}"] = val

        return True, "profitabel & wachsend", rev_dict
    except Exception as e:
        log.debug(f"Fehler GuV-Prüfung: {e}")
        return False, "Fehler bei GuV-Prüfung", {}


def check_multi_year_fcf(t_obj, sector):
    if sector == "Financial Services":
        return True, "Financials: FCF N/A"
    try:
        cf = getattr(t_obj, "cashflow", None)
        fcf_series = row_value(cf, ["Free Cash Flow"])
        if fcf_series.empty or len(fcf_series) < FCF_MIN_YEARS:
            return False, "zu wenig FCF-Historie"

        fcf_tail = fcf_series.tail(FCF_MIN_YEARS)

        if FCF_SUM_POSITIVE:
            if fcf_tail.sum() <= 0:
                return False, "FCF-Summe negativ"
        else:
            if any(v <= 0 for v in fcf_tail.values):
                return False, "negativer FCF"

        return True, "positiver FCF-Trend"
    except Exception as e:
        log.debug(f"Fehler FCF-Prüfung: {e}")
        return False, "Fehler bei FCF-Prüfung"


def check_equity_trend(t_obj):
    try:
        balance = getattr(t_obj, "balance_sheet", None)
        equity = row_value(balance, ["Stockholders Equity", "Total Stockholder Equity"])
        if equity.empty or len(equity) < EQUITY_MIN_YEARS:
            return False, "zu wenig Equity-Historie", 0.0

        eq_tail = equity.tail(EQUITY_MIN_YEARS)
        eq_vals = eq_tail.values

        if eq_vals[-1] <= eq_vals[0]:
            return False, "Equity nicht wachsend", float(eq_vals[-1])

        if not growth_trend_ok(eq_vals, 0.5):
            return False, "Equity-Trend nicht steigend", float(eq_vals[-1])

        return True, "Equity wachsend", float(eq_vals[-1])
    except Exception as e:
        log.debug(f"Fehler Equity-Prüfung: {e}")
        return False, "Fehler bei Equity-Prüfung", 0.0


def check_net_cash(info, sector):
    if sector in NET_CASH_EXEMPT_SECTORS:
        return True, "Sektor-Ausnahme", None
    total_cash = safe_float(info.get("totalCash"), 0)
    total_debt = safe_float(info.get("totalDebt"), 0)
    if total_cash is None or total_debt is None:
        return True, "keine Daten", None
    net_cash = total_cash - total_debt
    return net_cash >= 0, "Net-Cash positiv" if net_cash >= 0 else "Net-Cash negativ", net_cash


# ==========================================
# BEWERTUNG & SCENARIOS
# ==========================================
def calculate_scenarios(info, current_price, forward_eps, sector):
    growth = info.get("earningsGrowth") or info.get("revenueGrowth") or 0.08
    growth = min(max(safe_float(growth, 0.08), -0.20), EPS_GROWTH_CAP)

    if not forward_eps or forward_eps <= 0 or not current_price:
        return {
            "EPS_2Y": 0, "Target_PE": 0,
            "Fair_Value_Base": 0, "Upside_Base_%": 0,
            "Future_PE": 0, "PEG": None
        }

    eps_2y = forward_eps * (1 + growth) * (1 + growth)

    target_pe = SECTOR_TARGET_PE.get(sector, DEFAULT_TARGET_PE)
    current_forward_pe = safe_float(info.get("forwardPE"), target_pe)
    target_pe = min(target_pe, max(current_forward_pe, 5.0))
    target_pe = max(target_pe, 5.0)

    fair_base = eps_2y * target_pe
    upside_base = (fair_base / current_price - 1) * 100
    future_pe = current_price / eps_2y if eps_2y > 0 else 0.0

    earnings_growth = safe_float(info.get("earningsGrowth"),
                                 safe_float(info.get("revenueGrowth"), 0.10))
    earnings_growth = min(max(earnings_growth, 0.05), 0.25)
    forward_pe = safe_float(info.get("forwardPE"), 0.0)
    peg = round(forward_pe / (earnings_growth * 100), 2) if forward_pe > 0 else None

    return {
        "EPS_2Y": round(eps_2y, 2),
        "Target_PE": round(target_pe, 1),
        "Fair_Value_Base": round(fair_base, 2),
        "Upside_Base_%": round(upside_base, 1),
        "Future_PE": round(future_pe, 1),
        "PEG": peg
    }


# ==========================================
# ANALYSE EINZELNER TICKER
# ==========================================
def analyze_ticker(ticker):
    try:
        t_obj = yf.Ticker(ticker)
        info = t_obj.info
        if not info or not isinstance(info, dict):
            return None, "keine Info"

        current_price = safe_float(info.get("currentPrice", info.get("regularMarketPrice")))
        market_cap = safe_float(info.get("marketCap"))
        fifty_two_high = safe_float(info.get("fiftyTwoWeekHigh"))

        if not current_price or not market_cap or market_cap < MIN_MARKET_CAP:
            return None, "zu klein oder kein Preis"

        if fifty_two_high and fifty_two_high > 0:
            if current_price < (fifty_two_high * (1.0 - MAX_DIST_FROM_HIGH)):
                return None, "zu weit unter 52W-High"

        sector = info.get("sector", "") or ""

        equity_val = None
        if USE_EQUITY_TREND_FILTER:
            eq_ok, eq_reason, equity_val = check_equity_trend(t_obj)
            if not eq_ok:
                return None, eq_reason

        rev_history_dict = {}
        if USE_5Y_PROFIT_FILTER:
            passed, reason, rev_history_dict = check_5y_profit_and_growth(t_obj)
            if not passed:
                return None, reason

        if USE_FCF_FILTER:
            passed, reason = check_multi_year_fcf(t_obj, sector)
            if not passed:
                return None, reason

        net_cash_val = None
        if USE_NET_CASH_FILTER:
            passed, reason, net_cash_val = check_net_cash(info, sector)
            if not passed:
                return None, reason

        rev_growth = safe_float(info.get("revenueGrowth"), 0.0)
        profit_margin = safe_float(info.get("profitMargins"), 0.0)
        forward_pe = safe_float(info.get("forwardPE"), 0.0)
        roe = safe_float(info.get("returnOnEquity"), 0.0)
        payout = safe_float(info.get("payoutRatio"))
        dte = safe_float(info.get("debtToEquity"))
        forward_eps = safe_float(info.get("forwardEps"))

        quality_score = 0
        if roe >= 0.15: quality_score += 1
        if safe_float(info.get("freeCashflow"), 0) > 0: quality_score += 1
        if profit_margin >= 0.15: quality_score += 1
        if payout is not None and 0 < payout <= 0.60: quality_score += 1
        if sector != "Financial Services" and dte is not None and dte <= 100: quality_score += 1
        if rev_growth >= 0.08: quality_score += 1
        if 0 < forward_pe < 30: quality_score += 1
        if equity_val and equity_val > 0: quality_score += 1

        scen = calculate_scenarios(info, current_price, forward_eps, sector)

        composite_score = (
            (quality_score * 15.0)
            + (max(rev_growth, 0) * 100 * 0.4)
            + (max(profit_margin, 0) * 100 * 0.4)
            + (max(roe, 0) * 100 * 0.2)
            + min(max(scen["Upside_Base_%"], -50), 100) * 0.2
        )

        res_dict = {
            "Ticker": ticker,
            "Name": info.get("shortName", ticker),
            "Sector": sector,
            "Price": current_price,
            "RevenueGrowth_%": round(rev_growth * 100, 2),
            "NetMargin_%": round(profit_margin * 100, 2),
            "ROE_%": round(roe * 100, 1),
            "Forward_PE": round(forward_pe, 1),
            "PEG": scen.get("PEG"),
            "Equity_B": round(equity_val / 1e9, 2) if equity_val else None,
            "Net_Cash_B": round(net_cash_val / 1e9, 2) if net_cash_val is not None else None,
            "Quality_Score": quality_score,
            "Composite_Score": round(composite_score, 1),
            "Fair_Value_Base": scen["Fair_Value_Base"],
            "Target_PE": scen["Target_PE"],
            "Upside_Base_%": scen["Upside_Base_%"],
            "Future_PE": scen["Future_PE"]
        }

        for k, v in rev_history_dict.items():
            res_dict[k] = f"${v / 1e9:.2f}B" if v else "N/A"

        return res_dict, "OK"
    except Exception as e:
        log.debug(f"Exception bei {ticker}: {e}")
        return None, f"Exception: {e}"


def process_batch(ticker_batch):
    results = []
    for ticker in ticker_batch:
        time.sleep(REQUEST_DELAY)
        for attempt in range(RETRIES):
            try:
                res, reason = analyze_ticker(ticker)
                if res:
                    for col in ALL_COLUMNS:
                        res.setdefault(col, None)
                    res = {col: res.get(col) for col in ALL_COLUMNS}
                    results.append(res)
                break
            except Exception as e:
                if attempt == RETRIES - 1:
                    log.error(f"Fehler bei {ticker} nach {RETRIES} Versuchen: {e}")
                else:
                    time.sleep(1.5 * (attempt + 1))

    if results:
        df_batch = pd.DataFrame(results, columns=ALL_COLUMNS)
        with csv_lock:
            header_flag = not os.path.exists(CSV_FILENAME)
            df_batch.to_csv(CSV_FILENAME, mode="a", header=header_flag,
                            index=False, encoding="utf-8-sig")

    return len(results)


# ==========================================
# SEKTOR-CAP + MINDEST-UPSIDE FÜR TOP 5
# ==========================================
def select_top5_with_sector_cap(ranked_df, max_per_sector=2, top_n=5, min_upside=-10.0):
    selected = []
    sector_map = {}

    for _, row in ranked_df.iterrows():
        upside = safe_float(row.get("Upside_Base_%"), 0.0)
        if upside is not None and upside < min_upside:
            log.debug(f"Überspringe {row['Ticker']}: Upside {upside:.1f}% < {min_upside}%")
            continue

        sec = row.get("Sector", "") or "Unknown"
        if sector_map.get(sec, 0) < max_per_sector:
            selected.append(row)
            sector_map[sec] = sector_map.get(sec, 0) + 1
        if len(selected) == top_n:
            break

    if len(selected) < top_n:
        selected_tickers = {r["Ticker"] for r in selected}
        for _, row in ranked_df.iterrows():
            if row["Ticker"] not in selected_tickers:
                selected.append(row)
                selected_tickers.add(row["Ticker"])
            if len(selected) == top_n:
                break

    return pd.DataFrame(selected).reset_index(drop=True)


# ==========================================
# BENACHRICHTIGUNGEN & REPORTING
# ==========================================
def send_telegram_message(message):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        log.warning("Telegram-Zugangsdaten fehlen.")
        return
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code == 200:
            log.info("Telegram-Nachricht erfolgreich gesendet.")
        else:
            log.error(f"Telegram-Fehler: {r.text}")
    except Exception as e:
        log.exception(f"Telegram-Ausnahme: {e}")


def send_whatsapp_message(top_5, fg_score, fg_rating, new_candidates, dropped_candidates, rank_changes):
    phone = os.getenv("WHATSAPP_PHONE")
    apikey = os.getenv("WHATSAPP_API_KEY")
    if not phone or not apikey:
        log.warning("WhatsApp-Zugangsdaten fehlen.")
        return

    wa_msg = "🏆 *TOP 5 AKTIEN (SCREENER)*\n"
    if fg_score is not None:
        wa_msg += f"🌐 F&G: {fg_score} ({fg_rating})\n\n"

    medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
    for i, row in top_5.iterrows():
        rank = i + 1
        medal = medals[rank - 1] if rank <= len(medals) else f"{rank}."
        is_new = row["Ticker"] in new_candidates
        new_tag = " 🆕" if is_new else ""

        rev_str = format_rev_string(row)
        upside_disp = min(row['Upside_Base_%'], 100)
        upside_flag = " ⚠️" if row['Upside_Base_%'] > 100 else ""

        wa_msg += (
            f"{medal} *{row['Ticker']}*{new_tag} | Q:{int(row['Quality_Score'])}/8\n"
            f"• {row['Sector']} | Ums-Wachstum: {row['RevenueGrowth_%']:.0f}% | Upside: {upside_disp:.0f}%{upside_flag}\n"
            f"• Umsätze: {rev_str}\n\n"
        )

    wa_msg += f"Neu: {', '.join(new_candidates) if new_candidates else 'Keine'}\n"
    wa_msg += f"Raus: {', '.join(dropped_candidates) if dropped_candidates else 'Keine'}\n"

    try:
        url = (
            f"https://api.callmebot.com/whatsapp.php?phone={phone}&text="
            f"{requests.utils.quote(wa_msg)}&apikey={apikey}"
        )
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            log.info("WhatsApp-Nachricht erfolgreich gesendet.")
        else:
            log.error(f"WhatsApp-Fehler: {r.text}")
    except Exception as e:
        log.exception(f"WhatsApp-Ausnahme: {e}")


def evaluate_and_report():
    if not os.path.exists(CSV_FILENAME):
        log.warning("Keine Ergebnisse in der CSV gefunden.")
        return

    try:
        df = pd.read_csv(CSV_FILENAME)
    except pd.errors.ParserError:
        log.warning("CSV hat inkonsistente Spalten – nutze robusten Parser")
        df = pd.read_csv(CSV_FILENAME, engine="python", on_bad_lines="skip")

    if df.empty:
        log.warning("Ergebnissatz ist leer.")
        return

    ranked_df = df.sort_values(by="Composite_Score", ascending=False).drop_duplicates(subset=["Ticker"]).reset_index(drop=True)

    top_5 = select_top5_with_sector_cap(ranked_df, MAX_PER_SECTOR_TOP5, TOP5_SIZE, MIN_UPSIDE_TOP5)
    if top_5.empty:
        log.warning("Keine Top-5-Kandidaten ermittelbar.")
        return

    old_top_5 = {}
    if os.path.exists(HISTORY_FILENAME):
        try:
            with open(HISTORY_FILENAME, "r", encoding="utf-8") as f:
                old_top_5 = json.load(f)
        except Exception:
            pass

    current_top_5 = {row["Ticker"]: idx + 1 for idx, row in top_5.iterrows()}
    new_candidates = [t for t in current_top_5 if t not in old_top_5]
    dropped_candidates = [t for t in old_top_5 if t not in current_top_5]
    rank_changes = [(t, old_top_5[t], r) for t, r in current_top_5.items() if t in old_top_5 and old_top_5[t] != r]

    fg_score, fg_rating = get_fear_and_greed_index()

    report = "🏆 *TOP 5 AKTIEN-SCREENER (OPTION B+)* 🏆\n\n"
    if fg_score is not None:
        report += f"🌐 *Marktstimmung (Fear & Greed):* *{fg_score} / 100* ({fg_rating})\n\n"
    else:
        report += "🌐 *Marktstimmung (Fear & Greed):* N/A\n\n"

    medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
    for i, row in top_5.iterrows():
        rank = i + 1
        medal = medals[rank - 1] if rank <= len(medals) else f"{rank}."
        new_tag = " 🆕 *[NEU]*" if row["Ticker"] in new_candidates else ""

        report += f"{medal} *{row['Ticker']}* ({row['Name']}){new_tag}\n"
        report += f"• Sektor: {row['Sector']} | Score: {row['Composite_Score']:.1f} | Qualität: {int(row['Quality_Score'])}/8\n"
        report += f"• Umsatzw.: {row['RevenueGrowth_%']:.1f}% | Marge: {row['NetMargin_%']:.2f}%\n"

        rev_str = format_rev_string(row)
        report += f"• Umsätze: {rev_str}\n"

        upside_disp = min(row['Upside_Base_%'], 100)
        upside_flag = " ⚠️ (Modell-Extrem)" if row['Upside_Base_%'] > 100 else ""
        report += f"• Fair Value (Base): {row['Fair_Value_Base']} | Upside: {upside_disp:.0f}%{upside_flag}"
        if pd.notna(row.get("PEG")):
            report += f" | PEG: {row['PEG']}"
        report += "\n\n"

    report += "🔄 *ÄNDERUNGEN:*\n"
    report += f"• 🟢 *Neu:* {', '.join(new_candidates) if new_candidates else 'Keine'}\n"
    report += f"• 🔴 *Raus:* {', '.join(dropped_candidates) if dropped_candidates else 'Keine'}\n"
    if rank_changes:
        changes_str = ", ".join([f"{t} ({o}➔{n})" for t, o, n in rank_changes])
        report += f"• 📊 *Ranking-Änderungen:* {changes_str}\n"
    else:
        report += "• 📊 *Ranking-Änderungen:* Keine\n"

    print("\n" + "=" * 50)
    print(report)
    print("=" * 50 + "\n")

    try:
        with open(HISTORY_FILENAME, "w", encoding="utf-8") as f:
            json.dump(current_top_5, f, ensure_ascii=False, indent=4)
    except Exception as e:
        log.error(f"Fehler beim Speichern der Historie: {e}")

    send_telegram_message(report)
    send_whatsapp_message(top_5, fg_score, fg_rating, new_candidates, dropped_candidates, rank_changes)


# ==========================================
# MAIN
# ==========================================
def main():
    log.info("1. Starte Laden der kombinierten Ticker-Liste (S&P 500 & Nasdaq 100)...")
    tickers = fetch_combined_tickers()
    if not tickers:
        log.error("Keine Ticker geladen. Abbruch.")
        return

    if os.path.exists(CSV_FILENAME):
        try:
            os.remove(CSV_FILENAME)
        except OSError:
            pass

    batches = [tickers[i: i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    log.info(f"2. Starte Analyse von {len(tickers)} Titeln in {len(batches)} Batches...")

    start_time = time.time()
    processed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_batch, batch) for batch in batches]
        for future in as_completed(futures):
            processed += 1
            if processed % 5 == 0 or processed == len(batches):
                elapsed = time.time() - start_time
                avg = elapsed / processed
                rem = (len(batches) - processed) * avg
                log.info(f"Fortschritt: {processed}/{len(batches)} | Restzeit: ca. {int(rem // 60)}m {int(rem % 60)}s")

    log.info("3. Erstelle Auswertung und Berichte...")
    evaluate_and_report()
    log.info("Skript erfolgreich beendet!")


if __name__ == "__main__":
    main()
