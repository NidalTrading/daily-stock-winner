# -*- coding: utf-8 -*-
"""
Optimierter Fundamentaler Aktien-Screener (S&P 500 & Nasdaq 100)
Kombiniert robuste Datenverarbeitung, strikte 5-Jahres-Profitabilitätsfilter,
Fair-Value-Szenarien und automatisierte Benachrichtigungen (Telegram & WhatsApp)
inklusive der 5-Jahres-Umsatzhistorie in allen Ausgaben.
"""

import os
import json
import time
import logging
import tempfile
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

# Filter-Einstellungen
MIN_MARKET_CAP = 2_000_000_000  # 2 Mrd. USD
MAX_DIST_FROM_HIGH = 0.35       # Max. Abstand zum 52-Wochen-Hoch

USE_5Y_PROFIT_FILTER = True     # Mind. 5 Jahre profitabel & wachsend
REQUIRED_YEARS = 5
MIN_AVAILABLE_YEARS = 4

USE_FCF_FILTER = True
FCF_MIN_YEARS = 3

SCENARIO_HORIZON_YEARS = 2
EPS_GROWTH_CAP = 0.50           # Wachstum für Projektionen deckeln

EXCLUDE_TICKERS = {"GOOG"}      # GOOGL bleibt als liquide Klasse

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


# ==========================================
# TICKER & MARKTSTIMMUNG
# ==========================================
def fetch_combined_tickers():
    headers = {"User-Agent": USER_AGENT}
    all_tickers = []

    try:
        url_sp = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        df_sp = pd.read_html(url_sp, storage_options=headers)[0]
        sp_tickers = df_sp["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
        all_tickers.extend(sp_tickers)
        log.info(f"S&P 500 Ticker geladen: {len(sp_tickers)}")
    except Exception as e:
        log.exception(f"Fehler S&P 500: {e}")

    ndx_tickers = []
    try:
        for df in pd.read_html("https://www.slickcharts.com/nasdaq100", storage_options=headers):
            if "Symbol" in df.columns:
                ndx_tickers = df["Symbol"].dropna().astype(str).str.replace(".", "-", regex=False).tolist()
                if len(ndx_tickers) > 50:
                    break
    except Exception as e:
        log.warning(f"Slickcharts Hinweis: {e}")

    if len(ndx_tickers) < 50:
        try:
            api_url = "https://api.nasdaq.com/api/quote/list-type/nasdaq100"
            res = requests.get(api_url, headers=headers, timeout=10)
            if res.status_code == 200:
                rows = res.json().get("data", {}).get("data", {}).get("rows", [])
                ndx_tickers = [r["symbol"].strip().replace(".", "-") for r in rows if "symbol" in r]
        except Exception as e:
            log.warning(f"Nasdaq API Hinweis: {e}")

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
    """Kernregel: Keine Verlustjahre in den letzten 5 Jahren & wachsender Trend + Umsatz-Historie."""
    try:
        income = getattr(t_obj, "income_stmt", None)
        if income is None or income.empty:
            return False, "keine GuV-Daten", {}

        # Korrigiert: Schließende Klammer war hier fehlerhaft
        net_income = row_value(income, ["Net Income", "Net Income Common Stockholders"])
        revenue = row_value(income, ["Total Revenue", "Operating Revenue"])

        if len(net_income) < MIN_AVAILABLE_YEARS or len(revenue) < MIN_AVAILABLE_YEARS:
            return False, "zu wenig Historie", {}

        ni_tail = net_income.tail(REQUIRED_YEARS)
        rev_tail = revenue.tail(REQUIRED_YEARS)

        ni_vals = ni_tail.values
        rev_vals = rev_tail.values

        # 1. Kein Verlustjahr erlaubt
        if any(v <= 0 for v in ni_vals):
            return False, "Verlustjahr in Historie", {}

        # 2. Langfristiges Wachstum (letzter Wert > erster Wert)
        if ni_vals[-1] <= ni_vals[0] or rev_vals[-1] <= rev_vals[0]:
            return False, "Gewinn oder Umsatz nicht gewachsen", {}

        # Umsätze der letzten Jahre extrahieren
        rev_dict = {}
        for idx, val in rev_tail.items():
            year_str = str(idx.year) if hasattr(idx, "year") else str(idx)[:4]
            rev_dict[f"Rev_{year_str}"] = val

        return True, "profitabel & wachsend", rev_dict
    except Exception:
        return False, "Fehler bei GuV-Prüfung", {}


def check_multi_year_fcf(t_obj, sector):
    if sector == "Financial Services":
        return True, "Financials: FCF N/A"
    try:
        cf = getattr(t_obj, "cashflow", None)
        fcf_series = row_value(cf, ["Free Cash Flow"])
        if fcf_series.empty or len(fcf_series) < FCF_MIN_YEARS:
            return False, "zu wenig FCF-Historie"
        if any(v <= 0 for v in fcf_series.tail(FCF_MIN_YEARS).values):
            return False, "negativer FCF"
        return True, "positiver FCF"
    except Exception:
        return False, "Fehler bei FCF-Prüfung"


# ==========================================
# BEWERTUNG & SCENARIOS
# ==========================================
def calculate_scenarios(info, current_price, forward_eps):
    growth = info.get("earningsGrowth") or info.get("revenueGrowth") or 0.08
    growth = min(max(safe_float(growth, 0.08), -0.20), EPS_GROWTH_CAP)

    if not forward_eps or forward_eps <= 0 or not current_price:
        return {"Fair_Value_Base": 0, "Upside_Base_%": 0, "Future_PE": 0}

    eps_2y = forward_eps * (1 + growth) * (1 + growth)
    current_forward_pe = safe_float(info.get("forwardPE"), 20.0)
    target_pe = min(max(current_forward_pe, 15.0), 30.0)

    fair_base = eps_2y * target_pe
    upside_base = (fair_base / current_price - 1) * 100
    future_pe = current_price / eps_2y if eps_2y > 0 else 0.0

    return {
        "EPS_2Y": round(eps_2y, 2),
        "Target_PE": round(target_pe, 1),
        "Fair_Value_Base": round(fair_base, 2),
        "Upside_Base_%": round(upside_base, 1),
        "Future_PE": round(future_pe, 1)
    }


# ==========================================
# ANALYSE EINZELNER TICKER
# ==========================================
def analyze_ticker(ticker):
    t_obj = yf.Ticker(ticker)
    info = t_obj.info
    if not info or not isinstance(info, dict):
        return None

    current_price = safe_float(info.get("currentPrice", info.get("regularMarketPrice")))
    market_cap = safe_float(info.get("marketCap"))
    fifty_two_high = safe_float(info.get("fiftyTwoWeekHigh"))

    if not current_price or not market_cap or market_cap < MIN_MARKET_CAP:
        return None

    if fifty_two_high and fifty_two_high > 0:
        if current_price < (fifty_two_high * (1.0 - MAX_DIST_FROM_HIGH)):
            return None, "zu weit unter 52W-High"

    sector = info.get("sector", "")

    rev_history_dict = {}
    if USE_5Y_PROFIT_FILTER:
        passed, reason, rev_history_dict = check_5y_profit_and_growth(t_obj)
        if not passed:
            return None, reason

    if USE_FCF_FILTER:
        passed, reason = check_multi_year_fcf(t_obj, sector)
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

    scen = calculate_scenarios(info, current_price, forward_eps)

    composite_score = (
        (quality_score * 15.0)
        + (rev_growth * 100 * 0.5)
        + (profit_margin * 100 * 0.3)
        + (roe * 100 * 0.2)
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
        "Quality_Score": quality_score,
        "Composite_Score": round(composite_score, 1),
        "Fair_Value_Base": scen["Fair_Value_Base"],
        "Upside_Base_%": scen["Upside_Base_%"],
        "Future_PE": scen["Future_PE"]
    }
    
    # Formatierte Umsätze ($B) anhängen
    for k, v in rev_history_dict.items():
        res_dict[k] = f"${v / 1e9:.2f}B" if v else "N/A"

    return res_dict, "OK"


def process_batch(ticker_batch):
    results = []
    for ticker in ticker_batch:
        time.sleep(REQUEST_DELAY)
        success = False
        for attempt in range(RETRIES):
            try:
                res, reason = analyze_ticker(ticker)
                if res:
                    results.append(res)
                success = True
                break
            except Exception as e:
                if attempt == RETRIES - 1:
                    log.error(f"Fehler bei {ticker} nach {RETRIES} Versuchen: {e}")
                else:
                    time.sleep(1.5 * (attempt + 1))
        if not success:
            continue

    if results:
        df_batch = pd.DataFrame(results)
        with csv_lock:
            header_flag = not os.path.exists(CSV_FILENAME)
            df_batch.to_csv(CSV_FILENAME, mode="a", header=header_flag, index=False, encoding="utf-8-sig")

    return len(results)


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
        
        rev_cols = [c for c in row.index if str(c).startswith("Rev_")]
        rev_str = " ➔ ".join([f"{c.replace('Rev_', '')}: {row[c]}" for c in sorted(rev_cols)]) if rev_cols else "N/A"

        wa_msg += (
            f"{medal} *{row['Ticker']}*{new_tag} | Q:{int(row['Quality_Score'])}/7\n"
            f"• Ums-Wachstum: {row['RevenueGrowth_%']:.0f}% | Upside: {row['Upside_Base_%']:.0f}%\n"
            f"• Umsätze (5J): {rev_str}\n\n"
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

    df = pd.read_csv(CSV_FILENAME)
    if df.empty:
        log.warning("Ergebnissatz ist leer.")
        return

    ranked_df = df.sort_values(by="Composite_Score", ascending=False).drop_duplicates(subset=["Ticker"]).reset_index(drop=True)
    top_5 = ranked_df.head(5)
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

    report = "🏆 *TOP 5 AKTIEN-SCREENER (OPTIMIERT)* 🏆\n\n"
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
        report += f"• Score: {row['Composite_Score']:.1f} | Qualität: {int(row['Quality_Score'])}/7\n"
        report += f"• Umsatzw.: {row['RevenueGrowth_%']:.1f}% | Marge: {row['NetMargin_%']:.2f}%\n"

        # Umsätze der letzten 5 Jahre formatiert im Bericht
        rev_cols = [c for c in row.index if str(c).startswith("Rev_")]
        if rev_cols:
            rev_str = " ➔ ".join([f"{c.replace('Rev_', '')}: {row[c]}" for c in sorted(rev_cols)])
            report += f"• Umsätze (5J): {rev_str}\n"

        report += f"• Fair Value (Base): {row['Fair_Value_Base']} | Upside: {row['Upside_Base_%']}%\n\n"

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

    batches = [tickers[i : i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
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
