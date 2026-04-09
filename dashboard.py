"""
dashboard.py — Fully Self-Contained Streamlit App
==================================================
NAB Fraud Prevention & Detection Risk Scoring System

Deploy to Streamlit Cloud by uploading just:
  - dashboard.py
  - requirements.txt

No pre-trained model or data files needed.
The model trains automatically on first load and is cached in session.

Run locally:
    pip install -r requirements.txt
    streamlit run dashboard.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import date
from typing import Dict, Any, List

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NAB | Fraud Risk Scoring",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── NAB brand palette ── */
:root {
    --nab-black:  #000000;
    --nab-red:    #D4001A;
    --nab-orange: #E8621A;
    --nab-grey:   #6D6E71;
    --nab-ltgrey: #F5F5F5;
}
/* Header */
.nab-header {
    display: flex; align-items: center; gap: 14px;
    border-bottom: 3px solid #D4001A; padding-bottom: 14px; margin-bottom: 18px;
}
.nab-wordmark {
    font-size: 1.7rem; font-weight: 900; letter-spacing: -0.02em;
    color: #000; font-family: Georgia, serif;
}
.nab-wordmark span { color: #D4001A; }
.nab-subtitle { font-size: 0.8rem; color: #6D6E71; margin-top: 2px; }
/* Score card metric tweaks */
[data-testid="stMetric"] { background: #FAFAFA; border-radius: 8px; padding: 10px 14px; border: 1px solid #E0E0E0; }
/* Concern box */
.concern-box {
    background: #F7F7F7; border-left: 4px solid #6D6E71;
    border-radius: 4px; padding: 10px 14px; margin: 4px 0;
    font-size: 0.82rem; color: #6D6E71; line-height: 1.6;
}
/* Rule box */
.rule-danger { border-left: 4px solid #D4001A; background: #FFF5F5; padding: 10px 14px; border-radius: 4px; margin: 6px 0; font-size: 0.85rem; line-height: 1.6; }
.rule-trust  { border-left: 4px solid #007A4D; background: #F0FBF5; padding: 10px 14px; border-radius: 4px; margin: 6px 0; font-size: 0.85rem; line-height: 1.6; }
/* Sidebar section headings */
.sidebar-section { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: #6D6E71; margin: 14px 0 6px; padding-bottom: 4px; border-bottom: 1px solid #E0E0E0; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# ██  SECTION 1 — DATA GENERATION
# ═══════════════════════════════════════════════════════════════

def generate_dataset(n: int = 50_000) -> pd.DataFrame:
    """
    Generates a synthetic fraud dataset.
    Concerns (PPF/PPO/IDTO/PHI etc.) do NOT affect the fraud label —
    only red flags and green flags drive the probability.
    """
    rng = np.random.default_rng(42)

    data = {}

    # Fraud green flags
    data["gps_at_address"]    = rng.binomial(1, 0.65, n)
    data["registered_device"] = rng.binomial(1, 0.70, n)
    data["registered_ip"]     = rng.binomial(1, 0.68, n)
    data["branch_device"]     = rng.binomial(1, 0.20, n)

    # Scam green flags (includes new n2n_payment)
    data["paid_before"]          = rng.binomial(1, 0.55, n)
    data["n2n_payment"]          = rng.binomial(1, 0.30, n)
    data["close_to_beneficiary"] = rng.binomial(1, 0.35, n)
    data["name_match"]           = rng.binomial(1, 0.60, n)
    data["demographic_match"]    = rng.binomial(1, 0.50, n)
    data["no_prior_fraud"]       = rng.binomial(1, 0.80, n)

    # Fraud red flags
    data["new_ip"]                   = rng.binomial(1, 0.25, n)
    data["new_device"]               = rng.binomial(1, 0.20, n)
    data["ekyc_account"]             = rng.binomial(1, 0.15, n)
    data["recently_created_account"] = rng.binomial(1, 0.18, n)
    data["multiple_device_logins"]   = rng.binomial(1, 0.12, n)
    data["fast_traveller"]           = rng.binomial(1, 0.08, n)
    data["password_reset"]           = rng.binomial(1, 0.14, n)
    data["number_change"]            = rng.binomial(1, 0.10, n)
    data["account_takeover_alert"]   = rng.binomial(1, 0.07, n)
    data["language_change"]          = rng.binomial(1, 0.09, n)

    # Scam red flags
    data["weak_name_match"]        = rng.binomial(1, 0.18, n)
    data["remote_access_session"]  = rng.binomial(1, 0.08, n)
    data["crypto_payment"]         = rng.binomial(1, 0.12, n)
    data["remitter_payment"]       = rng.binomial(1, 0.14, n)
    data["vulnerable_customer"]    = rng.binomial(1, 0.12, n)
    data["international_payment"]  = rng.binomial(1, 0.22, n)
    data["high_risk_bsb"]          = rng.binomial(1, 0.10, n)
    data["prior_victim"]           = rng.binomial(1, 0.06, n)
    data["suspicious_reference"]   = rng.binomial(1, 0.09, n)
    data["active_call"]            = rng.binomial(1, 0.07, n)
    data["gambling_activity"]      = rng.binomial(1, 0.10, n)

    # Fraud concerns — PPF/PPO/IDTO (analyst annotations — NOT used in labels)
    data["login_irregularity"] = rng.binomial(1, 0.15, n)
    data["sim_swap"]           = rng.binomial(1, 0.05, n)
    data["identity_theft"]     = rng.binomial(1, 0.06, n)

    # Scam concerns — PHI etc. (analyst annotations — NOT used in labels)
    data["phishing"]                  = rng.binomial(1, 0.08, n)
    data["investment_scam"]           = rng.binomial(1, 0.07, n)
    data["romance_scam"]              = rng.binomial(1, 0.05, n)
    data["business_email_compromise"] = rng.binomial(1, 0.06, n)
    data["goods_services_scam"]       = rng.binomial(1, 0.07, n)
    data["remote_access_scam"]        = rng.binomial(1, 0.05, n)
    data["hi_mum_scam"]               = rng.binomial(1, 0.04, n)
    data["job_scam"]                  = rng.binomial(1, 0.05, n)

    # Numerical
    data["transaction_amount"] = rng.exponential(scale=500, size=n).clip(1, 50_000).round(2)
    data["account_age_days"]   = rng.integers(0, 36_500, n)
    data["tx_count_24h"]       = rng.poisson(lam=3, size=n).clip(0, 30)
    data["hour_of_day"]        = rng.integers(0, 24, n)

    df = pd.DataFrame(data)

    # ── Label generation: ONLY red flags + green flags ──
    s = np.zeros(n)
    s += df["new_ip"] * 0.15;              s += df["new_device"] * 0.18
    s += df["ekyc_account"] * 0.12;        s += df["recently_created_account"] * 0.10
    s += df["multiple_device_logins"]*0.12; s += df["fast_traveller"] * 0.10
    s += df["password_reset"] * 0.12;      s += df["number_change"] * 0.10
    s += df["account_takeover_alert"]*0.25; s += df["language_change"] * 0.08
    s += df["weak_name_match"] * 0.08;     s += df["remote_access_session"] * 0.18
    s += df["crypto_payment"] * 0.12;      s += df["remitter_payment"] * 0.10
    s += df["vulnerable_customer"]*0.12;   s += df["international_payment"] * 0.08
    s += df["high_risk_bsb"] * 0.12;      s += df["prior_victim"] * 0.18
    s += df["suspicious_reference"]*0.12;  s += df["active_call"] * 0.15
    s += df["gambling_activity"] * 0.08
    # Green flags reduce risk
    s -= df["gps_at_address"]*0.12;    s -= df["registered_device"]*0.15
    s -= df["registered_ip"]*0.12;     s -= df["branch_device"]*0.10
    s -= df["paid_before"]*0.10;       s -= df["name_match"]*0.08
    s -= df["no_prior_fraud"]*0.12;    s -= df["n2n_payment"]*0.10
    # Combination bonuses
    s += (df["new_device"] & df["new_ip"] & df["password_reset"]) * 0.30
    s += (df["remote_access_session"] & df["active_call"] & df["international_payment"]) * 0.35
    s += (df["crypto_payment"] & df["suspicious_reference"] & df["prior_victim"]) * 0.35
    s += (df["number_change"] & df["account_takeover_alert"]) * 0.20
    s += (df["high_risk_bsb"] & df["remitter_payment"] & df["weak_name_match"]) * 0.25
    s += (df["fast_traveller"] & df["multiple_device_logins"]) * 0.20
    s += (df["transaction_amount"] > 5000).astype(int) * 0.08
    s += (df["account_age_days"] < 30).astype(int) * 0.12
    s += rng.normal(0, 0.05, n)
    s = s.clip(0, 1)

    df["is_fraud"]        = (rng.uniform(0, 1, n) < s * 0.35).astype(int)
    df["raw_fraud_score"] = s
    return df


# ═══════════════════════════════════════════════════════════════
# ██  SECTION 2 — FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════

FRAUD_RED_FLAGS = [
    "new_ip", "new_device", "ekyc_account", "recently_created_account",
    "multiple_device_logins", "fast_traveller", "password_reset",
    "number_change", "account_takeover_alert", "language_change",
]
GREEN_FLAGS = [
    "gps_at_address", "registered_device", "registered_ip", "branch_device",
    "paid_before", "n2n_payment", "close_to_beneficiary", "name_match",
    "demographic_match", "no_prior_fraud",
]
SCAM_RED_FLAGS = [
    "weak_name_match", "remote_access_session", "crypto_payment", "remitter_payment",
    "vulnerable_customer", "international_payment", "high_risk_bsb", "prior_victim",
    "suspicious_reference", "active_call", "gambling_activity",
]


def engineer_features_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["device_change_count"]      = df["new_device"] + df["multiple_device_logins"] + df["fast_traveller"]
    df["ip_change_frequency"]      = df["new_ip"] + df["remote_access_session"] + df["international_payment"]
    df["behavioural_risk_score"]   = df[FRAUD_RED_FLAGS].sum(axis=1)
    df["trust_score"]              = df[GREEN_FLAGS].sum(axis=1)
    df["account_takeover_pattern"] = (df["new_device"] + df["new_ip"] + df["password_reset"]
                                      + df["account_takeover_alert"] + df["number_change"])
    df["scam_pattern_score"]       = df[SCAM_RED_FLAGS].sum(axis=1)
    df["net_risk_score"]           = df["behavioural_risk_score"] + df["scam_pattern_score"] - df["trust_score"]
    return df


def engineer_single(t: dict) -> dict:
    g = lambda k: int(t.get(k, 0))
    t["device_change_count"]      = g("new_device") + g("multiple_device_logins") + g("fast_traveller")
    t["ip_change_frequency"]      = g("new_ip") + g("remote_access_session") + g("international_payment")
    t["behavioural_risk_score"]   = sum(g(f) for f in FRAUD_RED_FLAGS)
    t["trust_score"]              = sum(g(f) for f in GREEN_FLAGS)
    t["account_takeover_pattern"] = g("new_device")+g("new_ip")+g("password_reset")+g("account_takeover_alert")+g("number_change")
    t["scam_pattern_score"]       = sum(g(f) for f in SCAM_RED_FLAGS)
    t["net_risk_score"]           = t["behavioural_risk_score"] + t["scam_pattern_score"] - t["trust_score"]
    return t


# ═══════════════════════════════════════════════════════════════
# ██  SECTION 3 — MODEL TRAINING
# ═══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def train_model():
    """Generates data and trains a Random Forest. Cached so it only runs once per session."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    df = generate_dataset(n=50_000)
    df = engineer_features_df(df)
    df = df.drop(columns=["raw_fraud_score"], errors="ignore")

    feature_cols = (
        FRAUD_RED_FLAGS + SCAM_RED_FLAGS + GREEN_FLAGS +
        ["login_irregularity", "sim_swap", "identity_theft",
         "phishing", "investment_scam", "romance_scam", "business_email_compromise",
         "goods_services_scam", "remote_access_scam", "hi_mum_scam", "job_scam"] +
        ["transaction_amount", "account_age_days", "tx_count_24h", "hour_of_day",
         "device_change_count", "ip_change_frequency", "behavioural_risk_score",
         "trust_score", "account_takeover_pattern", "scam_pattern_score", "net_risk_score"]
    )
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy()
    y = df["is_fraud"]

    # Scale numerical columns
    num_cols = ["transaction_amount", "account_age_days", "tx_count_24h", "hour_of_day",
                "device_change_count", "ip_change_frequency", "behavioural_risk_score",
                "trust_score", "account_takeover_pattern", "scam_pattern_score", "net_risk_score"]
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_leaf=20,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Feature importance dataframe for chart
    imp_df = pd.DataFrame({
        "feature":    feature_cols,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    # Class distribution for chart
    class_dist = y.value_counts().to_dict()

    return clf, feature_cols, scaler, num_cols, imp_df, class_dist, X_test, y_test


# ═══════════════════════════════════════════════════════════════
# ██  SECTION 4 — RULES ENGINE
# ═══════════════════════════════════════════════════════════════

FRAUD_RULES = [
    {"name": "ATO Pattern: New Device + New IP + Password Reset",
     "check": lambda t: t.get("new_device") and t.get("new_ip") and t.get("password_reset"),
     "score": 0.35,
     "explanation": "A new device, new IP address, and a recent password reset detected together — strong Account Takeover (ATO) indicator. A fraudster may have gained access by resetting the password and logging in from an unrecognised device."},
    {"name": "Account Takeover Alert Triggered",
     "check": lambda t: t.get("account_takeover_alert"),
     "score": 0.30,
     "explanation": "An internal ATO alert has been raised. Behavioural monitoring has flagged access patterns inconsistent with the customer's normal behaviour."},
    {"name": "New Device + New IP + ATO Alert",
     "check": lambda t: t.get("new_device") and t.get("new_ip") and t.get("account_takeover_alert"),
     "score": 0.35,
     "explanation": "New device and new IP combined with an active ATO alert — high-confidence account takeover signal."},
    {"name": "eKYC Account + Recently Created",
     "check": lambda t: t.get("ekyc_account") and t.get("recently_created_account"),
     "score": 0.20,
     "explanation": "Digitally-created eKYC account that was recently opened. Fraudsters frequently open new accounts using stolen identity documents as money mule accounts."},
    {"name": "Fast Traveller + Multiple Device Logins",
     "check": lambda t: t.get("fast_traveller") and t.get("multiple_device_logins"),
     "score": 0.20,
     "explanation": "Impossible travel signs combined with multiple device logins — credentials may be compromised and in use simultaneously by the customer and a fraudster."},
    {"name": "Language Change + Password Reset",
     "check": lambda t: t.get("language_change") and t.get("password_reset"),
     "score": 0.18,
     "explanation": "Language setting changed alongside a password reset. Fraudsters sometimes change language to prevent customers from understanding security alert messages."},
    {"name": "Number Change + ATO Alert",
     "check": lambda t: t.get("number_change") and t.get("account_takeover_alert"),
     "score": 0.25,
     "explanation": "Phone number change combined with an active ATO alert — attacker may be updating contact details to intercept authentication messages."},
]

SCAM_RULES = [
    {"name": "Remote Access Session + Active Call + International Transfer",
     "check": lambda t: t.get("remote_access_session") and t.get("active_call") and t.get("international_payment"),
     "score": 0.40,
     "explanation": "Remote access session active, customer on a phone call, payment going overseas — hallmark of a remote access scam where a fraudster posing as tech support directs an international transfer."},
    {"name": "Crypto Payment + Suspicious Reference + Prior Victim",
     "check": lambda t: t.get("crypto_payment") and t.get("suspicious_reference") and t.get("prior_victim"),
     "score": 0.40,
     "explanation": "Crypto payment with suspicious reference by a prior scam victim. Crypto is irreversible — repeat victimisation is a serious concern."},
    {"name": "High-Risk BSB + Remitter Payment + Weak Name Match",
     "check": lambda t: t.get("high_risk_bsb") and t.get("remitter_payment") and t.get("weak_name_match"),
     "score": 0.30,
     "explanation": "High-risk BSB, remitter (third-party) payment, payee name only partially matching — suggests a mule account or fraudulent business."},
    {"name": "Vulnerable Customer + Active Call + Remote Access Session",
     "check": lambda t: t.get("vulnerable_customer") and t.get("active_call") and t.get("remote_access_session"),
     "score": 0.45,
     "explanation": "Vulnerable customer on a call while a remote access session runs — extremely high-risk. Vulnerable customers are disproportionately targeted by remote access scammers posing as bank staff."},
    {"name": "Prior Victim — Elevated Re-targeting Risk",
     "check": lambda t: t.get("prior_victim"),
     "score": 0.18,
     "explanation": "Customer has previously been a scam victim. Prior victims are more likely to be re-targeted and their details may have been shared on scam network lists."},
    {"name": "Crypto Payment + International Transfer",
     "check": lambda t: t.get("crypto_payment") and t.get("international_payment"),
     "score": 0.22,
     "explanation": "Crypto payment going overseas — common in investment scams where funds are directed offshore and virtually impossible to recover."},
    {"name": "Suspicious Reference + Remitter Payment",
     "check": lambda t: t.get("suspicious_reference") and t.get("remitter_payment"),
     "score": 0.18,
     "explanation": "Suspicious payment reference with a remitter (third-party) payment — raises concerns about the legitimacy of the beneficiary."},
]

TRUST_RULES = [
    {"name": "Registered Device + Registered IP + Prior Payment — Lower Risk",
     "check": lambda t: t.get("registered_device") and t.get("registered_ip") and t.get("paid_before"),
     "score": -0.20,
     "explanation": "Transaction from a registered device, known IP, and the customer has paid this beneficiary before — strong combination of trust signals."},
    {"name": "Branch Device or GPS at Address — Positive Trust Indicator",
     "check": lambda t: t.get("branch_device") or t.get("gps_at_address"),
     "score": -0.15,
     "explanation": "Transaction from a bank branch device or GPS matches registered address — physical presence at a known location significantly reduces remote fraud risk."},
    {"name": "N2N Payment — Name-to-Name Trust Signal",
     "check": lambda t: t.get("n2n_payment"),
     "score": -0.12,
     "explanation": "Name-to-Name (N2N) payment — sending and receiving account names match. N2N payments carry lower scam risk as the customer transacts with a known named individual."},
]

CONCERN_RULES = [
    {"name": "⚠️ PPO — SIM Swap / Portable Number Order",
     "check": lambda t: t.get("sim_swap"),
     "explanation": "ANALYST ANNOTATION — No score change. PPO (Portable Number Order / SIM swap) concern noted. Customer's phone may have been ported to a new SIM without their knowledge — often a precursor to ATO."},
    {"name": "⚠️ PPF — Login Irregularity",
     "check": lambda t: t.get("login_irregularity"),
     "explanation": "ANALYST ANNOTATION — No score change. PPF (Prohibited Pattern Flag — login irregularity). Review recent login history for unusual times, locations, or failure patterns."},
    {"name": "⚠️ IDTO — Identity Takeover",
     "check": lambda t: t.get("identity_theft"),
     "explanation": "ANALYST ANNOTATION — No score change. IDTO (Identity Takeover) concern. Consider whether customer's personal details may have been compromised."},
    {"name": "⚠️ PHI — Phishing Concern",
     "check": lambda t: t.get("phishing"),
     "explanation": "ANALYST ANNOTATION — No score change. PHI (Phishing) concern. Customer may have been directed to a fraudulent site and entered credentials under false pretences."},
    {"name": "⚠️ Investment Scam Concern",
     "check": lambda t: t.get("investment_scam"),
     "explanation": "ANALYST ANNOTATION — No score change. Investment scam suspected. Customer may have been contacted by a fake broker. Consider welfare call before processing."},
    {"name": "⚠️ Romance Scam Concern",
     "check": lambda t: t.get("romance_scam"),
     "explanation": "ANALYST ANNOTATION — No score change. Romance scam suspected. Fraudster may have built a fake online relationship before requesting money."},
    {"name": "⚠️ Business Email Compromise (BEC) Concern",
     "check": lambda t: t.get("business_email_compromise"),
     "explanation": "ANALYST ANNOTATION — No score change. BEC suspected. Fraudulent email may have impersonated a supplier/executive to redirect payment. Verify payee with known contacts."},
    {"name": "⚠️ Goods & Services Scam Concern",
     "check": lambda t: t.get("goods_services_scam"),
     "explanation": "ANALYST ANNOTATION — No score change. Goods/services scam suspected. Customer may be paying for items that will not be delivered."},
    {"name": "⚠️ Remote Access Scam Concern",
     "check": lambda t: t.get("remote_access_scam"),
     "explanation": "ANALYST ANNOTATION — No score change. Remote access scam context noted. Customer may have allowed scammer device access under guise of tech support."},
    {"name": "⚠️ Hi Mum / Hi Dad Scam Concern",
     "check": lambda t: t.get("hi_mum_scam"),
     "explanation": "ANALYST ANNOTATION — No score change. Family impersonation ('Hi Mum / Hi Dad') scam suspected. Fraudster may be posing as a family member in distress."},
    {"name": "⚠️ Job Scam Concern",
     "check": lambda t: t.get("job_scam"),
     "explanation": "ANALYST ANNOTATION — No score change. Job scam noted. Customer may have responded to a fake job offer involving receiving and forwarding funds."},
]


def run_rules(t: dict) -> dict:
    scoring, scoring_exp = [], []
    concern, concern_exp = [], []
    total = 0.0
    for rule in FRAUD_RULES + SCAM_RULES + TRUST_RULES:
        try:
            if rule["check"](t):
                scoring.append(rule["name"])
                scoring_exp.append(rule["explanation"])
                total += rule["score"]
        except Exception:
            pass
    for rule in CONCERN_RULES:
        try:
            if rule["check"](t):
                concern.append(rule["name"])
                concern_exp.append(rule["explanation"])
        except Exception:
            pass
    rule_score = float(np.clip(total, 0, 1))
    return {
        "triggered_rules":      scoring,
        "rule_explanations":    scoring_exp,
        "analyst_notes":        concern,
        "analyst_explanations": concern_exp,
        "rule_score":           round(rule_score, 4),
    }


# ═══════════════════════════════════════════════════════════════
# ██  SECTION 5 — SCORING HELPERS
# ═══════════════════════════════════════════════════════════════

def classify_risk(score: float) -> str:
    if score < 0.25: return "LOW"
    if score < 0.50: return "MEDIUM"
    if score < 0.75: return "HIGH"
    return "CRITICAL"

def map_rec(score: float) -> str:
    if score >= 0.70: return "BLOCK"
    if score >= 0.45: return "ESCALATE"
    if score >= 0.25: return "REVIEW"
    return "APPROVE"

def combine(ml: float, rules: float) -> float:
    return round(float(np.clip(ml * 0.6 + rules * 0.4, 0, 1)), 4)

def fmt_name(k: str) -> str:
    LABELS = {
        "close_to_beneficiary": "Close to BSB",
        "n2n_payment": "N2N Pmt",
        "login_irregularity": "PPF",
        "sim_swap": "PPO",
        "identity_theft": "IDTO",
        "phishing": "PHI",
    }
    return LABELS.get(k, k.replace("_", " ").title())

def days_from_date(opened: date) -> int:
    return max(0, (date.today() - opened).days)

def format_age(days: int) -> str:
    if days < 30:   return f"{days} days"
    if days < 365:  return f"{days // 30} months ({days:,} days)"
    yrs = days // 365
    mo  = (days % 365) // 30
    mo_str = f", {mo} mo" if mo else ""
    return f"{yrs} yr{'s' if yrs > 1 else ''}{mo_str} ({days:,} days)"

def get_top_factors(feature_names, feature_values, importances, top_n=5):
    contribs = importances * np.abs(feature_values)
    idx = np.argsort(contribs)[::-1][:top_n]
    return [feature_names[i] for i in idx if contribs[i] > 0]


# ═══════════════════════════════════════════════════════════════
# ██  SECTION 6 — TRAIN ON STARTUP
# ═══════════════════════════════════════════════════════════════

with st.spinner("🔄 Training model on 50,000 synthetic transactions — this takes about 30 seconds on first load…"):
    clf, feature_cols, scaler, num_cols, imp_df, class_dist, X_test, y_test = train_model()


# ═══════════════════════════════════════════════════════════════
# ██  SECTION 7 — PAGE HEADER
# ═══════════════════════════════════════════════════════════════

st.markdown("""
<div class="nab-header">
  <svg width="42" height="42" viewBox="0 0 42 42" xmlns="http://www.w3.org/2000/svg">
    <rect width="42" height="42" fill="#000"/>
    <polygon points="21,4 24.5,14.5 36,14.5 26.5,21 30,31.5 21,25 12,31.5 15.5,21 6,14.5 17.5,14.5"
             fill="#D4001A"/>
    <line x1="16" y1="12" x2="26" y2="12" stroke="white" stroke-width="1.6"/>
    <line x1="15" y1="15.5" x2="27" y2="15.5" stroke="white" stroke-width="1.6"/>
  </svg>
  <div>
    <div class="nab-wordmark">n<span>a</span>b</div>
    <div class="nab-subtitle">Fraud Prevention & Detection · Risk Scoring System</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
Combines a **Random Forest ML model** (trained on 50,000 synthetic transactions)
with a **bank-style rules engine**. Toggle signals in the sidebar — scores update live.
> **Fraud/Scam Concerns** (PPO, PPF, IDTO, PHI etc.) are **analyst annotations** — they
> appear in the notes section below but do **not** affect the risk score.
""")


# ═══════════════════════════════════════════════════════════════
# ██  SECTION 8 — SIDEBAR INPUTS
# ═══════════════════════════════════════════════════════════════

sb = st.sidebar
sb.markdown("## ⚙️ Transaction Features")

sb.markdown('<div class="sidebar-section">🟢 Fraud Green Flags</div>', unsafe_allow_html=True)
gps_at_address    = sb.checkbox("GPS at Address",    value=False)
registered_device = sb.checkbox("Registered Device", value=True)
registered_ip     = sb.checkbox("Registered IP",     value=True)
branch_device     = sb.checkbox("Branch Device",     value=False)

sb.markdown('<div class="sidebar-section">🟢 Scam Green Flags</div>', unsafe_allow_html=True)
paid_before          = sb.checkbox("Paid Before",       value=False)
n2n_payment          = sb.checkbox("N2N Pmt",           value=False, help="Name-to-Name payment — trust signal")
close_to_beneficiary = sb.checkbox("Close to BSB",      value=False)
name_match           = sb.checkbox("Name Match",        value=False)
demographic_match    = sb.checkbox("Demographic Match", value=False)
no_prior_fraud       = sb.checkbox("No Prior Fraud",    value=True)

sb.markdown('<div class="sidebar-section">🔴 Fraud Red Flags</div>', unsafe_allow_html=True)
new_ip                   = sb.checkbox("New IP",                   value=False)
new_device               = sb.checkbox("New Device",               value=False)
ekyc_account             = sb.checkbox("eKYC Account",             value=False)
recently_created_account = sb.checkbox("Recently Created Account", value=False)
multiple_device_logins   = sb.checkbox("Multiple Device Logins",   value=False)
fast_traveller           = sb.checkbox("Fast Traveller",           value=False)
password_reset           = sb.checkbox("Password Reset",           value=False)
number_change            = sb.checkbox("Number Change",            value=False)
account_takeover_alert   = sb.checkbox("Account Takeover Alert",   value=False)
language_change          = sb.checkbox("Language Change",          value=False)

sb.markdown('<div class="sidebar-section">🔴 Scam Red Flags</div>', unsafe_allow_html=True)
weak_name_match       = sb.checkbox("Weak Name Match",        value=False)
remote_access_session = sb.checkbox("Remote Access Session",  value=False)
crypto_payment        = sb.checkbox("Crypto Payment",         value=False)
remitter_payment      = sb.checkbox("Remitter Payment",       value=False)
vulnerable_customer   = sb.checkbox("Vulnerable Customer",    value=False)
international_payment = sb.checkbox("International Payment",  value=False)
high_risk_bsb         = sb.checkbox("High Risk BSB",          value=False)
prior_victim          = sb.checkbox("Prior Victim",           value=False)
suspicious_reference  = sb.checkbox("Suspicious Reference",   value=False)
active_call           = sb.checkbox("Active Call",            value=False)
gambling_activity     = sb.checkbox("Gambling Activity",      value=False)

sb.markdown("---")
sb.markdown('<div class="sidebar-section">⚠️ Fraud Concerns <small style="font-weight:400;text-transform:none">(annotation only)</small></div>', unsafe_allow_html=True)
login_irregularity = sb.checkbox("PPF — Login Irregularity", value=False)
sim_swap           = sb.checkbox("PPO — SIM Swap",           value=False)
identity_theft     = sb.checkbox("IDTO — Identity Theft",    value=False)

sb.markdown('<div class="sidebar-section">⚠️ Scam Concerns <small style="font-weight:400;text-transform:none">(annotation only)</small></div>', unsafe_allow_html=True)
phishing                  = sb.checkbox("PHI — Phishing",            value=False)
investment_scam           = sb.checkbox("Investment Scam",           value=False)
romance_scam              = sb.checkbox("Romance Scam",              value=False)
business_email_compromise = sb.checkbox("Business Email Compromise", value=False)
goods_services_scam       = sb.checkbox("Goods & Services Scam",     value=False)
remote_access_scam        = sb.checkbox("Remote Access Scam",        value=False)
hi_mum_scam               = sb.checkbox("Hi Mum Scam",               value=False)
job_scam                  = sb.checkbox("Job Scam",                  value=False)

sb.markdown("---")
sb.markdown('<div class="sidebar-section">💰 Transaction Details</div>', unsafe_allow_html=True)
transaction_amount = sb.number_input("Transaction Amount ($)", min_value=1.0, max_value=100_000.0, value=250.0, step=50.0)

sb.markdown("**Account Opened Date**")
default_open   = date(date.today().year - 1, date.today().month, date.today().day)
account_opened = sb.date_input(
    "Account Opened Date",
    value=default_open,
    min_value=date(1920, 1, 1),
    max_value=date.today(),
    help="Supports accounts opened 60+ years ago",
    label_visibility="collapsed",
)
account_age_days = days_from_date(account_opened)
sb.caption(f"Age: **{format_age(account_age_days)}**")

tx_count_24h = sb.number_input("Transactions in 24h", min_value=0, max_value=50, value=2, step=1)
hour_of_day  = sb.slider("Hour of Day", 0, 23, 12)

# Preset scenarios
sb.markdown("---")
sb.markdown('<div class="sidebar-section">🎯 Quick Scenarios</div>', unsafe_allow_html=True)
scenario = sb.selectbox("Load scenario:", [
    "— Custom —",
    "🔴 Account Takeover (ATO)",
    "🔴 Remote Access Scam",
    "🔴 Crypto Investment Scam",
    "🟢 Low Risk — Trusted Customer",
    "🟡 Medium Risk",
])


# ═══════════════════════════════════════════════════════════════
# ██  SECTION 9 — BUILD INPUT DICT
# ═══════════════════════════════════════════════════════════════

raw = {
    "gps_at_address": int(gps_at_address), "registered_device": int(registered_device),
    "registered_ip": int(registered_ip),   "branch_device": int(branch_device),
    "paid_before": int(paid_before),        "n2n_payment": int(n2n_payment),
    "close_to_beneficiary": int(close_to_beneficiary), "name_match": int(name_match),
    "demographic_match": int(demographic_match),       "no_prior_fraud": int(no_prior_fraud),
    "new_ip": int(new_ip),               "new_device": int(new_device),
    "ekyc_account": int(ekyc_account),   "recently_created_account": int(recently_created_account),
    "multiple_device_logins": int(multiple_device_logins), "fast_traveller": int(fast_traveller),
    "password_reset": int(password_reset), "number_change": int(number_change),
    "account_takeover_alert": int(account_takeover_alert), "language_change": int(language_change),
    "weak_name_match": int(weak_name_match), "remote_access_session": int(remote_access_session),
    "crypto_payment": int(crypto_payment),  "remitter_payment": int(remitter_payment),
    "vulnerable_customer": int(vulnerable_customer), "international_payment": int(international_payment),
    "high_risk_bsb": int(high_risk_bsb),   "prior_victim": int(prior_victim),
    "suspicious_reference": int(suspicious_reference), "active_call": int(active_call),
    "gambling_activity": int(gambling_activity),
    "login_irregularity": int(login_irregularity), "sim_swap": int(sim_swap),
    "identity_theft": int(identity_theft), "phishing": int(phishing),
    "investment_scam": int(investment_scam), "romance_scam": int(romance_scam),
    "business_email_compromise": int(business_email_compromise),
    "goods_services_scam": int(goods_services_scam), "remote_access_scam": int(remote_access_scam),
    "hi_mum_scam": int(hi_mum_scam), "job_scam": int(job_scam),
    "transaction_amount": float(transaction_amount),
    "account_age_days": int(account_age_days),
    "tx_count_24h": int(tx_count_24h),
    "hour_of_day": int(hour_of_day),
}

# Apply preset scenario overrides
SCENARIOS = {
    "🔴 Account Takeover (ATO)": {
        "new_device":1,"new_ip":1,"password_reset":1,"account_takeover_alert":1,
        "multiple_device_logins":1,"recently_created_account":1,"number_change":1,
        "registered_device":0,"registered_ip":0,"no_prior_fraud":0,
        "transaction_amount":9999.0,"account_age_days":14,"tx_count_24h":7,"hour_of_day":2,
    },
    "🔴 Remote Access Scam": {
        "remote_access_session":1,"active_call":1,"international_payment":1,
        "vulnerable_customer":1,"suspicious_reference":1,
        "transaction_amount":15000.0,"account_age_days":400,"tx_count_24h":1,"hour_of_day":14,
    },
    "🔴 Crypto Investment Scam": {
        "crypto_payment":1,"suspicious_reference":1,"prior_victim":1,
        "international_payment":1,"weak_name_match":1,
        "transaction_amount":25000.0,"account_age_days":200,"tx_count_24h":3,"hour_of_day":20,
    },
    "🟢 Low Risk — Trusted Customer": {
        "gps_at_address":1,"registered_device":1,"registered_ip":1,
        "paid_before":1,"name_match":1,"no_prior_fraud":1,"n2n_payment":1,
        "transaction_amount":120.0,"account_age_days":1825,"tx_count_24h":2,"hour_of_day":11,
    },
    "🟡 Medium Risk": {
        "new_device":1,"new_ip":1,
        "transaction_amount":2000.0,"account_age_days":60,"tx_count_24h":4,"hour_of_day":22,
    },
}
if scenario != "— Custom —" and scenario in SCENARIOS:
    raw.update(SCENARIOS[scenario])

inp = engineer_single(dict(raw))


# ═══════════════════════════════════════════════════════════════
# ██  SECTION 10 — PREDICT
# ═══════════════════════════════════════════════════════════════

# Build scaled feature vector
fv_raw = pd.DataFrame([[inp.get(f, 0) for f in feature_cols]], columns=feature_cols)
fv_raw[num_cols] = scaler.transform(fv_raw[num_cols])
fv = fv_raw.values[0]

ml_score    = float(clf.predict_proba(fv_raw)[0][1])
rule_result = run_rules(inp)
rule_score  = rule_result["rule_score"]
fraud_score = combine(ml_score, rule_score)
risk_level  = classify_risk(fraud_score)
rec         = map_rec(fraud_score)

importances  = clf.feature_importances_
top_factors  = get_top_factors(feature_cols, fv, importances, top_n=5)

analyst_notes = rule_result.get("analyst_notes", [])
analyst_exps  = rule_result.get("analyst_explanations", [])


# ═══════════════════════════════════════════════════════════════
# ██  SECTION 11 — SCORE CARDS
# ═══════════════════════════════════════════════════════════════

st.markdown("## 📊 Risk Assessment")

c1, c2, c3, c4 = st.columns(4)
c1.metric("🤖 ML Score",       f"{ml_score:.1%}",    help="Raw probability from Random Forest")
c2.metric("📋 Rule Score",     f"{rule_score:.1%}",  help="Score from bank-style rules engine (concerns excluded)")
c3.metric("⚖️ Combined Score", f"{fraud_score:.1%}", help="Weighted average: 60% ML + 40% Rules")
c4.metric("🎯 Risk Level",     risk_level)

# Recommendation banner
REC_CFG = {
    "APPROVE":  ("✅", st.success, "Transaction cleared — no significant fraud signals detected."),
    "REVIEW":   ("🔍", st.info,    "Some risk signals present — manual review recommended before processing."),
    "ESCALATE": ("⚠️", st.warning, "Multiple risk indicators detected — escalate to fraud operations team."),
    "BLOCK":    ("🚫", st.error,   "High-risk pattern detected — block this transaction and contact the customer."),
}
icon, banner_fn, banner_msg = REC_CFG[rec]
banner_fn(f"{icon} **{rec}** — {banner_msg}  |  Combined score: {fraud_score:.1%}  |  {risk_level}")

# Gauge
st.markdown("### 🌡️ Fraud Risk Gauge")
gauge_cols = st.columns([6, 1])
with gauge_cols[0]:
    st.progress(min(fraud_score, 1.0))
    GAUGE_LABELS = {"LOW": "🟢 LOW RISK", "MEDIUM": "🟡 MEDIUM RISK", "HIGH": "🟠 HIGH RISK", "CRITICAL": "🔴 CRITICAL RISK"}
    st.caption(f"{GAUGE_LABELS[risk_level]}  ·  Score: **{fraud_score:.2%}**  ·  Account age: **{format_age(raw['account_age_days'])}**")

st.markdown("---")


# ═══════════════════════════════════════════════════════════════
# ██  SECTION 12 — TWO-COL: BREAKDOWN + TOP FACTORS
# ═══════════════════════════════════════════════════════════════

left, right = st.columns(2)

with left:
    st.markdown("### 📊 Score Breakdown")
    bd = pd.DataFrame({
        "Source":  ["ML Model", "Rules Engine", "Combined"],
        "Weight":  ["60%", "40%", "Final"],
        "Score":   [ml_score, rule_score, fraud_score],
    })
    st.dataframe(bd.style.format({"Score": "{:.2%}"}), use_container_width=True, hide_index=True)

    # Active signals summary
    RED_KEYS = FRAUD_RED_FLAGS + SCAM_RED_FLAGS
    CONCERN_KEYS = ["login_irregularity","sim_swap","identity_theft","phishing","investment_scam",
                    "romance_scam","business_email_compromise","goods_services_scam",
                    "remote_access_scam","hi_mum_scam","job_scam"]

    active_red     = [k for k in RED_KEYS    if inp.get(k, 0)]
    active_green   = [k for k in GREEN_FLAGS if inp.get(k, 0)]
    active_concern = [k for k in CONCERN_KEYS if inp.get(k, 0)]

    if active_red:
        st.error(f"🔴 **Red flags:** {', '.join(fmt_name(k) for k in active_red)}")
    if active_green:
        st.success(f"🟢 **Green flags:** {', '.join(fmt_name(k) for k in active_green)}")
    if active_concern:
        st.info(f"📝 **Analyst annotations (no score impact):** {', '.join(fmt_name(k) for k in active_concern)}")
    if not active_red and not active_green and not active_concern:
        st.info("No signals active — using default inputs.")

with right:
    st.markdown("### 🏆 Top Risk Factors (ML Model)")
    if top_factors:
        for i, factor in enumerate(top_factors, 1):
            val  = inp.get(factor, 0)
            icon = "🔴" if val else "⚪"
            st.markdown(f"**{i}.** {icon} {fmt_name(factor)}")
    else:
        st.info("No significant active risk factors.")


st.markdown("---")


# ═══════════════════════════════════════════════════════════════
# ██  SECTION 13 — TRIGGERED RULES
# ═══════════════════════════════════════════════════════════════

st.markdown("### 📋 Rules Engine — Triggered Scoring Rules")
if not rule_result["triggered_rules"]:
    st.success("✅ No scoring rules triggered — transaction appears legitimate.")
else:
    for name, exp in zip(rule_result["triggered_rules"], rule_result["rule_explanations"]):
        is_trust = any(w in name for w in ["Lower Risk", "Trust", "N2N"])
        if is_trust:
            with st.expander(f"🟢 {name}", expanded=False):
                st.markdown(f'<div class="rule-trust">{exp}</div>', unsafe_allow_html=True)
        else:
            with st.expander(f"🔴 {name}", expanded=True):
                st.markdown(f'<div class="rule-danger">{exp}</div>', unsafe_allow_html=True)

# Analyst notes (concerns)
if analyst_notes:
    st.markdown("### 📝 Analyst Notes *(no score impact)*")
    st.caption("These concern flags are analyst annotations. They do not affect the ML score or rule score.")
    for name, exp in zip(analyst_notes, analyst_exps):
        with st.expander(name, expanded=False):
            st.markdown(f'<div class="concern-box">{exp}</div>', unsafe_allow_html=True)

st.markdown("---")


# ═══════════════════════════════════════════════════════════════
# ██  SECTION 14 — CHARTS
# ═══════════════════════════════════════════════════════════════

chart_l, chart_r = st.columns(2)

with chart_l:
    st.markdown("### 📈 Top 15 Feature Importances")
    top15 = imp_df.head(15)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top15["feature"][::-1], top15["importance"][::-1], color="#D4001A")
    ax.set_xlabel("Importance Score")
    ax.set_title("Top 15 Features — Random Forest Model")
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with chart_r:
    st.markdown("### 📊 Training Data — Class Distribution")
    legit = class_dist.get(0, 0)
    fraud = class_dist.get(1, 0)
    fig2, ax2 = plt.subplots(figsize=(4, 5))
    bars = ax2.bar(["Legitimate", "Fraud"], [legit, fraud], color=["#000000", "#D4001A"])
    ax2.set_ylabel("Count")
    ax2.set_title("Class Distribution in Training Set")
    total = legit + fraud
    for bar, cnt in zip(bars, [legit, fraud]):
        ax2.text(bar.get_x() + bar.get_width()/2, cnt + total*0.005,
                 f"{cnt/total:.1%}", ha="center", fontsize=11, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

st.markdown("---")


# ═══════════════════════════════════════════════════════════════
# ██  SECTION 15 — MODEL PERFORMANCE METRICS
# ═══════════════════════════════════════════════════════════════

with st.expander("📐 Model Evaluation Metrics", expanded=False):
    from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    auc    = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"], output_dict=True)
    cm     = confusion_matrix(y_test, y_pred)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC-AUC",   f"{auc:.4f}")
    m2.metric("Precision", f"{report['Fraud']['precision']:.4f}")
    m3.metric("Recall",    f"{report['Fraud']['recall']:.4f}")
    m4.metric("F1-Score",  f"{report['Fraud']['f1-score']:.4f}")

    fig3, ax3 = plt.subplots(figsize=(4, 3))
    im = ax3.imshow(cm, cmap="Reds")
    ax3.set_xticks([0,1]); ax3.set_yticks([0,1])
    ax3.set_xticklabels(["Legit","Fraud"]); ax3.set_yticklabels(["Legit","Fraud"])
    ax3.set_xlabel("Predicted"); ax3.set_ylabel("Actual")
    ax3.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax3.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                     color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    st.markdown(f"""
**Interpretation:**
- **ROC-AUC {auc:.3f}** — the model distinguishes fraud from legitimate transactions
  with {auc*100:.1f}% accuracy (0.5 = random, 1.0 = perfect).
- **Recall {report['Fraud']['recall']:.1%}** — the model catches {report['Fraud']['recall']*100:.1f}% of all actual fraud cases.
  In fraud detection, recall is critical — missed fraud means real customer losses.
- **Precision {report['Fraud']['precision']:.1%}** — when the model flags fraud, it is correct
  {report['Fraud']['precision']*100:.1f}% of the time. Low precision = more false alarms for the operations team.
""")


# ═══════════════════════════════════════════════════════════════
# ██  SECTION 16 — ABOUT
# ═══════════════════════════════════════════════════════════════

with st.expander("📚 About this Project", expanded=False):
    st.markdown("""
## Fraud Prevention & Detection Risk Scoring System

Built as a university-level data science portfolio project demonstrating an end-to-end
fraud analytics pipeline.

### Architecture
| Layer | Technology | Purpose |
|---|---|---|
| Data generation | NumPy / pandas | 50,000 synthetic transactions |
| Feature engineering | pandas | 7 composite features (trust_score, ATO pattern etc.) |
| Machine learning | scikit-learn RandomForest | Fraud probability 0–1 |
| Rules engine | Pure Python | Bank-style operational rules |
| Score fusion | Weighted average | 60% ML + 40% Rules |
| Dashboard | Streamlit | Interactive demo |

### Key Design Decisions
- **Concerns are analyst-only** — PPO/PPF/IDTO/PHI etc. do not affect any score.
  They are recorded as analyst annotations for case notes.
- **Class imbalance handled** via `class_weight='balanced'` in the Random Forest.
- **N2N payment** is a scam green flag — name-to-name payments reduce scam risk.
- **Account age** supports 100+ year old accounts via date picker (min: 1920).
- **Close to BSB** = formerly "Close to Beneficiary" — label updated to banking terminology.

### Interview Talking Points
1. *Why a rules engine alongside ML?* — Rules give explainability; ML catches subtle
   correlations the rules miss. Neither alone is sufficient in banking.
2. *How is class imbalance handled?* — `class_weight='balanced'` adjusts loss weights
   so the model doesn't default to predicting "legitimate" for everything.
3. *Why are concerns excluded from scores?* — Concerns are analyst predictions about
   the type of fraud/scam. They inform the case note, not the probability.
4. *What is the combined score?* — 60% ML + 40% Rules, clipped to [0,1]. Thresholds
   are calibrated: <0.25 APPROVE, 0.25–0.45 REVIEW, 0.45–0.70 ESCALATE, >0.70 BLOCK.
""")

st.markdown(
    '<div style="text-align:center;color:#6D6E71;font-size:0.7rem;padding-top:20px">'
    'Fraud Prevention & Detection Risk Scoring System · University Demo · '
    'Not NAB\'s production system · Data is entirely synthetic'
    '</div>',
    unsafe_allow_html=True
)
