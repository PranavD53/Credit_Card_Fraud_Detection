import streamlit as st
import pandas as pd
import numpy as np
import time
import random

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield · AI Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Try loading real model; fall back to demo mode ────────────────────────────
try:
    import joblib
    model = joblib.load("fraud_model.pkl")
    DEMO_MODE = False
except Exception:
    model = None
    DEMO_MODE = True

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ─── Base ─────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: #080C14;
    color: #E2E8F0;
}

/* hide default header */
header[data-testid="stHeader"] { background: transparent; }

/* ─── Sidebar ──────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0D1321 0%, #0A0F1E 100%);
    border-right: 1px solid rgba(0, 212, 180, 0.15);
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #00D4B4;
    letter-spacing: 0.05em;
}

/* ─── Number inputs ────────────────────────────────────────────── */
input[type="number"] {
    background: #111827 !important;
    border: 1px solid rgba(0,212,180,0.25) !important;
    color: #E2E8F0 !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
}
input[type="number"]:focus {
    border-color: #00D4B4 !important;
    box-shadow: 0 0 0 2px rgba(0,212,180,0.15) !important;
}

/* ─── Selectbox / dropdown ─────────────────────────────────────── */
.stSelectbox > div > div {
    background: #111827 !important;
    border: 1px solid rgba(0,212,180,0.25) !important;
    color: #E2E8F0 !important;
    border-radius: 6px !important;
}

/* ─── Slider ───────────────────────────────────────────────────── */
.stSlider > div { color: #00D4B4; }

/* ─── Primary button ───────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #00D4B4 0%, #0095FF 100%);
    color: #080C14;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 15px;
    letter-spacing: 0.08em;
    border: none;
    border-radius: 10px;
    padding: 14px 32px;
    width: 100%;
    cursor: pointer;
    transition: all 0.2s ease;
    box-shadow: 0 0 24px rgba(0,212,180,0.25);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 40px rgba(0,212,180,0.45);
}

/* ─── Expander ─────────────────────────────────────────────────── */
details {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(0,212,180,0.12) !important;
    border-radius: 10px !important;
}

/* ─── Metric widgets ───────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"] { color: #94A3B8; font-size: 12px; }
[data-testid="stMetricValue"] { color: #E2E8F0; font-size: 26px; }

/* ─── Tabs ─────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03);
    border-radius: 10px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #94A3B8;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: rgba(0,212,180,0.12) !important;
    color: #00D4B4 !important;
    border-radius: 8px !important;
}

/* ─── Scrollbar ────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #080C14; }
::-webkit-scrollbar-thumb { background: rgba(0,212,180,0.3); border-radius: 10px; }

/* ─── Custom cards ─────────────────────────────────────────────── */
.glass-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
}
.stat-badge {
    display: inline-block;
    background: rgba(0,212,180,0.1);
    border: 1px solid rgba(0,212,180,0.3);
    color: #00D4B4;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.06em;
    margin-bottom: 8px;
}
.risk-high {
    background: rgba(239,68,68,0.08);
    border-color: rgba(239,68,68,0.35);
    color: #F87171;
}
.risk-low {
    background: rgba(16,185,129,0.08);
    border-color: rgba(16,185,129,0.35);
    color: #34D399;
}
.hero-number {
    font-family: 'JetBrains Mono', monospace;
    font-size: 48px;
    font-weight: 600;
    line-height: 1;
}
.sub-label {
    font-size: 12px;
    color: #64748B;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "scan_count" not in st.session_state:
    st.session_state.scan_count = 0

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ FraudShield")
    st.markdown("---")

    # ── Input mode ──
    input_mode = st.radio(
        "Input Mode",
        ["Manual Entry", "Preset Scenarios"],
        horizontal=True,
    )
    st.markdown("---")

    preset_features = None
    if input_mode == "Preset Scenarios":
        scenario = st.selectbox(
            "Select Scenario",
            [
                "Normal Online Purchase",
                "Large International Transfer",
                "Suspicious Micro-transactions",
                "ATM Withdrawal Pattern",
                "E-commerce Spike",
            ],
        )
        scenarios = {
            "Normal Online Purchase":        {"amount": 89.99,  "time": 43200, "risk": "low"},
            "Large International Transfer":  {"amount": 9500.0, "time": 3600,  "risk": "high"},
            "Suspicious Micro-transactions": {"amount": 1.00,   "time": 100,   "risk": "high"},
            "ATM Withdrawal Pattern":        {"amount": 200.0,  "time": 86000, "risk": "low"},
            "E-commerce Spike":              {"amount": 450.0,  "time": 21600, "risk": "low"},
        }
        s = scenarios[scenario]
        preset_features = [round(random.uniform(-3, 3) if s["risk"] == "high"
                                 else random.uniform(-0.5, 0.5), 4)
                           for _ in range(28)]
        amount_val = s["amount"]
        time_val   = s["time"]

        st.markdown(f"""
        <div class="glass-card">
            <div class="stat-badge">SCENARIO LOADED</div>
            <div style="font-size:13px;color:#94A3B8;margin-top:6px;">
                Amount: <b style="color:#E2E8F0;">${amount_val:,.2f}</b><br>
                Time offset: <b style="color:#E2E8F0;">{time_val:,}s</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── Transaction metadata ──
        st.markdown("### Transaction Details")
        amount_val = st.number_input("Amount (USD $)", min_value=0.01, value=100.00,
                                     step=0.01, format="%.2f")
        time_val   = st.number_input("Time (seconds from epoch)", value=10000,
                                     step=100)

        # ── PCA features ──
        st.markdown("### PCA Feature Vectors")
        st.caption("V1–V28 are PCA-transformed anonymised features.")

        with st.expander("⚙️ V1 – V14  (expand)", expanded=False):
            cols = st.columns(2)
            v_vals_a = []
            for i in range(1, 15):
                with cols[(i - 1) % 2]:
                    v_vals_a.append(
                        st.number_input(f"V{i}", value=0.0, step=0.01,
                                        format="%.4f", key=f"v{i}")
                    )

        with st.expander("⚙️ V15 – V28  (expand)", expanded=False):
            cols = st.columns(2)
            v_vals_b = []
            for i in range(15, 29):
                with cols[(i - 15) % 2]:
                    v_vals_b.append(
                        st.number_input(f"V{i}", value=0.0, step=0.01,
                                        format="%.4f", key=f"v{i}")
                    )

        preset_features = v_vals_a + v_vals_b

    st.markdown("---")
    analyze_btn = st.button("🔍 Analyze Transaction", use_container_width=True)

# ── Main content ───────────────────────────────────────────────────────────────
# Header
st.markdown("""
<div style="display:flex;align-items:center;gap:16px;margin-bottom:32px;margin-top:8px;">
    <div style="font-size:36px;">🛡️</div>
    <div>
        <div style="font-size:28px;font-weight:700;letter-spacing:-0.02em;">
            FraudShield <span style="color:#00D4B4;">AI</span>
        </div>
        <div style="color:#64748B;font-size:13px;letter-spacing:0.06em;">
            REAL-TIME CREDIT CARD FRAUD DETECTION ENGINE
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if DEMO_MODE:
    st.info("⚠️ **Demo Mode** — `fraud_model.pkl` not found. Predictions are simulated for UI demonstration.", icon="ℹ️")

# ── Top stats row ──
total_scans  = st.session_state.scan_count
fraud_caught = sum(1 for h in st.session_state.history if h["result"] == "fraud")
legit_count  = total_scans - fraud_caught
fraud_rate   = (fraud_caught / total_scans * 100) if total_scans else 0.0

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total Scans", f"{total_scans:,}")
with c2:
    st.metric("Flagged as Fraud", f"{fraud_caught:,}")
with c3:
    st.metric("Cleared", f"{legit_count:,}")
with c4:
    st.metric("Fraud Rate", f"{fraud_rate:.1f}%")

st.markdown("---")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔬 Analysis", "📋 Transaction Log"])

# ── Analysis tab ───────────────────────────────────────────────────────────────
with tab1:
    if analyze_btn:
        features_list = preset_features + [amount_val, time_val]
        input_arr = np.array(features_list).reshape(1, -1)

        with st.spinner("Running fraud detection model…"):
            time.sleep(0.7)  # tactile pause

            if DEMO_MODE:
                # Heuristic demo prediction
                v_sum   = sum(abs(x) for x in preset_features)
                is_fraud = (v_sum > 20) or (amount_val > 5000) or (time_val < 500)
                proba    = round(random.uniform(0.72, 0.97) if is_fraud
                                 else random.uniform(0.01, 0.18), 4)
                pred     = 1 if is_fraud else 0
            else:
                pred  = model.predict(input_arr)[0]
                try:
                    proba = model.predict_proba(input_arr)[0][1]
                except AttributeError:
                    proba = float(pred)

        # Record history
        st.session_state.scan_count += 1
        st.session_state.history.append({
            "id":     st.session_state.scan_count,
            "amount": amount_val,
            "time":   time_val,
            "result": "fraud" if pred == 1 else "legit",
            "score":  proba,
        })

        # ── Result banner ──
        if pred == 1:
            st.markdown(f"""
            <div class="glass-card risk-high" style="border-left:4px solid #EF4444;margin-top:8px;">
                <div class="stat-badge risk-high">🚨 FRAUD ALERT</div>
                <div class="hero-number" style="color:#F87171;">{proba:.1%}</div>
                <div class="sub-label">fraud probability score</div>
                <p style="margin-top:14px;color:#FCA5A5;font-size:14px;">
                    This transaction has been flagged as <b>HIGH RISK</b>.
                    Immediate review is recommended. The card has been provisionally held.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="glass-card risk-low" style="border-left:4px solid #10B981;margin-top:8px;">
                <div class="stat-badge risk-low">✅ CLEARED</div>
                <div class="hero-number" style="color:#34D399;">{(1-proba):.1%}</div>
                <div class="sub-label">legitimacy confidence</div>
                <p style="margin-top:14px;color:#6EE7B7;font-size:14px;">
                    This transaction appears <b>legitimate</b>. No suspicious patterns
                    detected across the 28 behavioral feature vectors.
                </p>
            </div>
            """, unsafe_allow_html=True)

        # ── Risk breakdown ──
        st.markdown("#### Risk Signal Breakdown")
        rb1, rb2, rb3 = st.columns(3)

        risk_color = "#EF4444" if pred == 1 else "#10B981"
        amount_risk = "High" if amount_val > 3000 else ("Medium" if amount_val > 500 else "Low")
        time_risk   = "Unusual" if time_val < 1000 or time_val > 80000 else "Normal"
        v_magnitude = round(sum(abs(x) for x in preset_features), 2)

        with rb1:
            st.markdown(f"""
            <div class="glass-card" style="text-align:center;">
                <div class="sub-label">Amount Risk</div>
                <div style="font-size:22px;font-weight:700;margin-top:8px;
                     color:{'#EF4444' if amount_risk=='High' else '#F59E0B' if amount_risk=='Medium' else '#34D399'}">
                    {amount_risk}
                </div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#64748B;margin-top:4px;">
                    ${amount_val:,.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with rb2:
            st.markdown(f"""
            <div class="glass-card" style="text-align:center;">
                <div class="sub-label">Time Pattern</div>
                <div style="font-size:22px;font-weight:700;margin-top:8px;
                     color:{'#F59E0B' if time_risk=='Unusual' else '#34D399'}">
                    {time_risk}
                </div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#64748B;margin-top:4px;">
                    {int(time_val):,}s
                </div>
            </div>
            """, unsafe_allow_html=True)

        with rb3:
            st.markdown(f"""
            <div class="glass-card" style="text-align:center;">
                <div class="sub-label">Feature Magnitude</div>
                <div style="font-size:22px;font-weight:700;margin-top:8px;color:{risk_color}">
                    {v_magnitude:.2f}
                </div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#64748B;margin-top:4px;">
                    Σ|V1…V28|
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Feature heatmap ──
        st.markdown("#### Feature Vector Heatmap (V1–V28)")
        heat_data = pd.DataFrame(
            [preset_features],
            columns=[f"V{i}" for i in range(1, 29)],
        )
        st.dataframe(
            heat_data.style
                .background_gradient(cmap="RdYlGn_r", axis=1)
                .format("{:.4f}"),
            use_container_width=True,
        )

    else:
        # Idle state
        st.markdown("""
        <div class="glass-card" style="text-align:center;padding:60px 24px;">
            <div style="font-size:64px;margin-bottom:16px;">🛡️</div>
            <div style="font-size:20px;font-weight:600;color:#E2E8F0;margin-bottom:8px;">
                Ready to Analyze
            </div>
            <div style="color:#64748B;font-size:14px;max-width:380px;margin:0 auto;">
                Configure your transaction details in the sidebar and click
                <b style="color:#00D4B4;">Analyze Transaction</b> to run the
                AI fraud detection model.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Transaction Log tab ────────────────────────────────────────────────────────
with tab2:
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        df["amount"] = df["amount"].map("${:,.2f}".format)
        df["score"]  = df["score"].map("{:.2%}".format)
        df["result"] = df["result"].map({"fraud": "🚨 Fraud", "legit": "✅ Legit"})
        df.columns   = ["#", "Amount", "Time (s)", "Result", "Fraud Score"]
        df = df.set_index("#").sort_index(ascending=False)
        st.dataframe(df, use_container_width=True)

        col_dl, col_cl = st.columns([2, 1])
        with col_dl:
            csv = df.to_csv().encode("utf-8")
            st.download_button("⬇️ Export Log as CSV", csv,
                               "fraudshield_log.csv", "text/csv",
                               use_container_width=True)
        with col_cl:
            if st.button("🗑️ Clear Log", use_container_width=True):
                st.session_state.history = []
                st.session_state.scan_count = 0
                st.rerun()
    else:
        st.markdown("""
        <div class="glass-card" style="text-align:center;padding:48px;">
            <div style="font-size:40px;margin-bottom:12px;">📋</div>
            <div style="color:#64748B;">No transactions analyzed yet.</div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#334155;font-size:12px;letter-spacing:0.05em;">
    FraudShield AI · Powered by Machine Learning · For authorized use only
</div>
""", unsafe_allow_html=True)