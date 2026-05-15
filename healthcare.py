"""
=============================================================================
Healthcare Clinic Optimization — Streamlit Dashboard
=============================================================================
Report 1: Business Case — AI-Powered No-Show Reduction & Revenue Recovery
Client  : Healthcare Clinic Chain (5 Clinics)
Author  : Strategy & Analytics Division
Date    : May 2026

Implements all 6 strategy modules as an interactive Streamlit dashboard:
  Module 1 — Current State Analysis & Revenue Metrics
  Module 2 — No-Show Prediction Model (ML / GradientBoosting)
  Module 3 — Smart Overbooking Engine
  Module 4 — Patient Segmentation (RFM)
  Module 5 — Automated Reminder Scheduler
  Module 6 — KPI Dashboard & 12-Month Projections
=============================================================================
"""

import math
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
#  Page Config
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Healthcare Clinic Optimization",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  Design System — Refined Clinical Dark Theme
# ══════════════════════════════════════════════════════════════════════════════
PALETTE = {
    "teal":       "#00C9A7",
    "teal_dark":  "#007A64",
    "red":        "#FF4757",
    "blue":       "#1E90FF",
    "blue_dark":  "#1565C0",
    "amber":      "#FFA502",
    "purple":     "#7B68EE",
    "green":      "#2ED573",
    "gray":       "#8895A7",
    "light":      "#E8EDF2",
    "dark":       "#0D1B2A",
    "card_bg":    "#12263A",
    "sidebar_bg": "#0A1929",
    "accent":     "#00C9A7",
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&family=Mulish:wght@300;400;500;600&display=swap');

/* ── Global ── */
html, body, [class*="css"] {{
    font-family: 'Mulish', sans-serif;
}}
.stApp {{
    background: linear-gradient(160deg, #071423 0%, #0D1B2A 60%, #091520 100%);
}}

/* ── Header ── */
.dash-header {{
    background: linear-gradient(135deg, rgba(0,201,167,0.12) 0%, rgba(30,144,255,0.08) 100%);
    border: 1px solid rgba(0,201,167,0.25);
    border-radius: 20px;
    padding: 2.2rem 2.8rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}}
.dash-header::before {{
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 40%;
    height: 200%;
    background: radial-gradient(circle, rgba(0,201,167,0.06) 0%, transparent 70%);
    pointer-events: none;
}}
.dash-header h1 {{
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: #FFFFFF;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
}}
.dash-header .subtitle {{
    font-family: 'Mulish', sans-serif;
    color: #8895A7;
    font-size: 1rem;
    font-weight: 300;
}}
.dash-header .tag {{
    display: inline-block;
    background: rgba(0,201,167,0.15);
    border: 1px solid rgba(0,201,167,0.4);
    color: #00C9A7;
    padding: 0.2rem 0.8rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-right: 0.5rem;
    margin-top: 0.6rem;
}}

/* ── Metric cards ── */
.metric-row {{ display: flex; gap: 1rem; margin-bottom: 1.2rem; flex-wrap: wrap; }}
.metric-card {{
    flex: 1;
    min-width: 140px;
    background: linear-gradient(145deg, #12263A, #0D1B2A);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, border-color 0.2s ease;
}}
.metric-card:hover {{
    transform: translateY(-2px);
    border-color: rgba(0,201,167,0.3);
}}
.metric-card::after {{
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: var(--card-accent, #00C9A7);
}}
.metric-card .mval {{
    font-family: 'Syne', sans-serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: #FFFFFF;
    line-height: 1.1;
}}
.metric-card .mlbl {{
    font-size: 0.75rem;
    color: #8895A7;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-top: 0.35rem;
}}
.metric-card .mdelta {{
    font-size: 0.82rem;
    font-weight: 600;
    margin-top: 0.3rem;
}}

/* ── Section headers ── */
.section-header {{
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #FFFFFF;
    border-left: 3px solid #00C9A7;
    padding-left: 0.9rem;
    margin: 1.8rem 0 1rem 0;
}}
.section-sub {{
    font-size: 0.88rem;
    color: #8895A7;
    margin-top: -0.6rem;
    margin-bottom: 1.2rem;
    padding-left: 1.2rem;
}}

/* ── Risk badges ── */
.risk-low    {{ background:#0D3D2E; color:#2ED573; border:1px solid #2ED573; border-radius:6px; padding:0.2rem 0.6rem; font-size:0.8rem; font-weight:600; }}
.risk-medium {{ background:#3D2E0D; color:#FFA502; border:1px solid #FFA502; border-radius:6px; padding:0.2rem 0.6rem; font-size:0.8rem; font-weight:600; }}
.risk-high   {{ background:#3D0D0D; color:#FF4757; border:1px solid #FF4757; border-radius:6px; padding:0.2rem 0.6rem; font-size:0.8rem; font-weight:600; }}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background: #0A1929 !important;
    border-right: 1px solid rgba(0,201,167,0.15);
}}
section[data-testid="stSidebar"] .stMarkdown h3 {{
    font-family: 'Syne', sans-serif;
    color: #00C9A7;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    border-bottom: 1px solid rgba(0,201,167,0.2);
    padding-bottom: 0.4rem;
    margin-top: 1.2rem;
}}

/* ── Info box ── */
.info-box {{
    background: rgba(30,144,255,0.08);
    border: 1px solid rgba(30,144,255,0.25);
    border-radius: 12px;
    padding: 1rem 1.3rem;
    margin: 0.8rem 0;
    font-size: 0.9rem;
    color: #B8C8D8;
}}
.warn-box {{
    background: rgba(255,165,2,0.08);
    border: 1px solid rgba(255,165,2,0.3);
    border-radius: 12px;
    padding: 1rem 1.3rem;
    margin: 0.8rem 0;
    font-size: 0.9rem;
    color: #FFCC80;
}}
.success-box {{
    background: rgba(0,201,167,0.08);
    border: 1px solid rgba(0,201,167,0.3);
    border-radius: 12px;
    padding: 1rem 1.3rem;
    margin: 0.8rem 0;
    font-size: 0.9rem;
    color: #80E8D8;
}}

/* ── DataFrame ── */
.stDataFrame {{ border-radius: 10px; overflow: hidden; }}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background: rgba(18,38,58,0.8);
    border-radius: 10px;
    padding: 0.3rem;
    border: 1px solid rgba(255,255,255,0.06);
}}
.stTabs [data-baseweb="tab"] {{
    color: #8895A7;
    font-family: 'Mulish', sans-serif;
    font-weight: 600;
    border-radius: 7px;
}}
.stTabs [aria-selected="true"] {{
    background: rgba(0,201,167,0.15) !important;
    color: #00C9A7 !important;
}}

/* ── Slider ── */
.stSlider [data-baseweb="slider"] {{ padding-top: 0.5rem; }}

/* ── Button ── */
.stButton > button {{
    background: linear-gradient(135deg, #00C9A7, #007A64);
    color: white;
    border: none;
    border-radius: 10px;
    font-family: 'Mulish', sans-serif;
    font-weight: 600;
    font-size: 0.9rem;
    padding: 0.6rem 1.6rem;
    transition: all 0.2s ease;
}}
.stButton > button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(0,201,167,0.35);
}}

/* ── Progress label ── */
.prog-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #8895A7;
    margin-bottom: 0.3rem;
}}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  Global Config
# ══════════════════════════════════════════════════════════════════════════════
CLINICS                  = 5
PATIENTS_PER_CLINIC_DAY  = 40
NO_SHOW_RATE             = 0.30
REVENUE_PER_VISIT        = 700
WORKING_DAYS             = 26
TOTAL_SLOTS_DAY          = CLINICS * PATIENTS_PER_CLINIC_DAY  # 200

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(18,38,58,0.4)",
    font=dict(family="Mulish, sans-serif", color="#B8C8D8", size=12),
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
)


def apply_layout(fig, title="", height=380):
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=title, font=dict(family="Syne", size=14, color="#FFFFFF"), x=0),
        height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 1: Current State Analysis
# ══════════════════════════════════════════════════════════════════════════════
class CurrentStateAnalysis:
    def __init__(self, no_show_rate=NO_SHOW_RATE, revenue=REVENUE_PER_VISIT):
        self.nsr           = no_show_rate
        self.rev           = revenue
        self.total_slots   = TOTAL_SLOTS_DAY
        self.attended      = int(self.total_slots * (1 - self.nsr))
        self.no_show_slots = self.total_slots - self.attended
        self.actual_daily  = self.attended * self.rev
        self.actual_monthly= self.actual_daily * WORKING_DAYS
        self.max_daily     = self.total_slots * self.rev
        self.max_monthly   = self.max_daily * WORKING_DAYS
        self.lost_daily    = self.no_show_slots * self.rev
        self.lost_monthly  = self.lost_daily * WORKING_DAYS
        self.utilisation   = self.attended / self.total_slots

    def at_target(self, target_nsr):
        att = int(self.total_slots * (1 - target_nsr))
        rev = att * self.rev * WORKING_DAYS
        return {"attended": att, "monthly_rev": rev,
                "uplift": rev - self.actual_monthly,
                "util": att / self.total_slots}

    def revenue_gap_chart(self):
        scenarios  = [0.30, 0.22, 0.14, 0.0]
        labels     = ["Current (30%)", "3-Month (22%)", "12-Month (14%)", "Max (0%)"]
        revenues   = [self.at_target(s)["monthly_rev"] / 1e5 for s in scenarios]
        colors     = [PALETTE["gray"], PALETTE["blue"], PALETTE["teal"], "#2ED573"]

        fig = go.Figure()
        for i, (lbl, rev, col) in enumerate(zip(labels, revenues, colors)):
            fig.add_trace(go.Bar(
                x=[lbl], y=[rev], name=lbl,
                marker_color=col, marker_line_width=0,
                text=[f"₹{rev:.1f}L"], textposition="outside",
                textfont=dict(size=13, color="#FFFFFF", family="Syne"),
                showlegend=False,
            ))
        # Lost revenue overlay on current bar
        lost = self.lost_monthly / 1e5
        fig.add_trace(go.Bar(
            x=["Current (30%)"], y=[lost], base=[revenues[0]],
            marker_color=PALETTE["red"], opacity=0.75,
            name=f"Lost to no-shows: ₹{lost:.1f}L",
            text=[f"-₹{lost:.1f}L"], textposition="inside",
            textfont=dict(size=10, color="white"),
        ))
        fig.add_hline(
            y=self.max_monthly / 1e5,
            line_dash="dash", line_color=PALETTE["teal"],
            annotation_text="Max potential", annotation_position="top right",
        )
        fig.update_layout(barmode="stack", **PLOTLY_LAYOUT,
                          title=dict(text="Revenue Gap Analysis — Current State vs. Targets",
                                     font=dict(family="Syne", size=14, color="#FFFFFF"), x=0),
                          height=380,
                          yaxis_title="Monthly Revenue (₹ Lakh)",
                          legend=dict(bgcolor="rgba(0,0,0,0)"))
        return fig

    def utilisation_gauge(self):
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=self.utilisation * 100,
            delta={"reference": 86, "valueformat": ".0f",
                   "suffix": "%", "decreasing": {"color": PALETTE["red"]}},
            number={"suffix": "%", "font": {"family": "Syne", "size": 46, "color": "#FFFFFF"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8895A7"},
                "bar":  {"color": PALETTE["teal"], "thickness": 0.25},
                "bgcolor": "rgba(18,38,58,0.6)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 70],  "color": "rgba(255,71,87,0.15)"},
                    {"range": [70, 80], "color": "rgba(255,165,2,0.15)"},
                    {"range": [80, 100],"color": "rgba(0,201,167,0.12)"},
                ],
                "threshold": {"line": {"color": "#2ED573", "width": 3},
                              "thickness": 0.8, "value": 86},
            },
            title={"text": "Slot Utilisation Rate", "font": {"family": "Syne", "color": "#8895A7"}},
        ))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=260,
                          margin=dict(l=20, r=20, t=10, b=10))
        return fig


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 2: No-Show Prediction Model
# ══════════════════════════════════════════════════════════════════════════════
class NoShowPredictionModel:
    FEATURE_NAMES = [
        "patient_age_group", "appointment_lead_days", "day_of_week",
        "time_slot", "past_no_show_count", "distance_km",
        "appointment_type", "clinic_id",
    ]

    def __init__(self, n_samples=5000):
        self.n       = n_samples
        self.scaler  = StandardScaler()
        self.model   = GradientBoostingClassifier(
            n_estimators=120, max_depth=4, learning_rate=0.08, random_state=42
        )
        self._gen_data()
        self.trained = False

    def _gen_data(self):
        n = self.n
        age   = np.random.choice([0, 1, 2], n, p=[0.30, 0.45, 0.25])
        lead  = np.random.randint(1, 30, n)
        dow   = np.random.randint(0, 5, n)
        tslot = np.random.choice([0, 1, 2], n, p=[0.40, 0.35, 0.25])
        past  = np.random.poisson(1.2, n).clip(0, 8)
        dist  = np.abs(np.random.normal(8, 5, n)).clip(0.5, 40)
        atype = np.random.choice([0, 1, 2], n, p=[0.50, 0.35, 0.15])
        cid   = np.random.randint(0, 5, n)

        log_odds = (
            -1.5 + 0.04*lead + 0.35*(tslot == 2).astype(int)
            + 0.45*past + 0.03*dist - 0.20*(age == 2).astype(int)
            + 0.25*(atype == 1).astype(int) - 0.10*(dow == 0).astype(int)
        )
        prob  = 1 / (1 + np.exp(-log_odds))
        label = (np.random.random(n) < prob).astype(int)

        self.df = pd.DataFrame({
            "patient_age_group": age, "appointment_lead_days": lead,
            "day_of_week": dow, "time_slot": tslot, "past_no_show_count": past,
            "distance_km": dist.round(1), "appointment_type": atype,
            "clinic_id": cid, "no_show": label,
        })

    def train(self):
        X  = self.scaler.fit_transform(self.df[self.FEATURE_NAMES].values)
        y  = self.df["no_show"].values
        self.X_tr, self.X_te, self.y_tr, self.y_te = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        self.model.fit(self.X_tr, self.y_tr)
        self.trained = True

    def evaluate(self):
        yp   = self.model.predict(self.X_te)
        yprob= self.model.predict_proba(self.X_te)[:, 1]
        return {
            "auc": roc_auc_score(self.y_te, yprob),
            "report": classification_report(self.y_te, yp, output_dict=True),
            "y_prob": yprob, "y_test": self.y_te,
        }

    def risk_band(self, score):
        if score < 0.30:   return "Low"
        elif score < 0.60: return "Medium"
        return "High"

    def reminder_protocol(self, score):
        band = self.risk_band(score)
        proto = {
            "Low":    {"reminders": 1, "touchpoints": ["24h before"],
                       "channel": "WhatsApp", "confirm": False},
            "Medium": {"reminders": 2, "touchpoints": ["48h", "2h before"],
                       "channel": "WhatsApp + SMS", "confirm": False},
            "High":   {"reminders": 3, "touchpoints": ["48h", "24h", "2h"],
                       "channel": "WhatsApp + SMS + Call", "confirm": True},
        }
        return {"risk_band": band, "score": round(score, 3), **proto[band]}

    def predict_score(self, features_dict):
        feat_arr = np.array([[features_dict.get(f, 0) for f in self.FEATURE_NAMES]])
        scaled   = self.scaler.transform(feat_arr)
        return float(self.model.predict_proba(scaled)[0, 1])

    def feature_importance_chart(self):
        imp  = self.model.feature_importances_
        idx  = np.argsort(imp)
        lbls = [self.FEATURE_NAMES[i].replace("_", " ").title() for i in idx]
        vals = imp[idx]
        cols = [PALETTE["teal"] if i == idx[-1] else PALETTE["blue"] for i in idx]

        fig = go.Figure(go.Bar(
            x=vals, y=lbls, orientation="h",
            marker_color=cols, marker_line_width=0,
            text=[f"{v:.3f}" for v in vals], textposition="outside",
            textfont=dict(size=10, color="#B8C8D8"),
        ))
        return apply_layout(fig, "Feature Importance — No-Show Prediction Model", 360)

    def risk_distribution_chart(self):
        probs = self.model.predict_proba(self.X_te)[:, 1]
        fig   = go.Figure()
        fig.add_trace(go.Histogram(
            x=probs[self.y_te == 0], name="Attended",
            marker_color=PALETTE["teal"], opacity=0.7,
            xbins=dict(start=0, end=1, size=0.03),
        ))
        fig.add_trace(go.Histogram(
            x=probs[self.y_te == 1], name="No-Show",
            marker_color=PALETTE["red"], opacity=0.7,
            xbins=dict(start=0, end=1, size=0.03),
        ))
        for threshold, color, label in [
            (0.30, PALETTE["amber"], "Low/Med (0.30)"),
            (0.60, PALETTE["purple"], "Med/High (0.60)"),
        ]:
            fig.add_vline(x=threshold, line_dash="dash", line_color=color,
                          annotation_text=label, annotation_position="top",
                          annotation_font=dict(size=10, color=color))
        fig.update_layout(barmode="overlay", **PLOTLY_LAYOUT,
                          title=dict(text="No-Show Risk Score Distribution",
                                     font=dict(family="Syne", size=14, color="#FFFFFF"), x=0),
                          height=360,
                          xaxis_title="Predicted No-Show Probability",
                          yaxis_title="Count",
                          legend=dict(bgcolor="rgba(0,0,0,0)"))
        return fig


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 3: Smart Overbooking Engine
# ══════════════════════════════════════════════════════════════════════════════
class SmartOverbookingEngine:
    TIME_BANDS = ["08–10", "10–12", "12–14", "14–16", "16–18"]
    HIST_NS    = {"08–10": 0.22, "10–12": 0.25, "12–14": 0.35, "14–16": 0.28, "16–18": 0.38}

    def __init__(self, slot_cap=8, initial_cap=0.10):
        self.slot_cap    = slot_cap
        self.initial_cap = initial_cap

    def overbooking_plan(self):
        rows = []
        for band, ns in self.HIST_NS.items():
            raw  = self.slot_cap * ns
            cap  = min(raw, self.slot_cap * self.initial_cap)
            buf  = max(1, round(cap))
            rows.append({
                "Time Band":         band,
                "No-Show %":         f"{ns:.0%}",
                "NS Rate":           ns,
                "Buffer Slots":      buf,
                "Total Booked":      self.slot_cap + buf,
                "Expected Attended": round(self.slot_cap * (1 - ns) + buf * 0.5),
                "Daily Uplift (₹)":  buf * REVENUE_PER_VISIT,
                "Monthly Uplift (₹)":buf * REVENUE_PER_VISIT * WORKING_DAYS * CLINICS,
            })
        return pd.DataFrame(rows)

    def waitlist_simulation(self, days=26, fill_rate=0.55):
        results = []
        for day in range(1, days + 1):
            opened = filled = 0
            for ns in self.HIST_NS.values():
                s = np.random.binomial(self.slot_cap, ns)
                f = np.random.binomial(s, fill_rate)
                opened += s; filled += f
            results.append({
                "Day": day, "Slots Opened": opened, "Filled": filled,
                "Fill Rate %": filled / max(opened, 1) * 100,
                "Added Revenue (₹K)": filled * REVENUE_PER_VISIT / 1000,
            })
        return pd.DataFrame(results)

    def overbooking_chart(self):
        plan = self.overbooking_plan()
        fig  = go.Figure()
        fig.add_trace(go.Bar(
            name="Base Slots", x=plan["Time Band"], y=[self.slot_cap]*5,
            marker_color=PALETTE["gray"], opacity=0.6,
        ))
        fig.add_trace(go.Bar(
            name="Overbook Buffer", x=plan["Time Band"], y=plan["Buffer Slots"],
            marker_color=PALETTE["teal"],
            text=[f"+{v}" for v in plan["Buffer Slots"]], textposition="outside",
            textfont=dict(size=12, color=PALETTE["teal"]),
        ))
        fig.update_layout(barmode="group", **PLOTLY_LAYOUT,
                          title=dict(text="Smart Overbooking Buffer by Time Band",
                                     font=dict(family="Syne", size=14, color="#FFFFFF"), x=0),
                          height=360,
                          yaxis_title="Slots",
                          legend=dict(bgcolor="rgba(0,0,0,0)"))
        return fig

    def waitlist_chart(self, sim):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(
            x=sim["Day"], y=sim["Added Revenue (₹K)"],
            name="Added Revenue (₹K/day)", fill="tozeroy",
            fillcolor="rgba(0,201,167,0.12)",
            line=dict(color=PALETTE["teal"], width=2.5),
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=sim["Day"], y=sim["Fill Rate %"],
            name="Waitlist Fill Rate (%)",
            line=dict(color=PALETTE["amber"], width=2, dash="dot"),
        ), secondary_y=True)
        fig.update_layout(**PLOTLY_LAYOUT,
                          title=dict(text="Waitlist Fill Simulation — Monthly View",
                                     font=dict(family="Syne", size=14, color="#FFFFFF"), x=0),
                          height=360,
                          legend=dict(bgcolor="rgba(0,0,0,0)"))
        fig.update_yaxes(title_text="Revenue (₹K/day)", secondary_y=False,
                         gridcolor="rgba(255,255,255,0.05)")
        fig.update_yaxes(title_text="Fill Rate (%)", secondary_y=True,
                         gridcolor="rgba(0,0,0,0)")
        return fig


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 4: Patient Segmentation (RFM)
# ══════════════════════════════════════════════════════════════════════════════
class PatientSegmentation:
    SEGMENT_CONFIG = {
        "Champions":    {"color": PALETTE["teal"],   "goal": "Retention & LTV",
                         "channel": "WhatsApp",              "reminders": 1},
        "At-Risk":      {"color": PALETTE["amber"],  "goal": "Reactivation",
                         "channel": "WhatsApp + SMS",        "reminders": 2},
        "High No-Show": {"color": PALETTE["red"],    "goal": "Reduce no-show rate",
                         "channel": "WA + SMS + Call",       "reminders": 3},
        "New Patients": {"color": PALETTE["blue"],   "goal": "First-visit conversion",
                         "channel": "WhatsApp",              "reminders": 2},
    }

    def __init__(self, n=1500):
        self.n = n
        self._gen()

    def _gen(self):
        segs = np.random.choice(
            ["Champions", "At-Risk", "High No-Show", "New Patients"],
            self.n, p=[0.28, 0.30, 0.20, 0.22]
        )
        recency = np.where(segs == "Champions",   np.random.randint(1, 15,  self.n),
                  np.where(segs == "At-Risk",      np.random.randint(30, 90, self.n),
                  np.where(segs == "High No-Show", np.random.randint(5,  45, self.n),
                                                   np.random.randint(1,  60, self.n))))
        freq    = np.where(segs == "Champions",   np.random.randint(8, 24, self.n),
                  np.where(segs == "At-Risk",      np.random.randint(3, 10, self.n),
                  np.where(segs == "High No-Show", np.random.randint(2, 10, self.n),
                                                   np.random.randint(1,  3, self.n))))
        spend   = freq * REVENUE_PER_VISIT * np.random.uniform(0.85, 1.15, self.n)
        ns_cnt  = np.where(segs == "High No-Show", np.random.randint(3, 9, self.n),
                  np.where(segs == "Champions",     np.random.randint(0, 2, self.n),
                                                    np.random.randint(0, 4, self.n)))
        self.df = pd.DataFrame({
            "segment": segs, "recency_days": recency.astype(int),
            "frequency": freq, "monetary": spend.round(0),
            "no_shows": ns_cnt,
        })

    def summary(self):
        g = self.df.groupby("segment").agg(
            count=("segment", "count"),
            avg_recency=("recency_days", "mean"),
            avg_freq=("frequency", "mean"),
            avg_spend=("monetary", "mean"),
            avg_no_shows=("no_shows", "mean"),
        ).reset_index()
        g["share_%"] = (g["count"] / g["count"].sum() * 100).round(1)
        g["avg_freq"] = g["avg_freq"].round(1)
        g["avg_spend"] = g["avg_spend"].round(0)
        g["avg_no_shows"] = g["avg_no_shows"].round(2)
        return g

    def pie_chart(self):
        s = self.summary()
        fig = go.Figure(go.Pie(
            labels=s["segment"], values=s["count"],
            hole=0.55,
            marker=dict(colors=[self.SEGMENT_CONFIG[sg]["color"] for sg in s["segment"]],
                        line=dict(color="#0D1B2A", width=3)),
            textinfo="label+percent",
            textfont=dict(family="Mulish", size=12),
            hovertemplate="%{label}<br>Count: %{value}<br>Share: %{percent}<extra></extra>",
        ))
        fig.add_annotation(text="RFM<br>Segments", x=0.5, y=0.5,
                           font=dict(family="Syne", size=15, color="#FFFFFF"),
                           showarrow=False)
        return apply_layout(fig, "Patient Segmentation Distribution", 380)

    def rfm_scatter(self):
        seg_colors = {s: c["color"] for s, c in self.SEGMENT_CONFIG.items()}
        fig = go.Figure()
        for seg in self.df["segment"].unique():
            sub = self.df[self.df["segment"] == seg]
            fig.add_trace(go.Scatter(
                x=sub["frequency"], y=sub["monetary"],
                mode="markers", name=seg,
                marker=dict(color=seg_colors[seg], size=6, opacity=0.65,
                            line=dict(width=0)),
                text=sub["segment"],
            ))
        return apply_layout(fig, "RFM Scatter — Frequency vs. Spend", 380)


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 5: Reminder Scheduler
# ══════════════════════════════════════════════════════════════════════════════
class ReminderScheduler:
    CHANNEL_COST = {"WhatsApp": 0.50, "SMS": 0.25, "Call": 2.00}

    def daily_schedule(self, model: NoShowPredictionModel):
        rows = []
        for _ in range(TOTAL_SLOTS_DAY):
            score = float(model.model.predict_proba(
                model.scaler.transform(np.random.randn(1, 8))
            )[0, 1])
            p = model.reminder_protocol(score)
            cost = (
                (self.CHANNEL_COST["WhatsApp"] if "WhatsApp" in p["channel"] else 0)
                + (self.CHANNEL_COST["SMS"]     if "SMS"      in p["channel"] else 0)
                + (self.CHANNEL_COST["Call"]    if "Call"     in p["channel"] else 0)
            ) * p["reminders"]
            rows.append({"risk_band": p["risk_band"], "score": p["score"],
                         "reminders": p["reminders"], "cost_inr": cost})
        return pd.DataFrame(rows)

    def cost_summary(self, sched):
        total_msg  = sched["reminders"].sum()
        total_cost = sched["cost_inr"].sum()
        return {
            "total_messages": int(total_msg),
            "daily_cost":     round(total_cost, 2),
            "monthly_cost":   round(total_cost * WORKING_DAYS, 2),
            "band_counts":    sched["risk_band"].value_counts().to_dict(),
        }

    def band_breakdown_chart(self, sched):
        bc  = sched["risk_band"].value_counts().reset_index()
        bc.columns = ["Band", "Count"]
        cols = {"Low": PALETTE["teal"], "Medium": PALETTE["amber"], "High": PALETTE["red"]}
        fig  = go.Figure(go.Bar(
            x=bc["Band"], y=bc["Count"],
            marker_color=[cols.get(b, PALETTE["gray"]) for b in bc["Band"]],
            text=bc["Count"], textposition="outside",
            textfont=dict(size=13, color="#FFFFFF"),
        ))
        return apply_layout(fig, "Daily Reminder Band Distribution", 320)


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 6: KPI Dashboard
# ══════════════════════════════════════════════════════════════════════════════
class KPIDashboard:
    NO_SHOW_TRAJ = [0.30, 0.295, 0.275, 0.255, 0.235, 0.215,
                    0.198, 0.185, 0.175, 0.168, 0.158, 0.150, 0.140]
    MONTHS = list(range(13))

    def revenue_traj(self):
        return [
            int(TOTAL_SLOTS_DAY * (1 - ns) * REVENUE_PER_VISIT * WORKING_DAYS) / 1e5
            for ns in self.NO_SHOW_TRAJ
        ]

    def patients_traj(self):
        return [int(TOTAL_SLOTS_DAY * (1 - ns)) for ns in self.NO_SHOW_TRAJ]

    def projection_chart(self):
        rev  = self.revenue_traj()
        pats = self.patients_traj()
        ns   = [x * 100 for x in self.NO_SHOW_TRAJ]

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(
            x=self.MONTHS, y=rev, name="Monthly Revenue (₹L)",
            line=dict(color=PALETTE["teal"], width=3),
            fill="tozeroy", fillcolor="rgba(0,201,167,0.08)",
            mode="lines+markers", marker=dict(size=6),
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=self.MONTHS, y=ns, name="No-Show Rate (%)",
            line=dict(color=PALETTE["red"], width=2, dash="dot"),
            mode="lines+markers", marker=dict(size=5),
        ), secondary_y=True)

        # Phase markers
        for x, lbl, col in [(3, "Phase 1", PALETTE["amber"]), (12, "Phase 2", PALETTE["green"])]:
            fig.add_vline(x=x, line_dash="dash", line_color=col, opacity=0.5,
                          annotation_text=lbl, annotation_position="top",
                          annotation_font=dict(size=10, color=col))

        fig.update_layout(**PLOTLY_LAYOUT,
                          title=dict(text="12-Month KPI Projection",
                                     font=dict(family="Syne", size=14, color="#FFFFFF"), x=0),
                          height=380, legend=dict(bgcolor="rgba(0,0,0,0)"))
        fig.update_yaxes(title_text="Monthly Revenue (₹ Lakh)", secondary_y=False,
                         gridcolor="rgba(255,255,255,0.05)")
        fig.update_yaxes(title_text="No-Show Rate (%)", secondary_y=True,
                         gridcolor="rgba(0,0,0,0)")
        return fig

    def waterfall_chart(self):
        base  = self.revenue_traj()[0]
        final = self.revenue_traj()[-1]
        fig   = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "relative", "total"],
            x=["Baseline", "Reminders", "Overbooking", "Segmentation",
               "Waitlist Fill", "Month 12 Target"],
            y=[base, 2.1, 1.4, 0.9, 0.6, 0],
            connector=dict(line=dict(color="rgba(255,255,255,0.15)", width=1, dash="dot")),
            increasing=dict(marker=dict(color=PALETTE["teal"])),
            decreasing=dict(marker=dict(color=PALETTE["red"])),
            totals=dict(marker=dict(color=PALETTE["green"])),
            text=[f"₹{v:.1f}L" if i in [0, 5] else f"+₹{v:.1f}L"
                  for i, v in enumerate([base, 2.1, 1.4, 0.9, 0.6, 0])],
            textposition="outside",
            textfont=dict(size=12, color="#FFFFFF"),
        ))
        return apply_layout(fig, "Revenue Uplift Waterfall — By Intervention", 380)


# ══════════════════════════════════════════════════════════════════════════════
#  Cached model loader
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Training prediction model…")
def load_all_modules():
    nsp = NoShowPredictionModel(n_samples=6000)
    nsp.train()
    return nsp


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem 0 0.5rem;'>
        <div style='font-family:Syne,sans-serif; font-size:1.1rem; font-weight:800; color:#FFFFFF;'>
            🏥 Clinic Optimizer
        </div>
        <div style='font-size:0.72rem; color:#8895A7; margin-top:0.2rem;'>
            AI Strategy Dashboard v1.0
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Scenario Controls")
    sim_no_show   = st.slider("Current No-Show Rate (%)", 10, 50, 30, 1) / 100
    sim_revenue   = st.slider("Revenue per Visit (₹)", 300, 2000, 700, 50)
    sim_fill_rate = st.slider("Waitlist Fill Rate (%)", 20, 85, 55, 5) / 100
    sim_overbook  = st.slider("Overbooking Buffer Cap (%)", 5, 25, 10, 1) / 100
    show_raw_data = st.checkbox("Show raw data tables", value=False)

    st.markdown("### 📊 Quick Metrics")
    _csa = CurrentStateAnalysis(sim_no_show, sim_revenue)
    st.markdown(f"""
    <div style='background:rgba(18,38,58,0.8); border:1px solid rgba(0,201,167,0.2);
                border-radius:10px; padding:0.9rem; font-size:0.82rem; color:#B8C8D8;'>
        <div style='margin-bottom:0.4rem;'>
            <span style='color:#8895A7;'>Monthly Revenue</span><br>
            <span style='font-family:Syne; font-weight:700; font-size:1.1rem; color:#00C9A7;'>
                ₹{_csa.actual_monthly/1e5:.2f}L
            </span>
        </div>
        <div style='margin-bottom:0.4rem;'>
            <span style='color:#8895A7;'>Lost to No-Shows</span><br>
            <span style='font-family:Syne; font-weight:700; font-size:1.1rem; color:#FF4757;'>
                ₹{_csa.lost_monthly/1e5:.2f}L
            </span>
        </div>
        <div>
            <span style='color:#8895A7;'>Slot Utilisation</span><br>
            <span style='font-family:Syne; font-weight:700; font-size:1.1rem; color:#1E90FF;'>
                {_csa.utilisation:.0%}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🔗 Report Sections")
    st.markdown("""
    <div style='font-size:0.82rem; color:#8895A7; line-height:2;'>
    Tab 1 — Situation Analysis<br>
    Tab 2 — No-Show Prediction<br>
    Tab 3 — Smart Overbooking<br>
    Tab 4 — Patient Segmentation<br>
    Tab 5 — Reminder Scheduler<br>
    Tab 6 — KPI Dashboard
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🏗️ Architecture")
    st.markdown("""
    <div style='font-size:0.78rem; color:#8895A7; line-height:1.9;'>
    Model: GradientBoosting (120 est.)<br>
    Segmentation: RFM 4-cluster<br>
    Overbooking: Conservative 10% cap<br>
    Channel: WhatsApp → SMS → Call<br>
    Data: 6,000 synthetic records
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="dash-header">
    <h1>Healthcare Clinic Optimization</h1>
    <p class="subtitle">
        End-to-end AI strategy simulation · 5-Clinic Chain · Strategy & Analytics Division · May 2026
    </p>
    <span class="tag">Report 1</span>
    <span class="tag">Live Simulation</span>
    <span class="tag">6 Modules Active</span>
    <span class="tag">Confidential</span>
</div>
""", unsafe_allow_html=True)

# ── Module instances ──────────────────────────────────────────────────────────
csa  = CurrentStateAnalysis(sim_no_show, sim_revenue)
obe  = SmartOverbookingEngine(initial_cap=sim_overbook)
ps   = PatientSegmentation(n=1500)
kpi  = KPIDashboard()
nsp  = load_all_modules()
rs   = ReminderScheduler()

# ── Top KPI row ───────────────────────────────────────────────────────────────
t12 = csa.at_target(0.14)
st.markdown("""<div class="metric-row">""", unsafe_allow_html=True)
col1, col2, col3, col4, col5 = st.columns(5)

def mcard(container, value, label, delta="", accent=PALETTE["teal"]):
    container.markdown(f"""
    <div class="metric-card" style="--card-accent:{accent};">
        <div class="mval">{value}</div>
        <div class="mlbl">{label}</div>
        {'<div class="mdelta" style="color:' + accent + ';">' + delta + '</div>' if delta else ''}
    </div>
    """, unsafe_allow_html=True)

mcard(col1, f"₹{csa.actual_monthly/1e5:.1f}L",
      "Monthly Revenue", f"Max ₹{csa.max_monthly/1e5:.1f}L", PALETTE["teal"])
mcard(col2, f"₹{csa.lost_monthly/1e5:.1f}L",
      "Lost to No-Shows", "Recoverable", PALETTE["red"])
mcard(col3, f"{csa.utilisation:.0%}",
      "Slot Utilisation", "Target: 86%", PALETTE["blue"])
mcard(col4, f"₹{t12['uplift']/1e5:.1f}L",
      "Potential Uplift (12M)", "At 14% no-show", PALETTE["green"])
mcard(col5, "50–100×",
      "Technology ROI", "₹8K–18K/mo cost", PALETTE["amber"])
st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📊 Situation Analysis",
    "🤖 No-Show Prediction",
    "📅 Smart Overbooking",
    "👥 Patient Segmentation",
    "📲 Reminder Scheduler",
    "🎯 KPI Dashboard",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — SITUATION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown('<div class="section-header">Current State Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Revenue metrics, slot utilisation, and the cost of inaction</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1])
    with c1:
        st.plotly_chart(csa.revenue_gap_chart(), use_container_width=True)
    with c2:
        st.plotly_chart(csa.utilisation_gauge(), use_container_width=True)
        st.markdown(f"""
        <div class="warn-box">
            ⚠️ <b>Critical Insight:</b> The chain is not capacity-constrained —
            it is <b>engagement-constrained</b>. ₹{csa.lost_monthly/1e5:.2f}L/month
            in perishable idle capacity is entirely recoverable without any new hires.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Revenue Scenarios</div>', unsafe_allow_html=True)
    scenarios = {
        "Current (30%)": 0.30, "3-Month Target (22%)": 0.22,
        "12-Month Target (14%)": 0.14, "Theoretical Max (0%)": 0.00
    }
    sc_cols = st.columns(4)
    for col, (name, rate) in zip(sc_cols, scenarios.items()):
        d = csa.at_target(rate)
        delta_color = PALETTE["red"] if rate == 0.30 else PALETTE["green"]
        col.markdown(f"""
        <div class="metric-card" style="--card-accent:{delta_color};">
            <div class="mlbl">{name}</div>
            <div class="mval" style="font-size:1.4rem;">₹{d['monthly_rev']/1e5:.1f}L</div>
            <div class="mdelta" style="color:{delta_color};">
                {'+₹' + f"{d['uplift']/1e5:.1f}L" if d['uplift'] > 0 else 'Baseline'}
            </div>
            <div class="mlbl" style="margin-top:0.3rem;">{d['util']:.0%} utilisation · {d['attended']}/day</div>
        </div>
        """, unsafe_allow_html=True)

    if show_raw_data:
        st.markdown("**Revenue Metrics Table**")
        summary_data = {
            "Metric": ["Total Slots/Day", "Patients Seen/Day", "No-Show Slots/Day",
                       "Actual Daily Revenue", "Actual Monthly Revenue", "Lost Daily Revenue", "Lost Monthly Revenue"],
            "Value": [csa.total_slots, csa.attended, csa.no_show_slots,
                      f"₹{csa.actual_daily:,}", f"₹{csa.actual_monthly:,}",
                      f"₹{csa.lost_daily:,}", f"₹{csa.lost_monthly:,}"],
        }
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — NO-SHOW PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown('<div class="section-header">AI No-Show Prediction Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">GradientBoosting classifier — 8 features · Risk-banded reminder routing</div>', unsafe_allow_html=True)

    # Model metrics
    eval_m = nsp.evaluate()
    auc    = eval_m["auc"]
    prec1  = eval_m["report"]["1"]["precision"]
    rec1   = eval_m["report"]["1"]["recall"]
    f1_1   = eval_m["report"]["1"]["f1-score"]

    m1, m2, m3, m4 = st.columns(4)
    for col, lbl, val, col_accent in [
        (m1, "ROC-AUC Score", f"{auc:.4f}", PALETTE["teal"]),
        (m2, "Precision (No-Show)", f"{prec1:.3f}", PALETTE["blue"]),
        (m3, "Recall (No-Show)", f"{rec1:.3f}", PALETTE["amber"]),
        (m4, "F1 Score (No-Show)", f"{f1_1:.3f}", PALETTE["purple"]),
    ]:
        mcard(col, val, lbl, accent=col_accent)

    st.markdown("")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(nsp.feature_importance_chart(), use_container_width=True)
    with c2:
        st.plotly_chart(nsp.risk_distribution_chart(), use_container_width=True)

    # ── Live Appointment Risk Scorer ──────────────────────────────────────────
    st.markdown('<div class="section-header">Live Risk Scorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Simulate the reminder protocol for a new appointment</div>', unsafe_allow_html=True)

    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        inp_age   = st.selectbox("Patient Age Group", [0, 1, 2],
                                 format_func=lambda x: ["18–30", "31–50", "51+"][x])
        inp_lead  = st.slider("Lead Days (booking → appt)", 1, 30, 7)
    with sc2:
        inp_dow   = st.selectbox("Day of Week", list(range(5)),
                                 format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri"][x])
        inp_tslot = st.selectbox("Time Slot", [0, 1, 2],
                                 format_func=lambda x: ["Morning","Afternoon","Evening"][x])
    with sc3:
        inp_past  = st.slider("Past No-Shows", 0, 8, 1)
        inp_dist  = st.slider("Distance (km)", 1, 40, 8)
    with sc4:
        inp_atype = st.selectbox("Appointment Type", [0, 1, 2],
                                 format_func=lambda x: ["Follow-up","New","Specialist"][x])
        inp_clinic= st.selectbox("Clinic ID", list(range(5)))

    if st.button("🔍 Score This Appointment"):
        feat = {
            "patient_age_group": inp_age, "appointment_lead_days": inp_lead,
            "day_of_week": inp_dow, "time_slot": inp_tslot,
            "past_no_show_count": inp_past, "distance_km": inp_dist,
            "appointment_type": inp_atype, "clinic_id": inp_clinic,
        }
        score = nsp.predict_score(feat)
        proto = nsp.reminder_protocol(score)
        band  = proto["risk_band"]
        band_css = {"Low": "risk-low", "Medium": "risk-medium", "High": "risk-high"}[band]
        band_color = {"Low": PALETTE["green"], "Medium": PALETTE["amber"], "High": PALETTE["red"]}[band]

        rc1, rc2, rc3 = st.columns(3)
        rc1.markdown(f"""
        <div class="metric-card" style="--card-accent:{band_color};">
            <div class="mval" style="font-size:2.5rem;">{score:.2f}</div>
            <div class="mlbl">No-Show Risk Score</div>
            <div style="margin-top:0.5rem;"><span class="{band_css}">{band} Risk</span></div>
        </div>
        """, unsafe_allow_html=True)
        rc2.markdown(f"""
        <div class="metric-card" style="--card-accent:{PALETTE['blue']};">
            <div class="mval">{proto['reminders']}</div>
            <div class="mlbl">Reminders Triggered</div>
            <div class="mdelta" style="color:{PALETTE['blue']};">{' → '.join(proto['touchpoints'])}</div>
        </div>
        """, unsafe_allow_html=True)
        rc3.markdown(f"""
        <div class="metric-card" style="--card-accent:{PALETTE['purple']};">
            <div class="mval" style="font-size:1rem;">{proto['channel']}</div>
            <div class="mlbl">Communication Channel</div>
            <div class="mdelta" style="color:{PALETTE['purple']};">
                Confirm required: {'✅ Yes' if proto['confirm'] else '❌ No'}
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — SMART OVERBOOKING
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown('<div class="section-header">Smart Overbooking Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Conservative 10% buffer per time band · Dynamic waitlist fill simulation</div>', unsafe_allow_html=True)

    plan = obe.overbooking_plan()
    total_uplift = plan["Monthly Uplift (₹)"].sum()

    ob1, ob2, ob3 = st.columns(3)
    mcard(ob1, f"₹{total_uplift/1e5:.2f}L", "Monthly Uplift (Overbooking)", "Conservative scenario", PALETTE["teal"])
    mcard(ob2, f"{plan['Buffer Slots'].sum()}", "Total Daily Buffer Slots", "Across all time bands", PALETTE["blue"])
    mcard(ob3, f"{sim_overbook:.0%}", "Overbooking Cap Applied", "Expandable after validation", PALETTE["amber"])

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(obe.overbooking_chart(), use_container_width=True)
    with c2:
        sim_wl = obe.waitlist_simulation(days=WORKING_DAYS, fill_rate=sim_fill_rate)
        st.plotly_chart(obe.waitlist_chart(sim_wl), use_container_width=True)

    st.markdown('<div class="section-header">Overbooking Plan by Time Band</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-box">
        💡 <b>Mechanism:</b> When a high-risk patient cancels or fails to confirm,
        the slot auto-opens and a WhatsApp notification fires to the next waitlisted patient.
        30-minute acceptance window. Conservative 10% cap prevents wait-time surges.
    </div>
    """, unsafe_allow_html=True)

    display_cols = ["Time Band", "No-Show %", "Buffer Slots", "Total Booked", "Daily Uplift (₹)", "Monthly Uplift (₹)"]
    st.dataframe(plan[display_cols].style.background_gradient(
        subset=["Buffer Slots"], cmap="Greens"
    ), use_container_width=True)

    # ── Waitlist stats ────────────────────────────────────────────────────────
    sw1, sw2, sw3 = st.columns(3)
    mcard(sw1, f"{sim_wl['Filled'].mean():.1f}", "Avg Daily Slots Filled from Waitlist", f"Fill rate: {sim_fill_rate:.0%}", PALETTE["green"])
    mcard(sw2, f"₹{sim_wl['Added Revenue (₹K)'].mean():.1f}K", "Avg Added Revenue/Day", "Via waitlist", PALETTE["teal"])
    mcard(sw3, f"₹{sim_wl['Added Revenue (₹K)'].sum():.0f}K", "Total Monthly Waitlist Revenue", "Monte-Carlo sim", PALETTE["blue"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — PATIENT SEGMENTATION
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown('<div class="section-header">RFM Patient Segmentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Recency · Frequency · Monetary — 4 behavioural segments with differentiated engagement</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(ps.pie_chart(), use_container_width=True)
    with c2:
        st.plotly_chart(ps.rfm_scatter(), use_container_width=True)

    st.markdown('<div class="section-header">Segment Strategy Matrix</div>', unsafe_allow_html=True)
    seg_cols = st.columns(4)
    seg_summary = ps.summary()

    for col, (seg, cfg) in zip(seg_cols, ps.SEGMENT_CONFIG.items()):
        row = seg_summary[seg_summary["segment"] == seg]
        count = int(row["count"].values[0]) if len(row) else 0
        share = float(row["share_%"].values[0]) if len(row) else 0
        avg_ns= float(row["avg_no_shows"].values[0]) if len(row) else 0
        col.markdown(f"""
        <div class="metric-card" style="--card-accent:{cfg['color']};">
            <div class="mlbl">{seg}</div>
            <div class="mval" style="font-size:1.5rem; color:{cfg['color']};">{share:.1f}%</div>
            <div style="font-size:0.8rem; color:#B8C8D8; margin-top:0.4rem;">
                <b>{count}</b> patients
            </div>
            <div style="font-size:0.75rem; color:#8895A7; margin-top:0.3rem; line-height:1.6;">
                📲 {cfg['channel']}<br>
                🔔 {cfg['reminders']} reminder(s)<br>
                🎯 {cfg['goal']}<br>
                ⚠️ Avg no-shows: {avg_ns:.1f}
            </div>
        </div>
        """, unsafe_allow_html=True)

    if show_raw_data:
        st.markdown("**Segment Summary Table**")
        st.dataframe(seg_summary.style.background_gradient(
            subset=["avg_no_shows"], cmap="RdYlGn_r"
        ), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — REMINDER SCHEDULER
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown('<div class="section-header">Automated Reminder Scheduler</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">WhatsApp → SMS → Voice fallback · Risk-adaptive touchpoints</div>', unsafe_allow_html=True)

    with st.spinner("Generating daily schedule…"):
        daily_sched = rs.daily_schedule(nsp)
        cost_info   = rs.cost_summary(daily_sched)

    rem1, rem2, rem3, rem4 = st.columns(4)
    mcard(rem1, f"{cost_info['total_messages']:,}", "Daily Messages Sent", "", PALETTE["teal"])
    mcard(rem2, f"₹{cost_info['daily_cost']:.0f}", "Daily Cost (₹)", "", PALETTE["blue"])
    mcard(rem3, f"₹{cost_info['monthly_cost']:,.0f}", "Monthly Cost (₹)", "All channels", PALETTE["amber"])
    mcard(rem4, f"≥70%", "Target Open Rate", "WhatsApp benchmark", PALETTE["green"])

    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(rs.band_breakdown_chart(daily_sched), use_container_width=True)
    with c2:
        # Reminder timeline visual
        st.markdown('<div class="section-header" style="font-size:1rem;">Reminder Cascade Logic</div>', unsafe_allow_html=True)
        for band, color, steps in [
            ("🟢 Low Risk (<0.30)",    PALETTE["teal"],   ["24h → WhatsApp only"]),
            ("🟡 Medium Risk (0.30–0.60)", PALETTE["amber"], ["48h → WhatsApp", "2h → SMS fallback"]),
            ("🔴 High Risk (>0.60)",   PALETTE["red"],    ["48h → WhatsApp", "24h → SMS", "2h → Call + mandatory confirm"]),
        ]:
            steps_html = "".join(
                f'<div style="padding:0.25rem 0; border-bottom:1px solid rgba(255,255,255,0.05);">'
                f'→ {s}</div>' for s in steps
            )
            st.markdown(f"""
            <div style="background:rgba(18,38,58,0.8); border-left:3px solid {color};
                        border-radius:0 10px 10px 0; padding:0.8rem 1rem; margin-bottom:0.7rem;">
                <div style="font-weight:600; color:{color}; margin-bottom:0.4rem;">{band}</div>
                <div style="font-size:0.82rem; color:#B8C8D8;">{steps_html}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="success-box">
        ✅ <b>Technology cost: ₹{cost_info['monthly_cost'] * WORKING_DAYS / WORKING_DAYS:,.0f}/month</b>
        (monthly) against an estimated revenue recovery of <b>₹5.5–8.5L/month</b>.
        ROI: <b>50–100×</b> on tech spend. All tools operate without engineering staff (Wati / Interakt / MSG91).
    </div>
    """, unsafe_allow_html=True)

    if show_raw_data:
        st.markdown("**Daily Schedule Sample (first 20 rows)**")
        st.dataframe(daily_sched.head(20).style.background_gradient(
            subset=["score"], cmap="RdYlGn_r"
        ), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — KPI DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown('<div class="section-header">12-Month KPI Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Projected outcomes for all 4 strategic interventions combined</div>', unsafe_allow_html=True)

    k1, k2 = st.columns(2)
    with k1:
        st.plotly_chart(kpi.projection_chart(), use_container_width=True)
    with k2:
        st.plotly_chart(kpi.waterfall_chart(), use_container_width=True)

    # ── KPI Scorecard ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">KPI Scorecard — Phase Targets</div>', unsafe_allow_html=True)

    kpi_rows = [
        ("No-show rate (chain avg.)", "30%",    "≤ 22%",   "≤ 14%"),
        ("Daily patients seen",       "140",     "155",     "172"),
        ("Monthly revenue",           "₹25.5L",  "₹29–30L", "₹32–34L"),
        ("Reminder open/response",    "0%",      "≥ 60%",   "≥ 70%"),
        ("Cancellation-to-reschedule","~0%",     "≥ 25%",   "≥ 40%"),
        ("Slot utilisation rate",     "70%",     "77%",     "86%"),
        ("At-risk patient reactivation","N/A",   "N/A",     "≥ 20%"),
        ("Waitlist fill rate",        "N/A",     "N/A",     "≥ 50%"),
    ]

    header_cols = st.columns([2, 1, 1, 1])
    for col, lbl in zip(header_cols, ["KPI Metric", "Baseline", "3-Month", "12-Month"]):
        col.markdown(f"<div style='font-family:Syne; font-size:0.78rem; font-weight:700; color:#8895A7; text-transform:uppercase; letter-spacing:1px;'>{lbl}</div>", unsafe_allow_html=True)

    st.markdown("<hr style='border:1px solid rgba(255,255,255,0.06); margin:0.3rem 0 0.8rem;'>", unsafe_allow_html=True)

    _blue  = PALETTE["blue"]
    _teal  = PALETTE["teal"]
    for metric, base, t3, t12 in kpi_rows:
        row_cols = st.columns([2, 1, 1, 1])
        row_cols[0].markdown(f"<div style='font-size:0.88rem; color:#B8C8D8; padding:0.3rem 0;'>{metric}</div>", unsafe_allow_html=True)
        row_cols[1].markdown(f"<div style='font-size:0.88rem; color:#8895A7; padding:0.3rem 0;'>{base}</div>", unsafe_allow_html=True)
        row_cols[2].markdown(f"<div style='font-size:0.88rem; color:{_blue}; font-weight:600; padding:0.3rem 0;'>{t3}</div>", unsafe_allow_html=True)
        row_cols[3].markdown(f"<div style='font-size:0.88rem; color:{_teal}; font-weight:600; padding:0.3rem 0;'>{t12}</div>", unsafe_allow_html=True)

    # ── Summary outcome cards ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">Projected Outcomes Summary</div>', unsafe_allow_html=True)
    so1, so2, so3 = st.columns(3)

    for col, phase, data, color in [
        (so1, "BASELINE", {"No-show rate": "30%", "Patients/day": "140",
                           "Monthly revenue": "₹25.48L", "Utilisation": "70%",
                           "Revenue uplift": "—"}, PALETTE["gray"]),
        (so2, "3 MONTHS", {"No-show rate": "22%", "Patients/day": "155",
                           "Monthly revenue": "₹29–30L", "Utilisation": "77%",
                           "Revenue uplift": "+₹3.5–4.5L"}, PALETTE["blue"]),
        (so3, "12 MONTHS", {"No-show rate": "12–15%", "Patients/day": "170–176",
                            "Monthly revenue": "₹32–34L", "Utilisation": "85–88%",
                            "Revenue uplift": "+₹6.5–8.5L"}, PALETTE["teal"]),
    ]:
        rows_html = "".join(
            f'<div style="display:flex; justify-content:space-between; '
            f'padding:0.3rem 0; border-bottom:1px solid rgba(255,255,255,0.05);">'
            f'<span style="color:#8895A7; font-size:0.8rem;">{k}</span>'
            f'<span style="color:#FFFFFF; font-size:0.8rem; font-weight:600;">{v}</span>'
            f'</div>'
            for k, v in data.items()
        )
        col.markdown(f"""
        <div class="metric-card" style="--card-accent:{color};">
            <div style="font-family:Syne; font-size:0.9rem; font-weight:700;
                        color:{color}; text-transform:uppercase; letter-spacing:2px;
                        margin-bottom:0.8rem;">{phase}</div>
            {rows_html}
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<br><hr style='border:1px solid rgba(255,255,255,0.06);'>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#4A5568; font-family:'Mulish',sans-serif;
            font-size:0.78rem; padding:0.8rem 0 0.3rem;">
    Healthcare Clinic Optimization Dashboard · Report 1 · Strategy & Analytics Division · May 2026<br>
    All projections are model estimates based on stated operating parameters and India outpatient healthcare benchmarks.
    Simulation uses synthetic data representative of clinic appointment patterns.
</div>
""", unsafe_allow_html=True)