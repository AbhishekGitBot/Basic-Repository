#  Healthcare Clinic Optimization — AI Strategy Dashboard

> **Report 1**
> End-to-end AI strategy simulation for a 5-clinic outpatient chain — no-show reduction, smart overbooking, patient segmentation, and automated reminders.

---

## Overview

This project implements the full business strategy from **Report 1: Healthcare Clinic Optimization** as an interactive Streamlit dashboard. The chain currently loses **₹10.92 lakh/month** to a 30% no-show rate — above India's outpatient benchmark of 15–20%. Every recommendation targets that gap without adding a single doctor, clinic, or price change.

The dashboard runs six strategy modules end-to-end, from current-state revenue analysis through a live ML-powered appointment risk scorer, and projects a 12-month path from ₹25.5L to ₹32–34L/month in recovered revenue.

---

## The Business Problem

| Parameter | Value |
|---|---|
| Clinics | 5 |
| Scheduled appointments/day | 200 |
| No-show rate | **30%** (benchmark: 15–20%) |
| Actual patients seen/day | 140 |
| Revenue per visit | ₹700 |
| **Monthly revenue lost** | **₹10.92L** |
| Technology cost to fix it | ₹8,000–18,000/month |
| **Estimated ROI** | **50–100×** |

The chain is not capacity-constrained. It is engagement-constrained.

---

## Modules

### Module 1 — Current State Analysis
Calculates actual vs. theoretical revenue, slot utilisation, and the cost of inaction across four no-show rate scenarios (30% → 22% → 14% → 0%).

### Module 2 — No-Show Prediction Model
GradientBoosting classifier trained on 6,000 synthetic appointment records. Outputs a 0–1 risk score per appointment and routes it to a three-tier reminder protocol:

| Risk Band | Score | Reminders | Channel |
|---|---|---|---|
| Low | < 0.30 | 1 | WhatsApp |
| Medium | 0.30–0.60 | 2 | WhatsApp + SMS |
| High | > 0.60 | 3 | WhatsApp + SMS + Call |

Includes a **live risk scorer** — enter patient details and get an instant band and protocol recommendation.

### Module 3 — Smart Overbooking Engine
Calculates conservative (10% cap) overbooking buffers per time band based on historical no-show rates. Runs a Monte-Carlo waitlist fill simulation across 26 working days.

### Module 4 — Patient Segmentation (RFM)
Recency–Frequency–Monetary segmentation across the patient base into four behavioural cohorts — Champions, At-Risk, High No-Show, and New Patients — each with a differentiated WhatsApp engagement strategy.

### Module 5 — Automated Reminder Scheduler
Generates a full daily 200-slot reminder schedule, computes per-channel costs (WhatsApp / SMS / Call), and shows the band distribution and cascade logic.

### Module 6 — KPI Dashboard
12-month dual-axis projection (revenue + no-show rate), revenue waterfall broken down by intervention, and a full phase-target scorecard (Baseline → 3-Month → 12-Month).

---

## Projected Outcomes

| | Baseline | 3 Months | 12 Months |
|---|---|---|---|
| No-show rate | 30% | ≤ 22% | ≤ 14% |
| Patients/day | 140 | 155 | 170–176 |
| Monthly revenue | ₹25.5L | ₹29–30L | ₹32–34L |
| Slot utilisation | 70% | 77% | 85–88% |
| Revenue uplift | — | +₹3.5–4.5L | +₹6.5–8.5L |

---

## Project Structure

```
├── healthcare.py                  # Streamlit dashboard (all 6 modules)
├── Healthcare Report.ipynb      # Core strategy implementation (Matplotlib version)
├── README.md
└── Healthcare Clinic Optimization_Report_1.pdf   # Full business report
```

`healthcare.py` is the primary deliverable — a self-contained interactive dashboard. `Healthcare Report.ipynb` contains the same six modules as a standalone script that generates a static PNG master dashboard.

---

## Setup

**Requirements**
```
Python 3.9+
streamlit
plotly
scikit-learn
pandas
numpy
```

**Install and run**
```bash
git clone https://github.com/<your-username>/healthcare-clinic-optimization.git
cd healthcare-clinic-optimization

pip install -r requirements.txt

streamlit run healthcare.py
```

**Or run the static report generator**
```bash
python Healthcare Report.ipynb
# Outputs: clinic_optimization_dashboard.png
```

`requirements.txt`:
```
streamlit>=1.35.0
plotly>=5.20.0
scikit-learn>=1.4.0
pandas>=2.0.0
numpy>=1.26.0
```

---

## Dashboard Features

- **Sidebar scenario controls** — adjust no-show rate, revenue per visit, waitlist fill rate, and overbooking cap live; all charts update instantly
- **Live ML risk scorer** — enter 8 appointment features and receive a risk score + reminder protocol in real time
- **Plotly interactive charts** — hover, zoom, and export on all 10+ visualisations
- **Raw data toggle** — expose underlying DataFrames for any module
- **Dark clinical theme** — Syne + Mulish typography, teal/navy palette

---

## Key Design Decisions

**Why GradientBoosting over Logistic Regression?** The non-linear interaction between past no-show count, lead time, and distance is better captured by an ensemble. ROC-AUC consistently lands at 0.74–0.78 on the synthetic dataset, sufficient for risk-band routing.

**Why synthetic data?** The model architecture, feature set, and risk-band thresholds are production-ready. Real deployment requires substituting the synthetic DataFrame with a clinic's historical appointment export (6–12 months minimum).

**Why a 10% overbooking cap?** Aggressive overbooking risks wait-time complaints and patient attrition — the single highest-cost outcome for a price-sensitive patient base. The cap is parameterised and can be raised incrementally as validation data accumulates.

---

## Recommended Tool Stack (Production)

| Layer | Tool | Cost/Month |
|---|---|---|
| WhatsApp automation | Wati / Interakt / AiSensy | ₹2,000–5,000 |
| SMS fallback | MSG91 / Textlocal | Pay-per-message |
| Voice reminders | Exotel / Sarvam AI IVR | ₹3,000–8,000 |
| ML prediction | Google Colab / DataRobot | Free |
| CRM & segments | Zoho CRM / Airtable | Free–₹1,500 |
| Scheduling | Practo / Sheets + Zapier | Free |
| Analytics | Looker Studio | Free |
| **Total** | | **₹8,000–18,000** |

---

## Author

**Abhishek Sharma**
 Strategy & Analytics Division · May 2026

---

*All projections are model estimates based on stated operating parameters and India outpatient healthcare benchmarks. Simulation uses synthetic data representative of clinic appointment patterns.*
