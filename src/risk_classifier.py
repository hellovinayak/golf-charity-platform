"""
risk_classifier.py
NITI Aayog-aligned 3-Tier Risk Classification & Early Warning Engine:
- Low Risk (>= 75): Front Runner / Achiever
- Medium Risk (50 - 74): Performer
- High Risk (< 50): Aspirant (Critical Intervention Required)
"""

import pandas as pd
import numpy as np

def classify_score(score):
    """Assigns risk category, label, and UI badge color based on NITI Aayog scale."""
    if score >= 75:
        return {
            "Risk_Category": "Low Risk",
            "NITI_Tier": "Front Runner",
            "Risk_Score": 1,
            "Color": "#10B981",  # Emerald Green
            "Badge": "🟢 Low Risk (Front Runner)"
        }
    elif score >= 50:
        return {
            "Risk_Category": "Medium Risk",
            "NITI_Tier": "Performer",
            "Risk_Score": 2,
            "Color": "#F59E0B",  # Amber Yellow
            "Badge": "🟡 Medium Risk (Performer)"
        }
    else:
        return {
            "Risk_Category": "High Risk",
            "NITI_Tier": "Aspirant",
            "Risk_Score": 3,
            "Color": "#EF4444",  # Crimson Red
            "Badge": "🔴 High Risk (Aspirant - Alert)"
        }

def enrich_with_risk_labels(df, score_col="Score"):
    """Enriches any dataframe containing a score column with full risk attributes."""
    results = df.copy()
    categories = []
    tiers = []
    risk_scores = []
    badges = []
    colors = []

    for val in results[score_col]:
        meta = classify_score(val)
        categories.append(meta["Risk_Category"])
        tiers.append(meta["NITI_Tier"])
        risk_scores.append(meta["Risk_Score"])
        badges.append(meta["Badge"])
        colors.append(meta["Color"])

    results["Risk_Category"] = categories
    results["NITI_Tier"] = tiers
    results["Risk_Score"] = risk_scores
    results["Risk_Badge"] = badges
    results["Risk_Color"] = colors
    return results

def get_early_warning_alerts(forecast_df, historical_df):
    """
    Identifies high-priority warning triggers:
    1. Threshold Breach: States falling below critical 50-point Aspirant mark in any forecast year.
    2. Borderline Vulnerability: States projected between 50 and 53 points (high risk of relapse).
    3. Chronic Aspirant: States starting <50 and remaining vulnerable.
    4. Growth Stagnation / Sluggish Trajectory: Annual growth rate < 1.3 pts/year in lagging areas (<60).
    """
    alerts = []
    latest_2023 = historical_df[historical_df["Year"] == 2023].copy()
    
    for (state, sdg), grp in forecast_df.groupby(["State", "SDG"]):
        hist_val_rows = latest_2023[(latest_2023["State"] == state)]
        if hist_val_rows.empty:
            continue
        
        hist_2023_val = float(hist_val_rows[sdg].values[0]) if sdg in hist_val_rows.columns else 50.0
        
        scores_by_year = {int(row["Year"]): float(row["Forecast_Score"]) for _, row in grp.iterrows()}
        val_2024 = scores_by_year.get(2024, hist_2023_val)
        val_2025 = scores_by_year.get(2025, hist_2023_val)
        val_2026 = scores_by_year.get(2026, hist_2023_val)

        growth_rate = float(grp["Annual_Growth_Rate"].values[0]) if "Annual_Growth_Rate" in grp.columns else 0.0

        sdg_label = {
            "SDG3_Health": "SDG 3 (Health)",
            "SDG4_Education": "SDG 4 (Education)",
            "SDG13_Climate": "SDG 13 (Climate Action)"
        }.get(sdg, sdg)

        # Trigger 1: Aspirant (<50) during forecast horizon
        if min(val_2024, val_2025, val_2026) < 50.0:
            alerts.append({
                "State": state,
                "SDG": sdg_label,
                "Severity": "CRITICAL",
                "Alert_Type": "Aspirant Tier Alert (<50 pts)",
                "Description": f"Projected below critical 50-point threshold (2024: {val_2024:.1f}, 2026: {val_2026:.1f}). Requires emergency intervention.",
                "2023_Baseline": hist_2023_val,
                "2026_Projected": val_2026
            })
        
        # Trigger 2: Borderline Vulnerability (50.0 - 53.0)
        elif 50.0 <= min(val_2024, val_2025, val_2026) <= 53.0:
            alerts.append({
                "State": state,
                "SDG": sdg_label,
                "Severity": "WARNING",
                "Alert_Type": "Borderline Vulnerability (50–53 pts)",
                "Description": f"Operating on knife-edge of Aspirant band (2026: {val_2026:.1f}). Susceptible to target regression.",
                "2023_Baseline": hist_2023_val,
                "2026_Projected": val_2026
            })

        # Trigger 3: Sluggish Trajectory (<1.3 pts/yr while score < 60)
        elif growth_rate < 1.3 and val_2026 < 60.0:
            alerts.append({
                "State": state,
                "SDG": sdg_label,
                "Severity": "ADVISORY",
                "Alert_Type": "Sluggish Developmental Velocity",
                "Description": f"Annual velocity of +{growth_rate:.2f} pts/yr is below national pace needed to exit Performer band.",
                "2023_Baseline": hist_2023_val,
                "2026_Projected": val_2026
            })

    return pd.DataFrame(alerts)
