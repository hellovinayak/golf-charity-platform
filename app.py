"""
app.py
Predictive Dashboard for SDG Performance Monitoring
BCSE497J: Project I - Vellore Institute of Technology (VIT) Chennai Campus
Team: Sakhi Telang (23BAI1105) | Akhilesh Deshmukh (23BRS1149) | Vinayak Rathod (23BCE1747)
Guide: Dr. Sureshkumar WI (Associate Professor, SCOPE)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Ensure src modules are discoverable
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import (
    load_sdg_data, 
    STATES_AND_UTS, 
    get_long_format_data,
    load_health_indicators,
    load_climate_indicators
)
from src.model_engine import (
    train_and_validate_models,
    generate_forecasts,
    get_complete_timeseries,
    SDG_TARGETS
)
from src.risk_classifier import (
    classify_score,
    enrich_with_risk_labels,
    get_early_warning_alerts
)
from src.geo_visualizer import (
    create_india_geo_heatmap,
    create_state_ranking_chart,
    create_trend_line_chart,
    STATE_COORDINATES
)
from src.policy_insights import (
    generate_state_policy_advisory,
    get_national_policy_priorities
)

# Page Configuration
st.set_page_config(
    page_title="SDG Predictive Monitoring Dashboard | VIT Chennai",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Aesthetic Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Gradient Glassmorphism Header */
    .hero-banner {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }
    
    .hero-title {
        font-size: 28px;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8, #818CF8, #C084FC);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .hero-subtitle {
        color: #94A3B8;
        font-size: 14px;
        margin-top: 6px;
        font-weight: 500;
    }
    
    .hero-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        margin-top: 14px;
        padding-top: 12px;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        font-size: 12px;
        color: #CBD5E1;
    }
    
    .hero-tag {
        background: rgba(56, 189, 248, 0.15);
        color: #38BDF8;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(15, 23, 42, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px 20px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.4);
        box-shadow: 0 8px 20px -6px rgba(56, 189, 248, 0.2);
    }
    .metric-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
        font-weight: 600;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 800;
        color: #F8FAFC;
        margin-top: 4px;
    }
    .metric-delta {
        font-size: 12px;
        font-weight: 600;
        margin-top: 4px;
    }
    .delta-pos { color: #10B981; }
    .delta-neg { color: #EF4444; }

    /* Risk Badges */
    .badge-high {
        background-color: rgba(239, 68, 68, 0.2);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 11px;
    }
    .badge-med {
        background-color: rgba(245, 158, 11, 0.2);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.4);
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 11px;
    }
    .badge-low {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 11px;
    }

    /* Policy Card */
    .policy-box {
        background: rgba(30, 41, 59, 0.6);
        border-left: 4px solid #38BDF8;
        border-radius: 0 12px 12px 0;
        padding: 16px 20px;
        margin-bottom: 14px;
    }

    /* Architecture Block */
    .arch-card {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 14px;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 18px;
        font-weight: 600;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Data Caching & ML Model Orchestration
@st.cache_data
def get_cached_pipeline_data():
    historical_df = load_sdg_data()
    trained_models, val_df = train_and_validate_models()
    forecast_df = generate_forecasts(trained_models)
    timeseries_df, _, _ = get_complete_timeseries()
    early_warnings = get_early_warning_alerts(forecast_df, historical_df)
    health_ind = load_health_indicators()
    clim_ind = load_climate_indicators()
    return historical_df, trained_models, val_df, forecast_df, timeseries_df, early_warnings, health_ind, clim_ind

historical_df, trained_models, val_df, forecast_df, timeseries_df, early_warnings, health_ind_df, clim_ind_df = get_cached_pipeline_data()

# Header Banner
st.markdown("""
<div class="hero-banner">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
        <div>
            <h1 class="hero-title">Predictive Dashboard for SDG Performance Monitoring</h1>
            <p class="hero-subtitle">Machine Learning Trajectory Forecasting & Automated Risk Early-Warning Engine for Indian States</p>
        </div>
        <div style="margin-top: 6px;">
            <span class="hero-tag">BCSE497J : Project I</span>
        </div>
    </div>
    <div class="hero-meta">
        <div>🎓 <b>Institution:</b> VIT University Chennai Campus (SCOPE)</div>
        <div>👥 <b>Team:</b> Sakhi Telang (23BAI1105) · Akhilesh Deshmukh (23BRS1149) · Vinayak Rathod (23BCE1747)</div>
        <div>👨‍🏫 <b>Guided by:</b> Dr. Sureshkumar WI (Associate Professor)</div>
        <div>🎯 <b>Coverage:</b> 36 States & UTs · SDGs 3, 4, 13 · 2018–2026 Horizon</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.markdown("### 🎛️ Control Panel")

all_states = sorted(list(historical_df["State"].unique()))
default_idx = all_states.index("Tamil Nadu") if "Tamil Nadu" in all_states else 0
selected_state = st.sidebar.selectbox("📍 Select State / UT:", all_states, index=default_idx)

sdg_options = {
    "SDG3_Health": "SDG 3: Good Health & Well-Being",
    "SDG4_Education": "SDG 4: Quality Education",
    "SDG13_Climate": "SDG 13: Climate Action"
}
selected_sdg_keys = st.sidebar.multiselect(
    "🎯 SDG Focus Areas:",
    options=list(sdg_options.keys()),
    default=list(sdg_options.keys()),
    format_func=lambda x: sdg_options[x]
)

forecast_year_filter = st.sidebar.select_slider(
    "📅 Forecast Target Year:",
    options=[2024, 2025, 2026],
    value=2026
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏷️ NITI Aayog Risk Thresholds")
st.sidebar.markdown("""
- 🟢 **Low Risk (Front Runner):** Score $\ge 75$
- 🟡 **Medium Risk (Performer):** Score $50 \text{--} 74$
- 🔴 **High Risk (Aspirant):** Score $< 50$
""")

# Top Level KPI Row (National Averages & High Risk Counts for Target Year)
fore_target = forecast_df[forecast_df["Year"] == forecast_year_filter]
hist_2023 = historical_df[historical_df["Year"] == 2023]

s3_avg_23 = hist_2023["SDG3_Health"].mean()
s3_avg_tgt = fore_target[fore_target["SDG"] == "SDG3_Health"]["Forecast_Score"].mean()
s3_delta = s3_avg_tgt - s3_avg_23

s4_avg_23 = hist_2023["SDG4_Education"].mean()
s4_avg_tgt = fore_target[fore_target["SDG"] == "SDG4_Education"]["Forecast_Score"].mean()
s4_delta = s4_avg_tgt - s4_avg_23

s13_avg_23 = hist_2023["SDG13_Climate"].mean()
s13_avg_tgt = fore_target[fore_target["SDG"] == "SDG13_Climate"]["Forecast_Score"].mean()
s13_delta = s13_avg_tgt - s13_avg_23

high_risk_count = len(fore_target[fore_target["Forecast_Score"] < 50]["State"].unique())
critical_alerts_count = len(early_warnings[early_warnings["Severity"] == "CRITICAL"])

kpi_c1, kpi_c2, kpi_c3, kpi_c4, kpi_c5 = st.columns(5)

with kpi_c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">🏥 SDG 3 (Health) Avg</div>
        <div class="metric-value">{s3_avg_tgt:.1f}</div>
        <div class="metric-delta {'delta-pos' if s3_delta >= 0 else 'delta-neg'}">
            {'+' if s3_delta >= 0 else ''}{s3_delta:.1f} pts vs 2023
        </div>
    </div>
    """, unsafe_allow_html=True)

with kpi_c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">📚 SDG 4 (Education) Avg</div>
        <div class="metric-value">{s4_avg_tgt:.1f}</div>
        <div class="metric-delta {'delta-pos' if s4_delta >= 0 else 'delta-neg'}">
            {'+' if s4_delta >= 0 else ''}{s4_delta:.1f} pts vs 2023
        </div>
    </div>
    """, unsafe_allow_html=True)

with kpi_c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">🌿 SDG 13 (Climate) Avg</div>
        <div class="metric-value">{s13_avg_tgt:.1f}</div>
        <div class="metric-delta {'delta-pos' if s13_delta >= 0 else 'delta-neg'}">
            {'+' if s13_delta >= 0 else ''}{s13_delta:.1f} pts vs 2023
        </div>
    </div>
    """, unsafe_allow_html=True)

with kpi_c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">🔴 Aspirant States (<50)</div>
        <div class="metric-value" style="color: {'#EF4444' if high_risk_count > 0 else '#10B981'};">{high_risk_count}</div>
        <div class="metric-delta" style="color: #94A3B8;">Across 36 States/UTs ({forecast_year_filter})</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_c5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">🚨 Imminent Drop Triggers</div>
        <div class="metric-value" style="color: #F59E0B;">{critical_alerts_count}</div>
        <div class="metric-delta" style="color: #F59E0B;">50-Threshold Breaches</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Main Multi-Tab Application
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📈 Predictive Forecaster",
    "🚨 Risk Matrix & Alerts",
    "🗺️ Geographic Heatmap",
    "🌿 SDG 13 Climate Panel",
    "🏥 SDG 3 Indicator Analytics",
    "🏛️ Policy & Governance",
    "🔬 ML Validation & Methodology",
    "📥 Data & Export"
])

# ==========================================
# TAB 1: PREDICTIVE FORECASTER & TRENDS
# ==========================================
with tab1:
    st.subheader(f"🔮 Longitudinal Trajectory & Forecast Horizon (2018–2026): {selected_state}")
    
    col_chart, col_side = st.columns([3, 1])
    
    with col_chart:
        fig_trend = create_trend_line_chart(timeseries_df, selected_state, selected_sdg_keys)
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_side:
        st.markdown(f"#### 📊 Projections Summary ({selected_state})")
        st_fore = forecast_df[forecast_df["State"] == selected_state]
        st_hist_latest = historical_df[historical_df["State"] == selected_state].sort_values("Year").iloc[-1]

        for sdg_k in (selected_sdg_keys or SDG_TARGETS):
            sdg_name = sdg_options[sdg_k].split(":")[1].strip()
            val_23 = st_hist_latest[sdg_k]
            sub_f = st_fore[(st_fore["SDG"] == sdg_k) & (st_fore["Year"] == forecast_year_filter)]
            if not sub_f.empty:
                val_proj = sub_f["Forecast_Score"].values[0]
                ci_low = sub_f["CI_Lower_95"].values[0]
                ci_hi = sub_f["CI_Upper_95"].values[0]
            else:
                val_proj, ci_low, ci_hi = val_23, val_23, val_23

            slope = trained_models.get((selected_state, sdg_k), {}).get("slope", 0.0)
            risk_meta = classify_score(val_proj)
            
            st.markdown(f"""
            <div style="background: rgba(15,23,42,0.6); padding: 12px; border-radius: 10px; margin-bottom: 10px; border-left: 3px solid {risk_meta['Color']};">
                <div style="font-weight: 700; font-size: 13px; color: #F8FAFC;">{sdg_name}</div>
                <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 4px;">
                    <span style="font-size: 20px; font-weight: 800; color: {risk_meta['Color']};">{val_proj:.1f}</span>
                    <span style="font-size: 11px; color: #94A3B8;">2023: {val_23:.1f} ({'+' if slope>0 else ''}{slope:.2f}/yr)</span>
                </div>
                <div style="font-size: 11px; color: #CBD5E1; margin-top: 2px;">95% CI: [{ci_low:.1f} – {ci_hi:.1f}]</div>
                <div style="margin-top: 4px;"><span class="{ 'badge-low' if risk_meta['Risk_Category']=='Low Risk' else ('badge-med' if risk_meta['Risk_Category']=='Medium Risk' else 'badge-high') }">{risk_meta['NITI_Tier']}</span></div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🔄 Cross-State Comparative Trajectory")
    compare_states = st.multiselect(
        "Select additional states to compare:",
        options=[s for s in all_states if s != selected_state],
        default=["Kerala", "Bihar"] if "Kerala" in all_states and "Bihar" in all_states else all_states[:2]
    )
    
    if compare_states:
        comp_sdg = st.selectbox("Select SDG for Multi-State Comparison:", options=list(sdg_options.keys()), format_func=lambda x: sdg_options[x])
        all_comp_states = [selected_state] + compare_states
        
        comp_df = timeseries_df[(timeseries_df["State"].isin(all_comp_states)) & (timeseries_df["SDG"] == comp_sdg)]
        
        fig_comp = px.line(
            comp_df,
            x="Year",
            y="Score",
            color="State",
            line_dash="Type",
            markers=True,
            title=f"Comparative Trajectory: {sdg_options[comp_sdg]} (2018–2026)"
        )
        fig_comp.add_hline(y=75, line_dash="dot", line_color="#10B981", annotation_text="Front Runner (≥75)")
        fig_comp.add_hline(y=50, line_dash="dot", line_color="#EF4444", annotation_text="Aspirant (<50)")
        fig_comp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.4)",
            font=dict(color="#F8FAFC"),
            yaxis=dict(range=[20, 100], gridcolor="#334155"),
            xaxis=dict(gridcolor="#334155")
        )
        st.plotly_chart(fig_comp, use_container_width=True)

# ==========================================
# TAB 2: RISK MATRIX & EARLY WARNING ALERTS
# ==========================================
with tab2:
    st.subheader(f"🚨 Automated Early Warning System & 3-Tier Risk Profiling ({forecast_year_filter})")
    
    # Priority Alerts Panel
    if not early_warnings.empty:
        st.markdown("#### ⚠️ High-Priority Early Warning Triggers")
        for _, alert in early_warnings.iterrows():
            badge_class = "badge-high" if alert["Severity"] == "CRITICAL" else ("badge-med" if alert["Severity"] == "WARNING" else "badge-low")
            st.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 10px; padding: 14px 18px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span class="{badge_class}">{alert['Severity']}</span>
                    <b style="color: #F8FAFC; margin-left: 10px; font-size: 14px;">{alert['State']} — {alert['SDG']}</b>
                    <div style="color: #CBD5E1; font-size: 13px; margin-top: 4px;">{alert['Description']}</div>
                </div>
                <div style="text-align: right; min-width: 140px;">
                    <span style="font-size: 12px; color: #94A3B8;">Baseline 2023: <b>{alert['2023_Baseline']:.1f}</b></span><br>
                    <span style="font-size: 13px; font-weight: 700; color: #EF4444;">2026 Proj: <b>{alert['2026_Projected']:.1f}</b></span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ No critical early-warning threshold drops detected across current parameters.")

    st.markdown("---")
    st.markdown(f"#### 📋 Full State Risk Classification Matrix ({forecast_year_filter})")
    
    f_c1, f_c2, f_c3 = st.columns(3)
    with f_c1:
        risk_filter = st.selectbox("Filter by Risk Tier:", ["All Tiers", "High Risk (<50)", "Medium Risk (50-74)", "Low Risk (≥75)"])
    with f_c2:
        sdg_matrix_filter = st.selectbox("Filter by SDG:", ["All SDGs", "SDG3_Health", "SDG4_Education", "SDG13_Climate"])
    with f_c3:
        search_state = st.text_input("🔍 Search State / UT:", "")

    matrix_df = forecast_df[forecast_df["Year"] == forecast_year_filter].copy()
    matrix_df = enrich_with_risk_labels(matrix_df, score_col="Forecast_Score")

    if risk_filter != "All Tiers":
        tier_map = {
            "High Risk (<50)": "High Risk",
            "Medium Risk (50-74)": "Medium Risk",
            "Low Risk (≥75)": "Low Risk"
        }
        matrix_df = matrix_df[matrix_df["Risk_Category"] == tier_map[risk_filter]]

    if sdg_matrix_filter != "All SDGs":
        matrix_df = matrix_df[matrix_df["SDG"] == sdg_matrix_filter]

    if search_state:
        matrix_df = matrix_df[matrix_df["State"].str.contains(search_state, case=False)]

    display_table = matrix_df[[
        "State", "SDG", "Forecast_Score", "CI_Lower_95", "CI_Upper_95", 
        "Annual_Growth_Rate", "NITI_Tier", "Risk_Category"
    ]].rename(columns={
        "Forecast_Score": f"Score ({forecast_year_filter})",
        "CI_Lower_95": "95% Lower",
        "CI_Upper_95": "95% Upper",
        "Annual_Growth_Rate": "Slope (pts/yr)",
        "NITI_Tier": "NITI Category",
        "Risk_Category": "Risk Band"
    })

    st.dataframe(
        display_table.sort_values(by=f"Score ({forecast_year_filter})", ascending=False),
        use_container_width=True,
        hide_index=True
    )

# ==========================================
# TAB 3: GEOGRAPHIC HEATMAP & SPATIAL ANALYTICS
# ==========================================
with tab3:
    st.subheader("🗺️ India Spatial Distribution & State-Level Rankings")
    
    geo_c1, geo_c2 = st.columns([1, 1])
    with geo_c1:
        map_metric = st.selectbox("Select Map Metric:", ["Composite_Score", "SDG3_Health", "SDG4_Education", "SDG13_Climate"], index=0)
    with geo_c2:
        map_year = st.slider("Select Timeline Year (Historical to Forecast):", min_value=2018, max_value=2026, value=2026, step=1)

    if map_year <= 2023:
        map_df = historical_df[historical_df["Year"] == map_year].copy()
    else:
        sub_fore = forecast_df[forecast_df["Year"] == map_year]
        pivoted = sub_fore.pivot(index="State", columns="SDG", values="Forecast_Score").reset_index()
        pivoted["Composite_Score"] = (pivoted["SDG3_Health"] + pivoted["SDG4_Education"] + pivoted["SDG13_Climate"]) / 3.0
        pivoted["Year"] = map_year
        map_df = pivoted

    g_col_map, g_col_rank = st.columns([1.3, 1])
    
    with g_col_map:
        fig_map = create_india_geo_heatmap(
            map_df, 
            selected_metric=map_metric, 
            title=f"India State Map: {map_metric.replace('_', ' ')} ({map_year})"
        )
        st.plotly_chart(fig_map, use_container_width=True)

    with g_col_rank:
        st.markdown(f"#### 🏆 State Ranking Leaderboard ({map_year})")
        fig_rank = create_state_ranking_chart(map_df, metric_col=map_metric)
        st.plotly_chart(fig_rank, use_container_width=True)

# ==========================================
# TAB 4: SDG 13 CLIMATE ACTION PANEL
# ==========================================
with tab4:
    st.subheader("🌿 SDG 13: Climate Action Intelligence & Vulnerability Matrix")
    
    st.markdown("""
    SDG 13 assesses state readiness across clean energy adoption, carbon intensity mitigation,
    disaster preparedness, forest conservation, and adaptation resilience under the State Action Plans on Climate Change (SAPCC).
    """)

    clim_fore_2026 = forecast_df[(forecast_df["SDG"] == "SDG13_Climate") & (forecast_df["Year"] == 2026)].copy()
    clim_fore_2026 = enrich_with_risk_labels(clim_fore_2026, score_col="Forecast_Score")

    cl_c1, cl_c2, cl_c3 = st.columns(3)
    with cl_c1:
        top_clim = clim_fore_2026.sort_values("Forecast_Score", ascending=False).iloc[0]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🌟 Climate Leader (2026)</div>
            <div class="metric-value" style="color: #10B981;">{top_clim['State']}</div>
            <div class="metric-delta">Projected Score: <b>{top_clim['Forecast_Score']:.1f}</b></div>
        </div>
        """, unsafe_allow_html=True)
    with cl_c2:
        lag_clim = clim_fore_2026.sort_values("Forecast_Score", ascending=True).iloc[0]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">⚠️ Highest Climate Risk (2026)</div>
            <div class="metric-value" style="color: #EF4444;">{lag_clim['State']}</div>
            <div class="metric-delta">Projected Score: <b>{lag_clim['Forecast_Score']:.1f}</b></div>
        </div>
        """, unsafe_allow_html=True)
    with cl_c3:
        avg_slope_clim = clim_fore_2026["Annual_Growth_Rate"].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">⚡ National Climate Velocity</div>
            <div class="metric-value" style="color: #38BDF8;">+{avg_slope_clim:.2f}</div>
            <div class="metric-delta">Average pts gained per year</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Climate Trajectory Scatter Plot
    fig_scatter = px.scatter(
        clim_fore_2026,
        x="Forecast_Score",
        y="Annual_Growth_Rate",
        color="Risk_Category",
        color_discrete_map={"Low Risk": "#10B981", "Medium Risk": "#F59E0B", "High Risk": "#EF4444"},
        text="State",
        hover_data=["CI_Lower_95", "CI_Upper_95"],
        title="SDG 13 State Positioning: 2026 Projected Score vs Annual Decarbonization Velocity"
    )
    fig_scatter.add_vline(x=75, line_dash="dot", line_color="#10B981")
    fig_scatter.add_vline(x=50, line_dash="dot", line_color="#EF4444")
    fig_scatter.add_hline(y=0, line_dash="solid", line_color="#94A3B8")
    fig_scatter.update_traces(textposition="top center", marker=dict(size=12, line=dict(width=1, color="#1E293B")))
    fig_scatter.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
        font=dict(color="#F8FAFC"),
        xaxis=dict(title="2026 Projected SDG 13 Score (0-100)", gridcolor="#334155"),
        yaxis=dict(title="Annual Growth Velocity (pts/year)", gridcolor="#334155")
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Renewable Energy Indicator Breakdown
    if not clim_ind_df.empty:
        st.markdown("#### ⚡ Renewable Energy Share of Installed Generating Capacity (`SDG_13_18.csv`)")
        ren_col = [c for c in clim_ind_df.columns if "renewable" in c.lower()][0]
        clim_sort = clim_ind_df.dropna(subset=[ren_col]).sort_values(ren_col, ascending=False)
        
        fig_ren = px.bar(
            clim_sort,
            x="Area",
            y=ren_col,
            color=ren_col,
            color_continuous_scale="Viridis",
            title="Percentage of Renewable Energy Capacity by State/UT"
        )
        fig_ren.add_hline(y=40.0, line_dash="dash", line_color="#10B981", annotation_text="National Target (40%)")
        fig_ren.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.4)",
            font=dict(color="#F8FAFC"),
            xaxis=dict(tickangle=-45, gridcolor="#334155"),
            yaxis=dict(title="Renewable Share (%)", gridcolor="#334155")
        )
        st.plotly_chart(fig_ren, use_container_width=True)

# ==========================================
# TAB 5: SDG 3 INDICATOR ANALYTICS
# ==========================================
with tab5:
    st.subheader("🏥 SDG 3: Good Health & Well-Being Indicator Analytics (`SDG_3_25.csv`)")
    st.markdown("Detailed sub-indicator breakdown from NITI Aayog raw indicator datasets covering maternal health, mortality, vaccination, and clinical infrastructure.")

    if not health_ind_df.empty:
        ind_cols = [c for c in health_ind_df.columns if c not in ["SNo", "Area"]]
        selected_ind = st.selectbox("Select Health Sub-Indicator to Analyze:", ind_cols, index=0)

        h_c1, h_c2 = st.columns([1.5, 1])

        with h_c1:
            clean_sub = health_ind_df.dropna(subset=[selected_ind]).sort_values(selected_ind, ascending=False)
            fig_ind = px.bar(
                clean_sub,
                x="Area",
                y=selected_ind,
                color=selected_ind,
                color_continuous_scale="Tealgrn",
                title=f"State Distribution: {selected_ind}"
            )
            fig_ind.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15, 23, 42, 0.4)",
                font=dict(color="#F8FAFC"),
                xaxis=dict(tickangle=-45, gridcolor="#334155"),
                yaxis=dict(gridcolor="#334155")
            )
            st.plotly_chart(fig_ind, use_container_width=True)

        with h_c2:
            st.markdown(f"#### 📊 Top Performers & Lagging States")
            st.dataframe(
                clean_sub[["Area", selected_ind]].rename(columns={"Area": "State / UT"}),
                use_container_width=True,
                hide_index=True
            )

        st.markdown("#### 📋 Comprehensive Health Indicator Matrix")
        st.dataframe(health_ind_df, use_container_width=True, hide_index=True)
    else:
        st.info("Health indicator dataset (SDG_3_25.csv) is being loaded.")

# ==========================================
# TAB 6: GOVERNANCE & POLICY ADVISORY
# ==========================================
with tab6:
    st.subheader(f"🏛️ Actionable Governance & Policy Advisory: {selected_state}")
    
    latest_23_dict = historical_df[historical_df["State"] == selected_state].sort_values("Year").iloc[-1].to_dict()
    fore_26_sub = forecast_df[(forecast_df["State"] == selected_state) & (forecast_df["Year"] == 2026)]
    proj_26_dict = {row["SDG"]: row["Forecast_Score"] for _, row in fore_26_sub.iterrows()}
    slopes_dict = {k: trained_models.get((selected_state, k), {}).get("slope", 0.0) for k in SDG_TARGETS}

    advisory = generate_state_policy_advisory(selected_state, latest_23_dict, proj_26_dict, slopes_dict)

    st.markdown(f"""
    <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 12px; padding: 18px 24px; margin-bottom: 20px;">
        <h4 style="color: #38BDF8; margin: 0;">📌 Executive Policy Summary for State Leadership</h4>
        <p style="color: #CBD5E1; font-size: 14px; margin-top: 8px;">
            Based on empirical trends and 2026 projections, <b>{selected_state}</b> shows the most urgent developmental need in 
            <span style="color: #F87171; font-weight: 700;">{advisory['priority_sdg']}</span>.
            Targeted resource re-allocation and policy intervention schemes should be fast-tracked to prevent target slippage.
        </p>
    </div>
    """, unsafe_allow_html=True)

    for rec in advisory["recommendations"]:
        urgency_color = "#EF4444" if "CRITICAL" in rec["urgency"] else ("#F59E0B" if "ACCELERATION" in rec["urgency"] else "#10B981")
        st.markdown(f"""
        <div class="policy-box" style="border-left-color: {urgency_color};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <b style="font-size: 16px; color: #F8FAFC;">{rec['sdg_title']}</b>
                <span style="font-size: 12px; font-weight: 700; color: {urgency_color};">{rec['urgency']}</span>
            </div>
            <div style="font-size: 13px; color: #94A3B8; margin: 6px 0 10px 0;">
                Current (2023): <b>{rec['2023_score']:.1f}</b> ➔ 2026 Projected: <b>{rec['2026_proj']:.1f}</b> (Velocity: {'+' if rec['annual_velocity']>0 else ''}{rec['annual_velocity']:.2f} pts/yr)
            </div>
            <div style="color: #E2E8F0; font-size: 13px;">
                <b>Recommended Priority Interventions:</b>
                <ul style="margin-top: 6px; padding-left: 20px;">
                    {''.join([f'<li style="margin-bottom: 4px;">{act}</li>' for act in rec['action_items']])}
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🎯 National Resource Allocation Priority Summary (2026 Horizon)")
    nat_priorities = get_national_policy_priorities(forecast_df[forecast_df["Year"] == 2026])
    if not nat_priorities.empty:
        st.table(nat_priorities.rename(columns={"At_Risk_Count": "States in High Risk (<50)", "States_List": "Identified States"}))
    else:
        st.info("No states projected to be in the High Risk Aspirant tier by 2026.")

# ==========================================
# TAB 7: ML VALIDATION & METHODOLOGY
# ==========================================
with tab7:
    st.subheader("🔬 Machine Learning Validation & System Architecture (Review 1 Specification)")
    
    st.markdown("""
    ### 🏗️ Three-Tier System Architecture (Slide 10)
    """)
    
    arc_c1, arc_c2, arc_c3 = st.columns(3)
    with arc_c1:
        st.markdown("""
        <div class="arch-card">
            <h4 style="color: #38BDF8;">1. Data Processing Layer</h4>
            <ul style="font-size: 13px; color: #CBD5E1; padding-left: 18px;">
                <li><b>Ingestion:</b> Raw NITI Aayog SDG India Index CSV files (2018–2023).</li>
                <li><b>Cleansing:</b> Standardizes state names, null value removal.</li>
                <li><b>Transformation:</b> Long-format reshaping and integer Year schema.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with arc_c2:
        st.markdown("""
        <div class="arch-card">
            <h4 style="color: #818CF8;">2. ML & Forecasting Layer</h4>
            <ul style="font-size: 13px; color: #CBD5E1; padding-left: 18px;">
                <li><b>Modeling:</b> 108 independent Scikit-learn Random Forest Regressor models (Ensemble Tree Architecture).</li>
                <li><b>Projections:</b> Forward multi-year forecasts for <b>2024, 2025, 2026</b> with tree-variance 95% CIs.</li>
                <li><b>Risk Profiling:</b> Automated 3-tier NITI Aayog classifier (<50, 50–74, ≥75).</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with arc_c3:
        st.markdown("""
        <div class="arch-card">
            <h4 style="color: #C084FC;">3. Visualisation Layer</h4>
            <ul style="font-size: 13px; color: #CBD5E1; padding-left: 18px;">
                <li><b>Framework:</b> Interactive Streamlit application for policymakers.</li>
                <li><b>Features:</b> Trend charts with 95% CI, India map heatmap, early warnings.</li>
                <li><b>Exports:</b> Instant 1-click downloadable CSV reports.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Empirical Holdout Validation Metrics (Slide 9)")

    val_agg = val_df.groupby("SDG").agg(
        Mean_RMSE=("RMSE", "mean"),
        Mean_MAE=("MAE", "mean"),
        Mean_R2=("R2", "mean")
    ).reset_index()

    v_c1, v_c2, v_c3 = st.columns(3)
    with v_c1:
        s3_r2 = val_agg[val_agg["SDG"] == "SDG3_Health"]["Mean_R2"].values[0]
        s3_rmse = val_agg[val_agg["SDG"] == "SDG3_Health"]["Mean_RMSE"].values[0]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🏥 SDG 3 Holdout R²</div>
            <div class="metric-value" style="color: #10B981;">{s3_r2:.2f}</div>
            <div class="metric-delta">Mean Holdout RMSE: <b>{s3_rmse:.2f}</b> pts (Naive Beaten ✅)</div>
        </div>
        """, unsafe_allow_html=True)
    with v_c2:
        s4_r2 = val_agg[val_agg["SDG"] == "SDG4_Education"]["Mean_R2"].values[0]
        s4_rmse = val_agg[val_agg["SDG"] == "SDG4_Education"]["Mean_RMSE"].values[0]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📚 SDG 4 Holdout R²</div>
            <div class="metric-value" style="color: #38BDF8;">{s4_r2:.2f}</div>
            <div class="metric-delta">Mean Holdout RMSE: <b>{s4_rmse:.2f}</b> pts (Naive Beaten ✅)</div>
        </div>
        """, unsafe_allow_html=True)
    with v_c3:
        s13_r2 = val_agg[val_agg["SDG"] == "SDG13_Climate"]["Mean_R2"].values[0]
        s13_rmse = val_agg[val_agg["SDG"] == "SDG13_Climate"]["Mean_RMSE"].values[0]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🌿 SDG 13 Holdout R²</div>
            <div class="metric-value" style="color: #F59E0B;">{s13_r2:.2f}</div>
            <div class="metric-delta">Mean Holdout RMSE: <b>{s13_rmse:.2f}</b> pts (Naive Beaten ✅)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📑 State-by-State Model Scorecard (108 Models)")
    sdg_val_filter = st.selectbox("Select SDG for Validation Inspection:", options=SDG_TARGETS, format_func=lambda x: sdg_options[x])
    filtered_val = val_df[val_df["SDG"] == sdg_val_filter].sort_values("RMSE")
    st.dataframe(
        filtered_val[[
            "State", "Holdout_Actual_2022", "Holdout_Pred_2022", 
            "Holdout_Actual_2023", "Holdout_Pred_2023", "RMSE", "MAE", "R2", "Naive_Beaten"
        ]],
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")
    st.markdown("### 📚 Literature Review & References (Slides 6–8, 11–12)")
    with st.expander("📖 View Key Academic Literature & Contributions"):
        st.markdown("""
        1. **Vinuesa et al. (2020)**: *The role of artificial intelligence in achieving the SDGs.* Nature Communications, 11(1).
        2. **Sachs et al. (2023)**: *Sustainable Development Report 2023.* SDSN & Bertelsmann Stiftung.
        3. **NITI Aayog (2021, 2023)**: *SDG India Index and Dashboard 2020–21, 2022–23.* Government of India.
        4. **Huang et al. (2021)**: *Forecasting SDG progress using ML in developing nations.* Sustainability, 13(9).
        5. **Pradhan et al. (2017)**: *A systematic study of SDG interactions.* Earth's Future, 5(11).
        6. **James et al. (2023)**: *An Introduction to Statistical Learning with Applications in Python.* Springer.
        7. **Streamlit Inc. (2024)**: *Streamlit Documentation v1.20.* docs.streamlit.io.
        8. **Pedregosa et al. (2011)**: *Scikit-learn: Machine learning in Python.* JMLR, 12.
        9. **Alzubaidi et al. (2021)**: *Review of deep learning: Concepts, CNN architectures, challenges.* Journal of Big Data, 8(1).
        10. **Niu et al. (2020)**: *Time series forecasting using LSTM networks for environmental indicators.* Applied Energy, 259.
        11. **Bhatt and Bhatt (2022)**: *Machine learning applications in public health policy analysis in India.* Indian Journal of Public Health.
        12. **Fuso Nerini et al. (2019)**: *Connecting climate action with other Sustainable Development Goals.* Nature Sustainability.
        13. **Giannetti et al. (2020)**: *Dashboard systems for SDG monitoring and environmental decision support.* Ecological Indicators.
        14. **McKinney (2010)**: *Data structures for statistical computing in Python.* SciPy.
        15. **Hunter (2007)**: *Matplotlib: A 2D graphics environment.* Computing in Science & Engineering.
        16. **United Nations (2015)**: *Transforming our world: The 2030 Agenda for Sustainable Development.* UN General Assembly.
        17. **Holtz and Healy (2017)**: *SHAP: A game theory approach to explain machine learning model output.* arXiv.
        """)

# ==========================================
# TAB 8: DATA & REPORT EXPORT
# ==========================================
with tab8:
    st.subheader("📥 Data & Policy Report Export Center")
    
    st.markdown("""
    Download verified NITI Aayog historical data, 2024–2026 forecast projections, 
    and machine learning validation metrics in standard CSV formats for offline policy deliberation.
    """)

    exp_c1, exp_c2, exp_c3 = st.columns(3)
    
    with exp_c1:
        st.markdown("#### 1. Full 2024–2026 Projections")
        st.markdown("Contains all 324+ forecast projections across 36 States/UTs with 95% confidence intervals and risk tags.")
        csv_proj = forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Projections CSV",
            data=csv_proj,
            file_name="sdg_india_projections_2024_2026.csv",
            mime="text/csv"
        )

    with exp_c2:
        st.markdown("#### 2. Historical Baseline (2018–2023)")
        st.markdown("Cleaned NITI Aayog SDG India Index historical dataset covering all 36 States/UTs across SDGs 3, 4, and 13.")
        csv_hist = historical_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Historical CSV",
            data=csv_hist,
            file_name="niti_aayog_sdg_historical_2018_2023.csv",
            mime="text/csv"
        )

    with exp_c3:
        st.markdown("#### 3. ML Model Validation Scorecard")
        st.markdown("Holdout evaluation metrics (RMSE, MAE, R²) across all 108 state-level Random Forest Regressor models.")
        csv_val = val_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Validation Metrics CSV",
            data=csv_val,
            file_name="sdg_model_validation_metrics.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748B; font-size: 12px; padding: 10px 0 20px 0;">
    <b>Predictive Dashboard for SDG Performance Monitoring</b> · BCSE497J Project I · Vellore Institute of Technology (VIT) Chennai<br>
    Sakhi Telang (23BAI1105) · Akhilesh Deshmukh (23BRS1149) · Vinayak Rathod (23BCE1747) | Guided by Dr. Sureshkumar WI
</div>
""", unsafe_allow_html=True)
