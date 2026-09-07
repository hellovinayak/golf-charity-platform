"""
geo_visualizer.py
India State-level Choropleth and Spatial Analytics for SDG 3, 4, 13 & Composite Scores.
Uses Plotly for responsive, high-aesthetic choropleth and ranking visualizations.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Coordinates for all 36 Indian States and UTs (for geo scatter & spatial visualization)
STATE_COORDINATES = {
    "Andhra Pradesh": {"lat": 15.9129, "lon": 79.7400, "region": "South"},
    "Arunachal Pradesh": {"lat": 28.2180, "lon": 94.7278, "region": "North-East"},
    "Assam": {"lat": 26.2006, "lon": 92.9376, "region": "North-East"},
    "Bihar": {"lat": 25.0961, "lon": 85.3131, "region": "East"},
    "Chhattisgarh": {"lat": 21.2787, "lon": 81.8661, "region": "Central"},
    "Goa": {"lat": 15.2993, "lon": 74.1240, "region": "West"},
    "Gujarat": {"lat": 22.2587, "lon": 71.1924, "region": "West"},
    "Haryana": {"lat": 29.0588, "lon": 76.0856, "region": "North"},
    "Himachal Pradesh": {"lat": 31.1048, "lon": 77.1734, "region": "North"},
    "Jharkhand": {"lat": 23.6102, "lon": 85.2799, "region": "East"},
    "Karnataka": {"lat": 15.3173, "lon": 75.7139, "region": "South"},
    "Kerala": {"lat": 10.8505, "lon": 76.2711, "region": "South"},
    "Madhya Pradesh": {"lat": 22.9734, "lon": 78.6569, "region": "Central"},
    "Maharashtra": {"lat": 19.7515, "lon": 75.7139, "region": "West"},
    "Manipur": {"lat": 24.6637, "lon": 93.9063, "region": "North-East"},
    "Meghalaya": {"lat": 25.4670, "lon": 91.3662, "region": "North-East"},
    "Mizoram": {"lat": 23.1645, "lon": 92.9376, "region": "North-East"},
    "Nagaland": {"lat": 26.1584, "lon": 94.5624, "region": "North-East"},
    "Odisha": {"lat": 20.9517, "lon": 85.0985, "region": "East"},
    "Punjab": {"lat": 31.1471, "lon": 75.3412, "region": "North"},
    "Rajasthan": {"lat": 27.0238, "lon": 74.2179, "region": "North"},
    "Sikkim": {"lat": 27.5330, "lon": 88.5122, "region": "North-East"},
    "Tamil Nadu": {"lat": 11.1271, "lon": 78.6569, "region": "South"},
    "Telangana": {"lat": 18.1124, "lon": 79.0193, "region": "South"},
    "Tripura": {"lat": 23.9408, "lon": 91.9882, "region": "North-East"},
    "Uttar Pradesh": {"lat": 26.8467, "lon": 80.9462, "region": "North"},
    "Uttarakhand": {"lat": 30.0668, "lon": 79.0193, "region": "North"},
    "West Bengal": {"lat": 22.9868, "lon": 87.8550, "region": "East"},
    "Andaman and Nicobar Islands": {"lat": 11.7401, "lon": 92.6586, "region": "UT"},
    "Chandigarh": {"lat": 30.7333, "lon": 76.7794, "region": "UT"},
    "Dadra and Nagar Haveli and Daman and Diu": {"lat": 20.4283, "lon": 72.8397, "region": "UT"},
    "Delhi": {"lat": 28.7041, "lon": 77.1025, "region": "UT"},
    "Jammu and Kashmir": {"lat": 33.7782, "lon": 76.5762, "region": "UT"},
    "Ladakh": {"lat": 34.1526, "lon": 77.5771, "region": "UT"},
    "Lakshadweep": {"lat": 10.5667, "lon": 72.6417, "region": "UT"},
    "Puducherry": {"lat": 11.9416, "lon": 79.8083, "region": "UT"}
}

def create_india_geo_heatmap(df_year, selected_metric="Composite_Score", title=None):
    """
    Creates an interactive geographic bubble/scatter map across all 36 Indian states & UTs.
    """
    map_data = []
    for _, row in df_year.iterrows():
        state = row["State"]
        coords = STATE_COORDINATES.get(state, {"lat": 20.5937, "lon": 78.9629, "region": "Other"})
        val = row.get(selected_metric, row.get("Score", 50.0))
        
        # Risk classification
        if val >= 75:
            tier = "Front Runner (Low Risk)"
            color = "#10B981"
        elif val >= 50:
            tier = "Performer (Medium Risk)"
            color = "#F59E0B"
        else:
            tier = "Aspirant (High Risk)"
            color = "#EF4444"

        map_data.append({
            "State": state,
            "Latitude": coords["lat"],
            "Longitude": coords["lon"],
            "Region": coords["region"],
            "Score": val,
            "Tier": tier,
            "Color": color
        })

    geo_df = pd.DataFrame(map_data)
    
    fig = px.scatter_geo(
        geo_df,
        lat="Latitude",
        lon="Longitude",
        hover_name="State",
        size="Score",
        color="Score",
        color_continuous_scale=[(0.0, "#EF4444"), (0.5, "#F59E0B"), (0.75, "#3B82F6"), (1.0, "#10B981")],
        range_color=[30, 100],
        hover_data={"Latitude": False, "Longitude": False, "Score": ":.1f", "Region": True, "Tier": True},
        title=title or f"State-Level {selected_metric} Distribution",
    )

    fig.update_geos(
        fitbounds="locations",
        visible=True,
        showcountries=True,
        countrycolor="#334155",
        showcoastlines=True,
        coastlinecolor="#334155",
        showland=True,
        landcolor="#0F172A",
        showocean=True,
        oceancolor="#020617",
        showsubunits=True,
        subunitcolor="#1E293B",
        resolution=50,
        center={"lat": 22.0, "lon": 79.0},
        projection_scale=4.5
    )

    fig.update_layout(
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F8FAFC", family="Inter, system-ui"),
        coloraxis_colorbar=dict(
            title="SDG Index (0-100)",
            thicknessmode="pixels",
            thickness=15,
            lenmode="fraction",
            len=0.75,
            bgcolor="rgba(15,23,42,0.8)",
            tickfont=dict(color="#CBD5E1")
        )
    )
    return fig

def create_state_ranking_chart(df_year, metric_col="Score", top_n=36):
    """Generates an interactive horizontal bar chart of state rankings."""
    sorted_df = df_year.sort_values(by=metric_col, ascending=True).tail(top_n).copy()
    
    # Assign color according to NITI Aayog threshold
    colors = []
    for s in sorted_df[metric_col]:
        if s >= 75:
            colors.append("#10B981")
        elif s >= 50:
            colors.append("#F59E0B")
        else:
            colors.append("#EF4444")
            
    fig = go.Figure(go.Bar(
        x=sorted_df[metric_col],
        y=sorted_df["State"],
        orientation="h",
        marker=dict(color=colors, line=dict(color="#1E293B", width=1)),
        text=[f"{v:.1f}" for v in sorted_df[metric_col]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Score: %{x:.1f}<extra></extra>"
    ))

    fig.update_layout(
        height=max(450, len(sorted_df) * 22),
        margin={"r": 30, "t": 20, "l": 10, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F8FAFC", family="Inter, system-ui"),
        xaxis=dict(
            range=[0, 105],
            showgrid=True,
            gridcolor="#334155",
            title="SDG Index Score (0-100)"
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(size=11)
        )
    )
    return fig

def create_trend_line_chart(timeseries_df, state_name, selected_sdgs=None):
    """
    Creates high-fidelity multi-SDG time series line chart with historical vs forecast split & confidence interval.
    """
    if selected_sdgs is None:
        selected_sdgs = ["SDG3_Health", "SDG4_Education", "SDG13_Climate"]

    sdg_colors = {
        "SDG3_Health": ("#10B981", "SDG 3: Good Health"),
        "SDG4_Education": ("#3B82F6", "SDG 4: Quality Education"),
        "SDG13_Climate": ("#F59E0B", "SDG 13: Climate Action")
    }

    fig = go.Figure()
    state_data = timeseries_df[timeseries_df["State"] == state_name]

    for sdg_key in selected_sdgs:
        if sdg_key not in sdg_colors:
            continue
        color_hex, sdg_label = sdg_colors[sdg_key]
        sub = state_data[state_data["SDG"] == sdg_key].sort_values("Year")

        hist_part = sub[sub["Type"] == "Historical"]
        fore_part = sub[sub["Type"] == "Forecast"]
        
        # Connect transition point (2023 historical to 2024 forecast)
        bridge = sub[sub["Year"].isin([2023, 2024])]

        # 1. Shaded 95% Confidence Interval for forecast
        if not fore_part.empty:
            fig.add_trace(go.Scatter(
                x=list(fore_part["Year"]) + list(fore_part["Year"][::-1]),
                y=list(fore_part["CI_Upper_95"]) + list(fore_part["CI_Lower_95"][::-1]),
                fill="toself",
                fillcolor=f"rgba({int(color_hex[1:3], 16)}, {int(color_hex[3:5], 16)}, {int(color_hex[5:7], 16)}, 0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
                name=f"{sdg_label} (95% CI)"
            ))

        # 2. Historical line (solid with markers)
        fig.add_trace(go.Scatter(
            x=hist_part["Year"],
            y=hist_part["Score"],
            mode="lines+markers",
            name=f"{sdg_label} (Historical)",
            line=dict(color=color_hex, width=3),
            marker=dict(size=7, color=color_hex),
            hovertemplate="<b>%{x} (Hist)</b>: %{y:.1f}<extra></extra>"
        ))

        # 3. Forecast line (dashed)
        if not fore_part.empty:
            # Combine 2023 end with forecast for continuous line
            fore_years = [2023] + list(fore_part["Year"])
            fore_scores = [hist_part[hist_part["Year"] == 2023]["Score"].values[0]] + list(fore_part["Score"])
            
            fig.add_trace(go.Scatter(
                x=fore_years,
                y=fore_scores,
                mode="lines+markers",
                name=f"{sdg_label} (Forecast 2024-26)",
                line=dict(color=color_hex, width=3, dash="dash"),
                marker=dict(size=8, symbol="diamond", color=color_hex),
                hovertemplate="<b>%{x} (Forecast)</b>: %{y:.1f}<extra></extra>"
            ))

    # Benchmark Thresholds (NITI Aayog)
    fig.add_hline(y=75, line_dash="dot", line_color="#10B981", opacity=0.7, annotation_text="Front Runner Threshold (≥75)", annotation_position="top right", annotation_font_color="#10B981")
    fig.add_hline(y=50, line_dash="dot", line_color="#EF4444", opacity=0.7, annotation_text="Critical Aspirant Threshold (<50)", annotation_position="bottom right", annotation_font_color="#EF4444")

    # Add vertical partition line for forecast horizon
    fig.add_vline(x=2023.5, line_dash="dash", line_color="#94A3B8", opacity=0.6, annotation_text="Forecast Horizon (2024-26)", annotation_font_color="#94A3B8")

    fig.update_layout(
        title=f"SDG Trajectory & Forward Projection: <b>{state_name}</b>",
        xaxis=dict(title="Year", tickmode="linear", tick0=2018, dtick=1, gridcolor="#334155"),
        yaxis=dict(title="SDG Index Score (0–100)", range=[20, 100], gridcolor="#334155"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
        font=dict(color="#F8FAFC", family="Inter, system-ui"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(15, 23, 42, 0.8)"),
        margin=dict(l=40, r=40, t=80, b=40)
    )
    return fig
