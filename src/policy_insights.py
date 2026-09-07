"""
policy_insights.py
Layer 3 Governance & Policy Intelligence:
Generates actionable, evidence-based policy briefs, resource allocation priorities,
and sector-specific interventions tailored for state administrators & NITI Aayog planners.
"""

import pandas as pd

POLICY_PLAYBOOKS = {
    "SDG3_Health": {
        "Critical (<50)": [
            "🚨 Mobilize National Health Mission (NHM) emergency infrastructure funds for Primary Health Centres (PHCs).",
            "💉 Launch intensified Mission Indradhanush vaccination drives in lagging aspirational districts.",
            "👩‍⚕️ Introduce rural service retention bonuses to fill critical vacancies for OB/GYN doctors and nurses."
        ],
        "Performer (50-74)": [
            "📈 Expand Ayushman Bharat digital health IDs and e-Sanjeevani telemedicine coverage to 100% of blocks.",
            "🏥 Upgrade Sub-Centres to Health and Wellness Centres (HWCs) with non-communicable disease screening.",
            "📉 Focus targeted maternal nutrition interventions via POSHAN Abhiyaan 2.0."
        ],
        "Front Runner (>=75)": [
            "🌟 Institutionalize advanced tertiary geriatric care and specialized oncology centres.",
            "🔬 Establish AI-enabled predictive epidemiological disease surveillance networks.",
            "🏆 Scale inter-state health best-practice exchange programs."
        ]
    },
    "SDG4_Education": {
        "Critical (<50)": [
            "🚨 Enforce zero-dropout retention protocols for secondary school transitions under Samagra Shiksha.",
            "📚 Deploy foundational literacy and numeracy (FLN) intensive camps under NIPUN Bharat.",
            "🏫 Urgent capital allocation for basic functional girls' sanitation and electricity in rural schools."
        ],
        "Performer (50-74)": [
            "💻 Establish smart ICT labs and DIKSHA digital classroom integrations across government schools.",
            "👩‍🏫 Targeted teacher re-skilling programs with performance-linked continuous professional development.",
            "🎓 Expand vocational education streams (PM-SHRI schools) aligned with regional industrial skills."
        ],
        "Front Runner (>=75)": [
            "🌟 Expand higher education gross enrolment ratio (GER) towards 50% target.",
            "🔬 Fund state university incubation hubs and STEM research grants.",
            "🌐 Integrate global curriculum standards and multilingual educational software."
        ]
    },
    "SDG13_Climate": {
        "Critical (<50)": [
            "🚨 Expedite State Climate Action Plan (SAPCC) revision with dedicated state disaster mitigation funds.",
            "⚡ Enforce renewable purchase obligations (RPO) and solar rooftop subsidies under PM Surya Ghar.",
            "🌳 Launch emergency Miyawaki afforestation corridors and mandatory industrial effluent monitoring."
        ],
        "Performer (50-74)": [
            "🔋 Incentivize state EV public transit adoption and distributed microgrid storage for vulnerable districts.",
            "🌾 Promote climate-resilient micro-irrigation under PM Krishi Sinchayee Yojana.",
            "🌊 Construct coastal cyclone shelters and early multi-hazard warning telemetry systems."
        ],
        "Front Runner (>=75)": [
            "🌟 Issue State Green Sovereign Bonds to fund utility-scale green hydrogen and offshore wind projects.",
            "🏙️ Mandate Net-Zero building codes (ECBC) for all new commercial and government infrastructures.",
            "♻️ Develop circular economy carbon offset trading mechanisms."
        ]
    }
}

def generate_state_policy_advisory(state_name, latest_scores, projected_scores, slopes):
    """
    Generates a personalized governance policy brief for a specific state.
    """
    sdg_keys = ["SDG3_Health", "SDG4_Education", "SDG13_Climate"]
    sdg_labels = {
        "SDG3_Health": "SDG 3: Good Health & Well-Being",
        "SDG4_Education": "SDG 4: Quality Education",
        "SDG13_Climate": "SDG 13: Climate Action"
    }

    advisory = {
        "state": state_name,
        "summary": "",
        "priority_sdg": None,
        "recommendations": []
    }

    lowest_score = 999
    priority_goal = None

    for k in sdg_keys:
        curr = latest_scores.get(k, 50.0)
        proj_2026 = projected_scores.get(k, curr)
        slope = slopes.get(k, 0.0)
        
        if proj_2026 < lowest_score:
            lowest_score = proj_2026
            priority_goal = k

        if proj_2026 < 50:
            tier_key = "Critical (<50)"
            urgency = "CRITICAL PRIORITY"
        elif proj_2026 < 75:
            tier_key = "Performer (50-74)"
            urgency = "ACCELERATION NEEDED"
        else:
            tier_key = "Front Runner (>=75)"
            urgency = "SUSTAIN & INNOVATE"

        playbook = POLICY_PLAYBOOKS.get(k, {}).get(tier_key, [])

        advisory["recommendations"].append({
            "sdg_key": k,
            "sdg_title": sdg_labels[k],
            "urgency": urgency,
            "2023_score": curr,
            "2026_proj": proj_2026,
            "annual_velocity": slope,
            "tier": tier_key,
            "action_items": playbook
        })

    advisory["priority_sdg"] = sdg_labels.get(priority_goal, "General")
    return advisory

def get_national_policy_priorities(df_fore_2026):
    """
    Identifies top national priority states requiring urgent budget allocation by 2026.
    """
    high_risk_states = df_fore_2026[df_fore_2026["Forecast_Score"] < 50].copy()
    
    summary = (
        high_risk_states.groupby("SDG")
        .agg(
            At_Risk_Count=("State", "nunique"),
            States_List=("State", lambda x: list(x.unique()))
        )
        .reset_index()
    )
    return summary
