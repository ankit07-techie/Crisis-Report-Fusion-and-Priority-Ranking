"""Crisis Report Fusion and Priority Ranking (AI-03)
Streamlit Showcase & Emergency Triage Dashboard.
Conforms strictly to Sections 9 & 10 of the PRD and reference operational design:
- Clean, realistic, professional emergency-response intelligence tool.
- Zero AI gimmicks, fake metrics, or fabricated confidence percentages.
- Live Triage: Real-time report analysis (Incident, Official Category, Operational Priority, Verifiable Evidence).
- Incident Fusion: Direct demonstration of semantic clustering into shared incidents.
- Batch Analysis: Operational batch review and structured CSV/JSONL export.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd
import streamlit as st

from src.schemas import Report, Prediction
from src.pipeline import CrisisPipeline
from src.config import DEFAULT_CONFIG

# -------------------------------------------------------------
# 1. Page Configuration
# -------------------------------------------------------------
st.set_page_config(
    page_title="Crisis Intelligence Dashboard | AI-03",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

def render_html(html_code: str) -> None:
    """Safely render HTML without Markdown indented-code-block or paragraph breakage."""
    if hasattr(st, "html"):
        st.html(html_code)
    else:
        cleaned = "\n".join(line.strip() for line in html_code.splitlines() if line.strip())
        st.markdown(cleaned, unsafe_allow_html=True)

# -------------------------------------------------------------
# 2. Official Metadata & Operational Taxonomies
# -------------------------------------------------------------
OFFICIAL_CATEGORIES = {
    "SearchAndRescue": "Request for rescue or search operations",
    "EmergingThreats": "Impending or worsening physical hazard",
    "GoodsServices": "Medical supplies, potable water, or relief aid",
    "MovePeople": "Evacuation instructions or transit guidance",
    "Location": "Geographic coordinates, routes, or road checkpoints",
    "FirstPartyObservation": "Direct eyewitness observation from the ground",
    "InformationWanted": "Inquiry or request for emergency information",
    "ServiceAvailable": "Operational shelters, medical clinics, or aid centers",
    "Volunteer": "Offers or requests for volunteer personnel",
    "MultimediaShare": "Photos, videos, or external media sharing",
    "NewSubEvent": "Emergent secondary event or escalations",
}

def get_priority_meta(score: float) -> Dict[str, str]:
    """Provide calibrated operational interpretations for priority scores."""
    if score >= 4.5:
        return {
            "label": "Critical",
            "badge_class": "badge-critical",
            "box_class": "metric-box-critical",
            "status_text": "Immediate attention required",
            "basis": "This report contains indications of immediate risk and an urgent need for emergency response.",
            "color": "#dc2626",
            "bg_tint": "#fef2f2"
        }
    elif score >= 3.5:
        return {
            "label": "High",
            "badge_class": "badge-high",
            "box_class": "metric-box-high",
            "status_text": "Urgent operational dispatch required",
            "basis": "This report describes escalating conditions requiring prioritized resource dispatch.",
            "color": "#ea580c",
            "bg_tint": "#fff7ed"
        }
    elif score >= 2.5:
        return {
            "label": "Medium",
            "badge_class": "badge-medium",
            "box_class": "metric-box-medium",
            "status_text": "Coordinated relief response",
            "basis": "This report indicates support services, supply logistics, or general assistance requirements.",
            "color": "#d97706",
            "bg_tint": "#fffbeb"
        }
    else:
        return {
            "label": "Low",
            "badge_class": "badge-low",
            "box_class": "metric-box-low",
            "status_text": "Routine situational awareness",
            "basis": "This report represents informational inquiries or non-immediate situational observations.",
            "color": "#2563eb",
            "bg_tint": "#eff6ff"
        }

# -------------------------------------------------------------
# 3. Custom Styling (Refined Emergency Operations Visuals)
# -------------------------------------------------------------
st.markdown("""
<style>
    /* Global Base */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
    }

    /* Sidebar Theme */
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        color: #f8fafc !important;
        border-right: 1px solid #1e293b;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {
        color: #cbd5e1 !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: #334155 !important;
    }

    /* Clean Card Containers */
    .op-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 20px 22px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .op-card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 6px;
    }
    .op-card-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .op-card-subtitle {
        font-size: 0.85rem;
        color: #64748b;
        margin-bottom: 14px;
        line-height: 1.4;
    }

    /* Top Application Header */
    .app-header-container {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        padding: 12px 0 18px 0;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 22px;
    }
    .app-header-title {
        font-size: 1.75rem;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.5px;
        margin: 0;
    }
    .app-header-subtitle {
        font-size: 0.95rem;
        color: #475569;
        margin-top: 4px;
    }
    .system-status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        color: #166534;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .system-status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #22c55e;
    }

    /* 4-Column Assessment Metric Boxes */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 18px;
    }
    .metric-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 12px 14px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 110px;
    }
    .metric-box-critical {
        background: #fef2f2 !important;
        border: 1px solid #fecaca !important;
    }
    .metric-box-high {
        background: #fff7ed !important;
        border: 1px solid #fed7aa !important;
    }
    .metric-box-medium {
        background: #fffbeb !important;
        border: 1px solid #fef08a !important;
    }
    .metric-box-low {
        background: #eff6ff !important;
        border: 1px solid #bfdbfe !important;
    }
    .metric-box-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .metric-box-value {
        font-size: 1.35rem;
        font-weight: 800;
        color: #0f172a;
        margin: 2px 0 4px 0;
        line-height: 1.1;
    }
    .metric-box-desc {
        font-size: 0.75rem;
        color: #64748b;
        line-height: 1.3;
    }

    /* Severity Badges */
    .badge-critical {
        display: inline-block;
        background-color: #dc2626;
        color: #ffffff;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        margin-left: 6px;
    }
    .badge-high {
        display: inline-block;
        background-color: #ea580c;
        color: #ffffff;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        margin-left: 6px;
    }
    .badge-medium {
        display: inline-block;
        background-color: #d97706;
        color: #ffffff;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        margin-left: 6px;
    }
    .badge-low {
        display: inline-block;
        background-color: #2563eb;
        color: #ffffff;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        margin-left: 6px;
    }

    /* Evidence Pills */
    .evidence-pill {
        display: inline-block;
        background-color: #f1f5f9;
        color: #1e293b;
        border: 1px solid #cbd5e1;
        border-radius: 4px;
        padding: 3px 9px;
        margin: 2px 4px 2px 0;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.82rem;
        font-weight: 600;
    }

    /* Prediction Basis Callout Box */
    .basis-box {
        background-color: #f0f9ff;
        border: 1px solid #bae6fd;
        border-radius: 6px;
        padding: 10px 14px;
        color: #0369a1;
        font-size: 0.82rem;
        line-height: 1.45;
        margin-top: 14px;
    }

    /* Button Primary */
    div.stButton > button:first-child {
        background-color: #00438c !important;
        color: #ffffff !important;
        border: 1px solid #00336e !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 8px 20px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
    }
    div.stButton > button:first-child:hover {
        background-color: #00336e !important;
        border-color: #00224d !important;
    }

    /* Empty State Box */
    .empty-state-box {
        background-color: #f8fafc;
        border: 1px dashed #cbd5e1;
        border-radius: 8px;
        padding: 36px 20px;
        text-align: center;
        color: #64748b;
    }

    /* Fusion Diagram Schematic */
    .fusion-diagram-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px 20px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.92rem;
        color: #0f172a;
        margin: 16px 0;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 4. Pipeline Resource Initialization & State Management
# -------------------------------------------------------------
@st.cache_resource
def get_pipeline() -> CrisisPipeline:
    """Initialize and cache the CrisisPipeline instance."""
    return CrisisPipeline()

pipeline = get_pipeline()

# Session State
if "history" not in st.session_state:
    st.session_state["history"] = []

if "current_prediction" not in st.session_state:
    st.session_state["current_prediction"] = None

if "current_report_obj" not in st.session_state:
    st.session_state["current_report_obj"] = None

if "report_input_text" not in st.session_state:
    st.session_state["report_input_text"] = "Several people are trapped inside a collapsed building near the railway station. Rescue teams are urgently needed."

if "report_counter" not in st.session_state:
    st.session_state["report_counter"] = 1

# -------------------------------------------------------------
# 5. Sidebar Layout & Controls
# -------------------------------------------------------------
with st.sidebar:
    # Small project identifier
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px; padding-bottom: 12px; border-bottom: 1px solid #1e293b;">
        <span style="font-size: 26px;">📡</span>
        <div>
            <div style="font-size: 19px; font-weight: 800; color: #ffffff; letter-spacing: -0.5px;">AI-03</div>
            <div style="font-size: 11px; color: #94a3b8; font-weight: 500;">Crisis Report Fusion & Priority Ranking</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Navigation Menu
    nav_selection = st.radio(
        "Application Navigation",
        options=[
            "01 — Live Triage",
            "02 — Incident Fusion",
            "03 — Batch Analysis"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Operational Subsystem Status
    st.markdown("<div style='font-size: 0.8rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; margin-bottom: 8px;'>Active Subsystems</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size: 0.82rem; color: #e2e8f0; margin-bottom: 4px;'>● <b>P1 Fusion</b>: <code>{pipeline.p1_source}</code></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size: 0.82rem; color: #e2e8f0; margin-bottom: 4px;'>● <b>P2 Classifier</b>: <code>{pipeline.p2_source}</code></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.82rem; color: #e2e8f0; margin-bottom: 12px;'>● <b>P3 Evidence</b>: <code>EvidenceRegistry</code></div>", unsafe_allow_html=True)

    # Configuration & Memory Control
    with st.expander("Operational Controls", expanded=False):
        st.caption("Runtime clustering distance cutoff.")
        sim_threshold = st.slider(
            "Clustering Cutoff (τ)",
            min_value=0.10,
            max_value=0.90,
            value=float(DEFAULT_CONFIG.similarity_threshold),
            step=0.02,
            help="Cosine similarity threshold for agglomerative incident fusion."
        )
        pipeline.config = type(DEFAULT_CONFIG)(similarity_threshold=sim_threshold)

        if st.button("Reset Indexed Incidents", use_container_width=True):
            pipeline.reset()
            st.session_state["history"] = []
            st.session_state["current_prediction"] = None
            st.session_state["current_report_obj"] = None
            st.session_state["report_counter"] = 1
            st.success("Incident memory reset.")

    st.markdown("""
    <div style="margin-top: 50px; padding-top: 16px; border-top: 1px solid #334155; color: #94a3b8; font-size: 0.82rem; line-height: 1.5;">
        <div style="font-weight: 600; color: #cbd5e1; margin-bottom: 4px;">About AI-03</div>
        Autonomous emergency incident intelligence based on official TREC-IS 2020-A events.<br><br>
        <i style="color: #64748b;">Turning Information into Action.</i>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------
# 6. Main Application Header
# -------------------------------------------------------------
st.markdown("""
<div class="app-header-container">
    <div>
        <h1 class="app-header-title">Crisis Intelligence Dashboard</h1>
        <div class="app-header-subtitle">From scattered reports to actionable, prioritized incidents.</div>
    </div>
    <div style="text-align: right;">
        <div class="system-status-badge">
            <div class="system-status-dot"></div>
            System Ready
        </div>
        <div style="font-size: 0.78rem; color: #64748b; margin-top: 4px;">
            TREC-IS 2020-A &nbsp;|&nbsp; Events 35–49 &nbsp;|&nbsp; <i>For Safer Communities</i>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 7. SCREEN 01: Live Triage (Primary Hackathon Demo Screen)
# -------------------------------------------------------------
if nav_selection == "01 — Live Triage":
    col_left, col_right = st.columns([1.05, 0.95], gap="large")

    # LEFT COLUMN: Incoming Report
    with col_left:
        st.markdown("""
        <div class="op-card-header">
            <h2 class="op-card-title">📄 Incoming Report</h2>
        </div>
        <div class="op-card-subtitle">
            Assess an incoming crisis report for incident, information type and operational priority.
        </div>
        """, unsafe_allow_html=True)

        report_text = st.text_area(
            label="Crisis Report Content",
            value=st.session_state["report_input_text"],
            height=140,
            placeholder="Enter a crisis report (e.g., from social media, helpline, news or field reports)...",
            label_visibility="collapsed"
        )
        st.session_state["report_input_text"] = report_text

        # Character Counter Display
        char_count = len(report_text)
        st.markdown(f"<div style='text-align: right; font-size: 0.75rem; color: #94a3b8; margin-top: -8px; margin-bottom: 12px;'>{char_count}/500</div>", unsafe_allow_html=True)

        # Quick Selectable Example Input Chips
        st.markdown("<div style='font-size: 0.78rem; font-weight: 600; color: #64748b; margin-bottom: 8px;'>Example reports:</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Trapped in collapsed building", use_container_width=True):
                st.session_state["report_input_text"] = "Several people are trapped inside a collapsed building near the railway station. Rescue teams are urgently needed."
                st.rerun()
            if st.button("Roads blocked due to flooding", use_container_width=True):
                st.session_state["report_input_text"] = "Heavy rainfall has caused severe flooding across several neighborhoods and main roads are becoming completely inaccessible."
                st.rerun()
        with c2:
            if st.button("Medical team available at camp", use_container_width=True):
                st.session_state["report_input_text"] = "Medical team and emergency trauma supplies available at central relief camp for affected residents."
                st.rerun()
            if st.button("Need information about shelter", use_container_width=True):
                st.session_state["report_input_text"] = "Residents are asking where they can find safe evacuation routes and open emergency shelters in District 4."
                st.rerun()

        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

        # Primary Submission Button
        if st.button("➔ Assess Report", type="primary", use_container_width=True):
            if not report_text.strip():
                st.warning("Please provide report text before assessing.")
            else:
                rep_id = f"CRISIS_{st.session_state['report_counter']:03d}"
                st.session_state["report_counter"] += 1
                report_obj = Report(id=rep_id, text=report_text.strip())

                try:
                    prediction = pipeline.process_report(report_obj)
                    st.session_state["current_prediction"] = prediction
                    st.session_state["current_report_obj"] = report_obj

                    # Record in session history
                    now_str = datetime.now().strftime("%I:%M %p")
                    st.session_state["history"].insert(0, {
                        "Time": now_str,
                        "ID": rep_id,
                        "Report (preview)": (report_text[:65] + "...") if len(report_text) > 65 else report_text,
                        "Incident": prediction.predicted_cluster_id,
                        "Category": prediction.category,
                        "Priority": f"{prediction.priority_score:.1f}",
                        "Evidence Count": len(prediction.evidence_ids),
                        "evidence_ids": prediction.evidence_ids
                    })
                    st.rerun()
                except Exception as ex:
                    st.error("Unable to process report. Please check the input and try again.")
                    st.caption(f"Error diagnostics: {ex}")

    # RIGHT COLUMN: Operational Assessment
    with col_right:
        pred: Optional[Prediction] = st.session_state.get("current_prediction")
        curr_report: Optional[Report] = st.session_state.get("current_report_obj")

        now_display = datetime.now().strftime("%d %b %Y, %I:%M %p")

        render_html(f"""
        <div class="op-card-header">
            <h2 class="op-card-title">🛡️ Operational Assessment</h2>
            <div style="font-size: 0.78rem; color: #64748b; font-weight: 500;">{now_display}</div>
        </div>
        <div class="op-card-subtitle">
            Generated using incident fusion (P1), information classification (P2) and priority ranking (P2).
        </div>
        """)

        if pred is None:
            render_html("""
            <div class="empty-state-box">
                <div style="font-size: 2rem; margin-bottom: 8px;">🛡️</div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #334155; margin-bottom: 4px;">No assessment yet</div>
                <div style="font-size: 0.85rem; color: #64748b;">
                    Enter an incoming report to generate an operational assessment.
                </div>
            </div>
            """)
        else:
            pri_meta = get_priority_meta(pred.priority_score)
            cat_desc = OFFICIAL_CATEGORIES.get(pred.category, "Humanitarian crisis information")
            ev_count = len(pred.evidence_ids)

            # 4-Box Metric Grid
            render_html(f"""
            <div class="metric-grid">
                <div class="metric-box">
                    <div>
                        <div class="metric-box-label">Incident</div>
                        <div class="metric-box-value">
                            <span style="font-size: 1.05rem; font-weight: 800; color: #0f172a;">{pred.predicted_cluster_id}</span>
                        </div>
                    </div>
                    <div class="metric-box-desc">Grouped with similar reports</div>
                </div>
                <div class="metric-box">
                    <div>
                        <div class="metric-box-label">Category</div>
                        <div class="metric-box-value" style="font-size: 1.05rem; color: #1d4ed8;">
                            {pred.category}
                        </div>
                    </div>
                    <div class="metric-box-desc">{cat_desc}</div>
                </div>
                <div class="metric-box {pri_meta['box_class']}">
                    <div>
                        <div class="metric-box-label">Priority</div>
                        <div class="metric-box-value" style="color: {pri_meta['color']};">
                            {pred.priority_score:.1f} <span style="font-size: 0.85rem; color: #64748b; font-weight: 500;">/ 5.0</span>
                            <span class="{pri_meta['badge_class']}">{pri_meta['label']}</span>
                        </div>
                    </div>
                    <div class="metric-box-desc" style="color: {pri_meta['color']}; font-weight: 600;">
                        {pri_meta['status_text']}
                    </div>
                </div>
                <div class="metric-box">
                    <div>
                        <div class="metric-box-label">Reports</div>
                        <div class="metric-box-value">{ev_count}</div>
                    </div>
                    <div class="metric-box-desc">(including this one)</div>
                </div>
            </div>
            """)

            # Supporting Evidence (Actual Dynamic IDs)
            render_html("""
            <div style="font-size: 0.88rem; font-weight: 700; color: #0f172a; margin-bottom: 2px;">
                🗄️ Supporting Evidence
            </div>
            <div style="font-size: 0.80rem; color: #64748b; margin-bottom: 8px;">
                Original reports associated with this incident:
            </div>
            """)

            if pred.evidence_ids:
                pills_html = "".join([f'<span class="evidence-pill">{eid}</span>' for eid in pred.evidence_ids])
                render_html(f"<div style='margin-bottom: 12px;'>{pills_html}</div>")
            else:
                render_html("<div style='font-size: 0.82rem; color: #64748b;'>No corroborating report IDs available.</div>")

            # Prediction Basis / Explainability
            render_html(f"""
            <div style="font-size: 0.88rem; font-weight: 700; color: #0f172a; margin-top: 14px; margin-bottom: 2px;">
                💡 Why this was prioritized
            </div>
            <div class="basis-box">
                <div>Priority is determined from report content and operational urgency signals.</div>
                <div style="margin-top: 4px; font-weight: 500;">{pri_meta['basis']}</div>
            </div>
            """)

            # Related Reports In Incident Expander
            related_reports = pipeline.get_related_reports(pred.predicted_cluster_id)
            if len(related_reports) > 1:
                with st.expander(f"Inspect All {len(related_reports)} Reports Linked to {pred.predicted_cluster_id}"):
                    for r in related_reports:
                        st.markdown(f"- **`{r.id}`**: {r.text}")

    # ---------------------------------------------------------
    # Recent Assessments Section (Full Width, Session-Scoped)
    # ---------------------------------------------------------
    st.markdown("---")
    st.markdown("""
    <div class="op-card-header">
        <div>
            <h3 class="op-card-title">🕒 Recent Assessments</h3>
            <div class="op-card-subtitle" style="margin-bottom: 4px;">Your latest processed reports in this session.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state["history"]:
        st.markdown("""
        <div class="empty-state-box" style="padding: 24px 20px;">
            <div style="font-size: 0.95rem; font-weight: 600; color: #475569;">No assessments yet</div>
            <div style="font-size: 0.82rem; color: #64748b; margin-top: 4px;">
                Enter an incoming report above to begin generating prioritized operational assessments.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        df_hist = pd.DataFrame(st.session_state["history"])
        display_cols = ["Time", "ID", "Report (preview)", "Incident", "Category", "Priority", "Evidence Count"]
        st.dataframe(df_hist[display_cols], use_container_width=True, hide_index=True)

        col_clr, _ = st.columns([1, 4])
        with col_clr:
            if st.button("Clear Session History", use_container_width=True):
                st.session_state["history"] = []
                st.rerun()

# -------------------------------------------------------------
# 8. SCREEN 02: Incident Fusion (PRD Section 10 Goal)
# -------------------------------------------------------------
elif nav_selection == "02 — Incident Fusion":
    st.markdown("""
    <div class="op-card-header">
        <div>
            <h2 class="op-card-title">🔗 Incident Fusion</h2>
            <div class="op-card-subtitle">
                Different reports can describe the same underlying incident. The system clusters them together semantically and preserves all corroborating evidence IDs.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    fusion_scenarios = {
        "Railway Bridge Collapse (Differently Worded)": (
            "Several people are trapped inside a collapsed railway bridge structure near the station. Emergency teams urgently needed.",
            "Water currents washed out the railway overpass completely. Multiple vehicles and civilians stranded in the wreckage."
        ),
        "Chemical Gas Leak Hazard": (
            "Toxic chemical odor and heavy smoke billowing from the chemical plant on Sector 4. Residents coughing and choking.",
            "Dangerous gas leak and hazardous fumes detected near the industrial chemical facility, evacuation ordered."
        ),
        "Divergent Emergency Dispatches (Should NOT Fuse)": (
            "Severe floodwaters washing away cars on River Road, families trapped on upper floors.",
            "Need immediate delivery of dry rations and potable drinking water at the central high school gym shelter."
        )
    }

    selected_scenario = st.selectbox(
        "Demonstration Scenario",
        options=list(fusion_scenarios.keys()),
        help="Select pre-configured dispatches to test semantic fusion."
    )
    preset_a, preset_b = fusion_scenarios[selected_scenario]

    c_a, c_b = st.columns(2, gap="medium")
    with c_a:
        st.markdown("**Report A**")
        id_a = st.text_input("Report ID A", value="DISPATCH_A")
        text_a = st.text_area("Content A", value=preset_a, height=110)
    with c_b:
        st.markdown("**Report B**")
        id_b = st.text_input("Report ID B", value="DISPATCH_B")
        text_b = st.text_area("Content B", value=preset_b, height=110)

    if st.button("➔ Find Related Incident", type="primary"):
        if not text_a.strip() or not text_b.strip():
            st.warning("Please provide valid report text for both dispatches.")
        else:
            rep_a = Report(id=id_a.strip(), text=text_a.strip())
            rep_b = Report(id=id_b.strip(), text=text_b.strip())

            try:
                preds = pipeline.process_batch([rep_a, rep_b])
                pred_a = next(p for p in preds if p.item_id == rep_a.id)
                pred_b = next(p for p in preds if p.item_id == rep_b.id)

                is_fused = (pred_a.predicted_cluster_id == pred_b.predicted_cluster_id)

                st.markdown("---")
                st.markdown("### Fusion Resolution")

                if is_fused:
                    render_html(f"""
                    <div class="fusion-diagram-box" style="border-left: 4px solid #16a34a;">
                        <b>Report A</b> ({id_a}) ──┐<br>
                        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──► <b>{pred_a.predicted_cluster_id}</b> &nbsp;[Category: <b>{pred_a.category}</b> | Priority: <b>{pred_a.priority_score:.1f}</b>]<br>
                        <b>Report B</b> ({id_b}) ──┘
                    </div>
                    """)

                    st.success(
                        f"✅ **Semantic Fusion Verified**: Both dispatches were dynamically clustered into `{pred_a.predicted_cluster_id}`. "
                        f"Original report IDs (`{id_a}` and `{id_b}`) were mutually preserved as verifiable evidence."
                    )
                else:
                    render_html(f"""
                    <div class="fusion-diagram-box" style="border-left: 4px solid #2563eb;">
                        <b>Report A</b> ({id_a}) ────► <b>{pred_a.predicted_cluster_id}</b> &nbsp;[Category: <b>{pred_a.category}</b> | Priority: <b>{pred_a.priority_score:.1f}</b>]<br><br>
                        <b>Report B</b> ({id_b}) ────► <b>{pred_b.predicted_cluster_id}</b> &nbsp;[Category: <b>{pred_b.category}</b> | Priority: <b>{pred_b.priority_score:.1f}</b>]
                    </div>
                    """)

                    st.info(
                        f"ℹ️ **Distinct Incidents Identified**: The system determined these reports describe separate incidents "
                        f"(`{pred_a.predicted_cluster_id}` vs `{pred_b.predicted_cluster_id}`) based on semantic distance."
                    )

                # Detail cards
                col_res1, col_res2 = st.columns(2, gap="medium")
                with col_res1:
                    st.markdown(f"#### Report A Assessment (`{id_a}`)")
                    st.markdown(f"- **Incident**: `{pred_a.predicted_cluster_id}`")
                    st.markdown(f"- **Category**: `{pred_a.category}`")
                    st.markdown(f"- **Priority**: `{pred_a.priority_score:.1f}`")
                    st.markdown(f"- **Evidence IDs**: {', '.join([f'`{e}`' for e in pred_a.evidence_ids])}")
                with col_res2:
                    st.markdown(f"#### Report B Assessment (`{id_b}`)")
                    st.markdown(f"- **Incident**: `{pred_b.predicted_cluster_id}`")
                    st.markdown(f"- **Category**: `{pred_b.category}`")
                    st.markdown(f"- **Priority**: `{pred_b.priority_score:.1f}`")
                    st.markdown(f"- **Evidence IDs**: {', '.join([f'`{e}`' for e in pred_b.evidence_ids])}")

            except Exception as ex:
                st.error("Unable to execute incident fusion. Please check input parameters.")
                st.caption(f"Error diagnostics: {ex}")

# -------------------------------------------------------------
# 9. SCREEN 03: Batch Analysis
# -------------------------------------------------------------
elif nav_selection == "03 — Batch Analysis":
    st.markdown("""
    <div class="op-card-header">
        <div>
            <h2 class="op-card-title">📊 Batch Analysis</h2>
            <div class="op-card-subtitle">
                Review multiple incoming crisis reports in one pass, generate structured predictions, and export results.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c_upload, c_sample = st.columns([2, 1], gap="medium")
    with c_upload:
        uploaded_file = st.file_uploader(
            "Upload Batch Crisis File (CSV or JSONL)",
            type=["csv", "jsonl"],
            help="Files should contain report identifier and text columns."
        )
    with c_sample:
        st.markdown("<div style='margin-top: 26px;'></div>", unsafe_allow_html=True)
        load_sample_btn = st.button("📂 Load 20 TREC-IS Sample Reports", use_container_width=True)

    if "batch_reports" not in st.session_state:
        st.session_state["batch_reports"] = []
    if "batch_source_name" not in st.session_state:
        st.session_state["batch_source_name"] = ""

    # Handle sample loader
    if load_sample_btn:
        sample_path = Path("data/processed/trecis2020_a_eval_opaque.jsonl")
        if sample_path.exists():
            sample_loaded = []
            with open(sample_path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    if idx >= 20:
                        break
                    if line.strip():
                        rec = json.loads(line)
                        sample_loaded.append(Report(id=str(rec["item_id"]), text=str(rec["text"])))
            st.session_state["batch_reports"] = sample_loaded
            st.session_state["batch_source_name"] = "20 TREC-IS 2020-A sample reports"
            st.success(f"Loaded {len(sample_loaded)} real TREC-IS 2020-A evaluation reports.")
        else:
            st.error("Sample dataset file not found at data/processed/trecis2020_a_eval_opaque.jsonl.")

    # Handle file upload
    elif uploaded_file is not None:
        try:
            uploaded_loaded = []
            fname = uploaded_file.name.lower()
            if fname.endswith(".csv"):
                df_raw = pd.read_csv(uploaded_file)
                id_col = next((c for c in ["item_id", "id", "report_id"] if c in df_raw.columns), None)
                text_col = next((c for c in ["text", "report_text", "content"] if c in df_raw.columns), None)
                if not id_col or not text_col:
                    st.error(f"CSV missing ID/text columns. Found: {list(df_raw.columns)}")
                else:
                    for _, row in df_raw.iterrows():
                        uploaded_loaded.append(Report(id=str(row[id_col]), text=str(row[text_col])))
            elif fname.endswith(".jsonl"):
                for line in uploaded_file.getvalue().decode("utf-8").splitlines():
                    if line.strip():
                        d = json.loads(line)
                        id_k = next((c for c in ["item_id", "id", "report_id"] if c in d), None)
                        text_k = next((c for c in ["text", "report_text", "content"] if c in d), None)
                        if id_k and text_k:
                            uploaded_loaded.append(Report(id=str(d[id_k]), text=str(d[text_k])))
            if uploaded_loaded:
                st.session_state["batch_reports"] = uploaded_loaded
                st.session_state["batch_source_name"] = uploaded_file.name
                st.info(f"Loaded **{len(uploaded_loaded)}** reports from `{uploaded_file.name}`.")
        except Exception as err:
            st.error("Unable to parse uploaded file. Please verify file format.")
            st.caption(f"Error diagnostics: {err}")

    # Processing & Presentation
    active_reports = st.session_state.get("batch_reports", [])
    if active_reports:
        st.markdown(f"<div style='font-size: 0.85rem; color: #475569; margin: 8px 0;'>Active batch: <b>{len(active_reports)} reports</b> loaded ({st.session_state.get('batch_source_name', 'external')}).</div>", unsafe_allow_html=True)
        if st.button("➔ Run Batch Triage", type="primary", use_container_width=True):
            with st.spinner("Processing batch predictions through AI-03 pipeline..."):
                predictions = pipeline.process_batch(active_reports)

            # Compute real summary statistics
            total_reps = len(predictions)
            unique_clusters = len(set(p.predicted_cluster_id for p in predictions))
            high_pri_count = sum(1 for p in predictions if p.priority_score >= 3.8)

            st.markdown("---")
            st.markdown("### Batch Results Summary")

            # 3 Real KPI cards
            kpi1, kpi2, kpi3 = st.columns(3)
            with kpi1:
                st.metric("Reports Processed", total_reps)
            with kpi2:
                st.metric("Incidents Identified", unique_clusters)
            with kpi3:
                st.metric("High-Priority Reports (≥ 3.8)", high_pri_count)

            # Results Dataframe
            results_rows = []
            for p, r in zip(predictions, active_reports):
                results_rows.append({
                    "Item ID": p.item_id,
                    "Report (preview)": (r.text[:65] + "...") if len(r.text) > 65 else r.text,
                    "Incident": p.predicted_cluster_id,
                    "Category": p.category,
                    "Priority": f"{p.priority_score:.1f}",
                    "Evidence Count": len(p.evidence_ids),
                    "evidence_ids": ", ".join(p.evidence_ids)
                })

            df_results = pd.DataFrame(results_rows)
            st.dataframe(
                df_results[["Item ID", "Report (preview)", "Incident", "Category", "Priority", "Evidence Count"]],
                use_container_width=True,
                hide_index=True
            )

            # CSV Download
            csv_bytes = df_results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Predictions (CSV)",
                data=csv_bytes,
                file_name=f"crisis_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    else:
        st.markdown("""
        <div class="empty-state-box">
            <div style="font-size: 2rem; margin-bottom: 8px;">📊</div>
            <div style="font-size: 1.05rem; font-weight: 700; color: #334155; margin-bottom: 4px;">No reports loaded</div>
            <div style="font-size: 0.85rem; color: #64748b;">
                Upload a CSV or JSONL file, or load the official 20-sample evaluation dataset to begin batch analysis.
            </div>
        </div>
        """, unsafe_allow_html=True)

# -------------------------------------------------------------
# 10. Footer
# -------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; color: #94a3b8; font-size: 0.78rem; padding-bottom: 12px;">
    <div>AI-03 &nbsp;|&nbsp; Crisis Report Fusion & Priority Ranking</div>
    <div>Built for real-world impact.</div>
</div>
""", unsafe_allow_html=True)
