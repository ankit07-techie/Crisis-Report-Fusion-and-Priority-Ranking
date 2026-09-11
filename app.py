"""Crisis Report Fusion and Priority Ranking (AI-03)
Streamlit Showcase & Emergency Triage Dashboard.
Conforms strictly to Sections 9 & 10 of the PRD:
- Professional, human-centered UI without chatbot gimmicks or fake animations.
- Single report analysis with incident cluster, category, priority, and evidence IDs.
- Demonstrates semantic fusion of differently worded reports into the same incident.
- Batch evaluation and export mode.
"""

import streamlit as st
import pandas as pd
import json
from typing import List

from src.schemas import Report, Prediction
from src.pipeline import CrisisPipeline
from src.config import DEFAULT_CONFIG

# Configure page
st.set_page_config(
    page_title="CrisisAI - Incident Fusion & Priority Ranking",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean, professional emergency operations UI
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
    .badge-critical {
        background-color: #ef4444;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-high {
        background-color: #f97316;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-medium {
        background-color: #eab308;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-low {
        background-color: #3b82f6;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .evidence-pill {
        display: inline-block;
        background-color: #e0e7ff;
        color: #3730a3;
        border: 1px solid #c7d2fe;
        border-radius: 12px;
        padding: 2px 8px;
        margin: 2px;
        font-family: monospace;
        font-size: 0.85rem;
    }
    .fusion-match {
        background-color: #ecfdf5;
        border: 1px solid #10b981;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline():
    """Cache the pipeline instance across reruns."""
    return CrisisPipeline()


pipeline = get_pipeline()

# Sidebar: System Diagnostics & Settings
with st.sidebar:
    st.header("Emergency Ops Status")
    st.info("System: **AI-03 Integration Platform**")
    
    st.markdown("### Active Subsystems")
    st.markdown(f"- **P1 Fusion Engine**: `{pipeline.p1_source}`")
    st.markdown(f"- **P2 Classifier**: `{pipeline.p2_source}`")
    st.markdown("- **P3 Evidence Tracker**: `src.evidence:EvidenceRegistry`")
    
    st.markdown("---")
    st.markdown("### Pipeline Thresholds")
    sim_threshold = st.slider(
        "Clustering Similarity Threshold",
        min_value=0.1,
        max_value=0.9,
        value=float(DEFAULT_CONFIG.similarity_threshold),
        step=0.05,
        help="Higher values require closer semantic matches to fuse reports into the same incident."
    )
    pipeline.config = type(DEFAULT_CONFIG)(similarity_threshold=sim_threshold)
    
    if st.button("Reset Indexed Incidents"):
        pipeline.reset()
        st.success("Incident memory cleared.")

# Header
st.markdown('<div class="main-title">🚨 Crisis Report Fusion & Priority Ranking</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">AI-03 Emergency Operations Center — Automated incident clustering, multi-class categorization, operational priority scoring, and verifiable evidence tracing.</div>',
    unsafe_allow_html=True
)

tab1, tab2, tab3 = st.tabs([
    "📥 Live Report Triage",
    "🔗 Semantic Fusion Demonstration",
    "📊 Batch Evaluation & Upload"
])

# -------------------------------------------------------------
# TAB 1: Live Report Triage
# -------------------------------------------------------------
with tab1:
    st.markdown("### Real-Time Report Ingestion")
    st.caption("Paste an incoming civilian or field report to classify its humanitarian category, score urgency, and link it to incident clusters.")
    
    col_input, col_preset = st.columns([3, 1])
    
    with col_preset:
        preset = st.selectbox(
            "Quick Incident Presets",
            [
                "Custom Text",
                "Flash Flood Rooftop Rescue",
                "Highway Bridge Structural Collapse",
                "Industrial Gas Leak",
                "Hospital Generator Failure",
                "Water & Ration Shortage"
            ]
        )
        
    preset_texts = {
        "Flash Flood Rooftop Rescue": "Water levels reaching the second floor on Elm Street. Elderly resident and 2 children stranded on the roof, immediate boat rescue required!",
        "Highway Bridge Structural Collapse": "The main suspension bridge on Highway 9 has collapsed under floodwaters. Multiple vehicles stranded and roadway blocked.",
        "Industrial Gas Leak": "Strong smell of propane and chemical gas leaking near the industrial park warehouse. Residents coughing, urgent evacuation advised.",
        "Hospital Generator Failure": "St. Jude regional hospital backup generator submerged and failing. ICU units at risk, emergency power needed immediately.",
        "Water & Ration Shortage": "Evacuation shelter at Community Center has run completely out of potable drinking water and infant formula."
    }
    
    default_text = preset_texts.get(preset, "")
    
    with col_input:
        report_id = st.text_input("Report / Item ID", value="REP-2026-081", help="Opaque report identifier")
        report_text = st.text_area("Crisis Report Text", value=default_text, height=130, placeholder="Enter field dispatch or civilian crisis report...")
        
    if st.button("Analyze Report", type="primary"):
        if not report_text.strip():
            st.warning("Please enter crisis report text before analyzing.")
        else:
            report_obj = Report(id=report_id.strip(), text=report_text.strip())
            prediction = pipeline.process_report(report_obj)
            
            st.markdown("---")
            st.subheader("Analysis Results")
            
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Assigned Incident", prediction.predicted_cluster_id)
            with m2:
                st.metric("Information Category", prediction.category)
            with m3:
                pri = prediction.priority_score
                pri_label = "CRITICAL" if pri >= 4.5 else "HIGH" if pri >= 3.8 else "MEDIUM" if pri >= 2.8 else "LOW"
                st.metric("Priority Score (1-5)", f"{pri:.1f} / 5.0", delta=pri_label, delta_color="inverse" if pri >= 3.8 else "normal")
            with m4:
                st.metric("Corroborating Reports", len(prediction.evidence_ids))
                
            # Evidence IDs preservation display
            st.markdown("#### Verifiable Evidence IDs")
            if prediction.evidence_ids:
                pills = " ".join([f'<span class="evidence-pill">{eid}</span>' for eid in prediction.evidence_ids])
                st.markdown(f"<div>{pills}</div>", unsafe_allow_html=True)
            else:
                st.write("No corroborating reports currently indexed.")
                
            # Related reports in this incident cluster
            related_reports = pipeline.get_related_reports(prediction.predicted_cluster_id)
            if len(related_reports) > 1:
                with st.expander(f"View All {len(related_reports)} Reports in {prediction.predicted_cluster_id}"):
                    for r in related_reports:
                        st.markdown(f"**`{r.id}`**: {r.text}")

# -------------------------------------------------------------
# TAB 2: Semantic Fusion Demonstration (PRD Section 10 Goal)
# -------------------------------------------------------------
with tab2:
    st.markdown("### Incident Semantic Fusion Evaluation (PRD Section 10)")
    st.caption(
        "Demonstrate that two differently worded crisis reports referring to the same underlying physical emergency "
        "are dynamically fused into the same incident cluster without hardcoding."
    )
    
    demo_scenarios = {
        "Disaster Overpass Collapse (Differently Worded)": (
            "The main bridge over Route 9 has collapsed under rapid floodwaters, traffic halted and cars stopped.",
            "Water currents washed out the Route 9 overpass structure completely, several vehicles stranded on the road."
        ),
        "Chemical Plant Hazard": (
            "Toxic chemical odor and heavy smoke billowing from the chemical plant on Sector 4.",
            "Gas leak and fumes detected near the industrial chemical facility, residents experiencing burning throats."
        ),
        "Divergent Incidents (Should NOT Fuse)": (
            "Severe floodwaters washing away cars on River Road, families trapped on upper floors.",
            "Need immediate delivery of dry rations and drinking water bottles at the high school gym shelter."
        )
    }
    
    selected_demo = st.selectbox("Select Demonstration Scenario", list(demo_scenarios.keys()))
    preset_a, preset_b = demo_scenarios[selected_demo]
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Report A**")
        id_a = st.text_input("ID A", value="DISPATCH_A")
        text_a = st.text_area("Report A Text", value=preset_a, height=110)
    with col_b:
        st.markdown("**Report B**")
        id_b = st.text_input("ID B", value="DISPATCH_B")
        text_b = st.text_area("Report B Text", value=preset_b, height=110)
        
    if st.button("Evaluate Semantic Fusion", type="primary"):
        rep_a = Report(id=id_a, text=text_a)
        rep_b = Report(id=id_b, text=text_b)
        
        preds = pipeline.process_batch([rep_a, rep_b])
        pred_a = next(p for p in preds if p.item_id == id_a)
        pred_b = next(p for p in preds if p.item_id == id_b)
        
        st.markdown("---")
        st.subheader("Fusion Results")
        
        r1, r2 = st.columns(2)
        with r1:
            st.markdown(f"#### Report A (`{id_a}`)")
            st.write(f"**Predicted Cluster**: `{pred_a.predicted_cluster_id}`")
            st.write(f"**Category**: `{pred_a.category}`")
            st.write(f"**Priority Score**: `{pred_a.priority_score}`")
            st.write(f"**Evidence IDs**: {pred_a.evidence_ids}")
            
        with r2:
            st.markdown(f"#### Report B (`{id_b}`)")
            st.write(f"**Predicted Cluster**: `{pred_b.predicted_cluster_id}`")
            st.write(f"**Category**: `{pred_b.category}`")
            st.write(f"**Priority Score**: `{pred_b.priority_score}`")
            st.write(f"**Evidence IDs**: {pred_b.evidence_ids}")
            
        is_fused = (pred_a.predicted_cluster_id == pred_b.predicted_cluster_id)
        if is_fused:
            st.success(
                f"✅ **Semantic Fusion Verified**: Both reports were assigned to identical cluster `{pred_a.predicted_cluster_id}`! "
                f"Evidence IDs were mutually preserved (`{id_a}` and `{id_b}`)."
            )
        else:
            st.info(
                f"ℹ️ **Distinct Incidents Identified**: The system determined these reports describe separate incidents (`{pred_a.predicted_cluster_id}` vs `{pred_b.predicted_cluster_id}`)."
            )

# -------------------------------------------------------------
# TAB 3: Batch Evaluation & Upload Mode
# -------------------------------------------------------------
with tab3:
    st.markdown("### Batch Evaluation Runner")
    st.caption("Upload a file with opaque report IDs and text (CSV or JSONL) to process multiple reports simultaneously.")
    
    uploaded_file = st.file_uploader("Upload Evaluation File", type=["csv", "jsonl"])
    
    if uploaded_file is not None:
        try:
            reports: List[Report] = []
            fname = uploaded_file.name.lower()
            
            if fname.endswith(".csv"):
                df_raw = pd.read_csv(uploaded_file)
                id_col = next((c for c in ["item_id", "id", "report_id"] if c in df_raw.columns), None)
                text_col = next((c for c in ["text", "report_text", "content"] if c in df_raw.columns), None)
                if not id_col or not text_col:
                    st.error(f"CSV must contain an ID column and a text column. Found: {list(df_raw.columns)}")
                else:
                    for _, row in df_raw.iterrows():
                        reports.append(Report(id=str(row[id_col]), text=str(row[text_col])))
            elif fname.endswith(".jsonl"):
                lines = uploaded_file.getvalue().decode("utf-8").splitlines()
                for line in lines:
                    if line.strip():
                        d = json.loads(line)
                        id_k = next((c for c in ["item_id", "id", "report_id"] if c in d), None)
                        text_k = next((c for c in ["text", "report_text", "content"] if c in d), None)
                        if id_k and text_k:
                            reports.append(Report(id=str(d[id_k]), text=str(d[text_k])))
                            
            st.write(f"Loaded **{len(reports)}** valid records.")
            
            if st.button("Run Batch Triage", type="primary"):
                batch_preds = pipeline.process_batch(reports)
                
                # Format into table
                results_data = []
                for p in batch_preds:
                    results_data.append({
                        "Item ID": p.item_id,
                        "Cluster ID": p.predicted_cluster_id,
                        "Category": p.category,
                        "Priority": p.priority_score,
                        "Evidence IDs": ", ".join(p.evidence_ids)
                    })
                    
                df_res = pd.DataFrame(results_data)
                st.dataframe(df_res, use_container_width=True)
                
                # Export options
                csv_out = df_res.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Predictions (CSV)",
                    data=csv_out,
                    file_name="evaluation_predictions.csv",
                    mime="text/csv"
                )
        except Exception as err:
            st.error(f"Error processing file: {err}")
