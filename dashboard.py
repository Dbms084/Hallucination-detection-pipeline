import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

# Set page config
st.set_page_config(
    page_title="Hallucination Detection & LLM Evaluation Platform",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark/glassmorphic styling
st.markdown("""
<style>
    /* Styling for metric cards */
    .metric-card {
        background-color: #1E293B;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        text-align: center;
        transition: transform 0.2s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #475569;
    }
    .metric-title {
        color: #94A3B8;
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #F8FAFC;
        font-size: 32px;
        font-weight: 700;
    }
    .metric-sub {
        font-size: 12px;
        margin-top: 6px;
        font-weight: 500;
    }
    /* Styling for insights card */
    .insights-card {
        background-color: #0F172A;
        border-radius: 12px;
        padding: 24px;
        border: 1px dashed #475569;
        margin-bottom: 20px;
    }
    .insights-title {
        color: #38BDF8;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔍 LLM Hallucination Evaluation Dashboard")
st.markdown("Evaluate factual correctness, semantic matching accuracy, and benchmarks across TruthfulQA questions.")
st.markdown("---")

CSV_FILE = "results/csv/evaluation_results.csv"

if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0:
    st.warning("⚠️ No evaluation results found. Please run the evaluation pipeline first to generate results (`python main.py` or run `scorer.py` on existing CSV).")
else:
    df = pd.read_csv(CSV_FILE)
    
    # Sidebar Filters and Controls
    st.sidebar.header("🎛️ Dashboard Controls")
    
    # Filter by Category
    categories = ["All"] + sorted(df["category"].unique().tolist())
    selected_category = st.sidebar.selectbox("Filter by Category", categories)
    
    # Interactive Threshold Simulation
    st.sidebar.subheader("🔬 Similarity Threshold Simulator")
    sim_threshold = st.sidebar.slider(
        "Semantic Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.60,
        step=0.05,
        help="Adjust the cosine similarity threshold to define what counts as a semantic match."
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 Export Summary Report")
    
    if st.sidebar.button("Generate PDF Report", help="Create a comprehensive PDF report using current threshold"):
        with st.spinner("Generating PDF Report..."):
            try:
                from src.reports.report_generator import build_pdf_report
                from src.scoring.disagreement_analytics import calculate_disagreement_stats, generate_disagreement_chart
                
                # Re-generate disagreement chart dynamically for the current threshold
                stats_for_chart = calculate_disagreement_stats(df, sim_threshold)
                if stats_for_chart:
                    generate_disagreement_chart(stats_for_chart, "results/visualizations")
                
                pdf_path = "results/reports/evaluation_summary.pdf"
                build_pdf_report(CSV_FILE, pdf_path, threshold=sim_threshold, vis_dir="results/visualizations")
                
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                    
                st.sidebar.download_button(
                    label="Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"evaluation_report_threshold_{sim_threshold:.2f}.pdf",
                    mime="application/pdf"
                )
                st.sidebar.success("PDF Report is ready!")
            except Exception as e:
                st.sidebar.error(f"Error generating PDF: {e}")
    
    # Filter dataframe by category
    filtered_df = df if selected_category == "All" else df[df["category"] == selected_category]
    
    # Re-calculate metric scores based on filter and threshold
    total_q = len(filtered_df)
    
    if total_q > 0:
        exact_acc = (filtered_df["exact_match"].sum() / total_q) * 100
        keyword_acc = (filtered_df["keyword_match"].sum() / total_q) * 100
        
        # Recalculate semantic match dynamically using the slider
        dynamic_semantic_matches = filtered_df["semantic_score"] >= sim_threshold
        semantic_acc = (dynamic_semantic_matches.sum() / total_q) * 100
        avg_semantic_score = filtered_df["semantic_score"].mean()
        
        # Calculate judge accuracy
        if "judge_score" in filtered_df.columns:
            judge_acc = (filtered_df["judge_score"].fillna(0).sum() / total_q) * 100
        else:
            judge_acc = 0
    else:
        exact_acc = keyword_acc = semantic_acc = avg_semantic_score = judge_acc = 0
    
    # KPI Metrics Section
    st.subheader("📈 Key Performance Indicators")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Evaluated</div>
            <div class="metric-value">{total_q}</div>
            <div class="metric-sub" style="color: #60A5FA;">Questions</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Exact Match</div>
            <div class="metric-value">{exact_acc:.1f}%</div>
            <div class="metric-sub" style="color: #F87171;">Literal Equality</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Keyword Match</div>
            <div class="metric-value">{keyword_acc:.1f}%</div>
            <div class="metric-sub" style="color: #34D399;">Key Terms Present</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Semantic Match</div>
            <div class="metric-value">{semantic_acc:.1f}%</div>
            <div class="metric-sub" style="color: #38BDF8;">Threshold: {sim_threshold:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Gemini Judge</div>
            <div class="metric-value">{judge_acc:.1f}%</div>
            <div class="metric-sub" style="color: #C084FC;">Factual correctness</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col6:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Avg Cos Sim</div>
            <div class="metric-value">{avg_semantic_score:.3f}</div>
            <div class="metric-sub" style="color: #F472B6;">Overall Similarity</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown(" ")
    st.markdown(" ")
    
    # Main Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Visual Analytics", 
        "🔍 Response Inspector", 
        "⚖️ Judge Disagreement Analytics",
        "⚠️ Failure Analysis",
        "🧩 Evaluator Summary & Insights",
        "📂 Dataset Viewer"
    ])
    
    with tab1:
        st.subheader("Performance & Metrics Visualizations")
        
        chart_col1, chart_col2 = st.columns(2)
        
        # Plot Category Performance Chart (Dynamic)
        with chart_col1:
            st.markdown("### Category Performance (Mean Semantic Similarity)")
            if total_q > 0:
                category_data = filtered_df.groupby("category")["semantic_score"].mean().reset_index()
                category_data.columns = ["Category", "Mean Similarity"]
                category_data = category_data.sort_values(by="Mean Similarity", ascending=True)
                
                # Apply style for clean dark plotting
                fig_cat, ax_cat = plt.subplots(figsize=(8, 5))
                fig_cat.patch.set_facecolor('#0F172A')
                ax_cat.set_facecolor('#1E293B')
                
                # Color bars dynamically based on whether they meet simulated threshold
                colors = ['#38BDF8' if x >= sim_threshold else '#F87171' for x in category_data["Mean Similarity"]]
                
                bars = ax_cat.barh(category_data["Category"], category_data["Mean Similarity"], color=colors)
                ax_cat.axvline(x=sim_threshold, color="#F43F5E", linestyle="--", linewidth=1.5, label=f"Threshold ({sim_threshold:.2f})")
                
                # Style ticks and labels
                ax_cat.tick_params(colors='#F8FAFC', labelsize=10)
                ax_cat.xaxis.label.set_color('#94A3B8')
                ax_cat.yaxis.label.set_color('#94A3B8')
                ax_cat.set_xlabel("Cosine Similarity Score")
                ax_cat.spines['top'].set_visible(False)
                ax_cat.spines['right'].set_visible(False)
                ax_cat.spines['bottom'].set_color('#475569')
                ax_cat.spines['left'].set_color('#475569')
                ax_cat.legend(facecolor='#1E293B', edgecolor='#475569', labelcolor='#F8FAFC')
                
                st.pyplot(fig_cat)
            else:
                st.info("No data available for Category Performance Chart.")
        
        # Plot Semantic Score Distribution (Dynamic)
        with chart_col2:
            st.markdown("### Semantic Score Distribution")
            if total_q > 0:
                fig_dist, ax_dist = plt.subplots(figsize=(8, 5))
                fig_dist.patch.set_facecolor('#0F172A')
                ax_dist.set_facecolor('#1E293B')
                
                n, bins, patches = ax_dist.hist(
                    filtered_df["semantic_score"], 
                    bins=10, 
                    range=(0.0, 1.0), 
                    color='#818CF8', 
                    edgecolor='#312E81',
                    alpha=0.85
                )
                
                # Highlight threshold line
                ax_dist.axvline(x=sim_threshold, color="#F43F5E", linestyle="--", linewidth=1.5, label=f"Threshold ({sim_threshold:.2f})")
                
                # Style ticks and labels
                ax_dist.tick_params(colors='#F8FAFC', labelsize=10)
                ax_dist.xaxis.label.set_color('#94A3B8')
                ax_dist.yaxis.label.set_color('#94A3B8')
                ax_dist.set_xlabel("Semantic Cosine Similarity Score")
                ax_dist.set_ylabel("Count of Responses")
                ax_dist.spines['top'].set_visible(False)
                ax_dist.spines['right'].set_visible(False)
                ax_dist.spines['bottom'].set_color('#475569')
                ax_dist.spines['left'].set_color('#475569')
                ax_dist.legend(facecolor='#1E293B', edgecolor='#475569', labelcolor='#F8FAFC')
                
                st.pyplot(fig_dist)
            else:
                st.info("No data available for Score Distribution.")
                
        st.markdown("---")
        st.markdown("### Pre-Generated Static Reports")
        report_col1, report_col2 = st.columns(2)
        with report_col1:
            comp_chart_path = "results/visualizations/evaluator_comparison.png"
            if os.path.exists(comp_chart_path):
                st.image(comp_chart_path, use_container_width=True, caption="Pre-generated accuracy across evaluators")
            else:
                st.info("Run `scorer.py` to generate the comparison chart image.")
        with report_col2:
            thresh_chart_path = "results/visualizations/threshold_analysis.png"
            if os.path.exists(thresh_chart_path):
                st.image(thresh_chart_path, use_container_width=True, caption="Pre-generated threshold sensitivity analysis")
            else:
                st.info("Run `scorer.py` to generate the threshold analysis chart image.")
                
    with tab2:
        st.subheader("TruthfulQA Response Inspector")
        st.markdown("Select a question to inspect the comparisons, scores, and full model outputs.")
        
        # Display interactive dropdown for selecting question
        question_list = filtered_df["question"].astype(str).str.replace("\\n", " ", regex=False).tolist()
        if question_list:
            selected_q_clean = st.selectbox("Select Question to Inspect", question_list, key="inspector_select")
            
            # Map clean selection back to original row
            original_idx = question_list.index(selected_q_clean)
            selected_row = filtered_df.iloc[original_idx]
            
            exp_col1, exp_col2 = st.columns([2, 3])
            
            with exp_col1:
                st.markdown("### Question Metadata & Targets")
                st.markdown(f"**Category:** `{selected_row['category']}`")
                
                st.markdown("**Ground Truth Answer:**")
                st.success(str(selected_row['ground_truth']).replace("\\n", "\n"))
                
                # Match results breakdown card
                st.markdown("### Score Breakdown")
                is_sem_match = selected_row['semantic_score'] >= sim_threshold
                
                # Check if judge fields are in data
                has_judge = "judge_score" in selected_row
                judge_score_val = selected_row["judge_score"] if has_judge else "N/A"
                judge_reason_val = selected_row["judge_reason"] if has_judge else "N/A"
                
                st.markdown(f"""
                *   **Exact Match:** {'✅ Match' if selected_row['exact_match'] else '❌ Mismatch'}
                *   **Keyword Match:** {'✅ Match' if selected_row['keyword_match'] else '❌ Mismatch'}
                *   **Semantic Match (at {sim_threshold:.2f}):** {'✅ Match' if is_sem_match else '❌ Mismatch'}
                *   **Cosine Similarity Score:** `{selected_row['semantic_score']:.4f}`
                """)
                
                if has_judge:
                    if pd.notna(judge_score_val):
                        st.markdown(f"""
                        *   **Gemini Judge Score:** {'✅ Factually Correct (1)' if judge_score_val == 1 else '❌ Factually Incorrect (0)'}
                        *   **Judge Explanation:** *"{judge_reason_val}"*
                        """)
                    elif pd.notna(judge_reason_val) and judge_reason_val != "N/A":
                        st.markdown(f"""
                        *   **Gemini Judge Score:** `Pending/Skipped`
                        *   **Judge Explanation:** *"{judge_reason_val}"*
                        """)
                
            with exp_col2:
                st.markdown("### LLM Response Output")
                # Format response by unescaping newlines
                formatted_response = str(selected_row['response']).replace("\\n", "\n")
                st.text_area("Gemini Output", value=formatted_response, height=350, disabled=True, key="inspector_text")
        else:
            st.write("No questions available for selection.")

    with tab3:
        st.subheader("⚖️ Judge Disagreement Analytics")
        st.markdown(
            "This section compares the vector-based **Semantic Similarity** and key-term **Keyword Match** "
            "metrics against the factual correctness verdicts from the **Gemini LLM Judge**."
        )
        
        from src.scoring.disagreement_analytics import calculate_disagreement_stats, generate_disagreement_chart
        
        stats = calculate_disagreement_stats(filtered_df, sim_threshold)
        
        if stats is None:
            st.warning("⚠️ No Gemini Judge score column (`judge_score`) found in the dataset, or the data is empty.")
        else:
            # Re-generate disagreement donut chart for current filtered dataframe & threshold
            generate_disagreement_chart(stats, "results/visualizations")
            
            # Show high-level metrics in a nice 3-column row
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric(
                    label="Agreement Rate",
                    value=f"{stats['agreement_rate']:.1f}%",
                    help=f"{stats['agreed_correct_count']} agreed correct, {stats['agreed_incorrect_count']} agreed incorrect"
                )
            with col_stat2:
                st.metric(
                    label="Semantic Disagreement Rate",
                    value=f"{stats['disagreement_rate']:.1f}%",
                    help=f"{stats['false_positive_count']} false positives, {stats['false_negative_count']} false negatives"
                )
            with col_stat3:
                st.metric(
                    label="Keyword Disagreement Rate",
                    value=f"{stats['keyword_disagreement_rate']:.1f}%",
                    help=f"{stats['keyword_fallacy_count']} keyword fallacies, {stats['keyword_omission_count']} keyword omissions"
                )
            
            # Detailed breakdown row
            col_b1, col_b2 = st.columns([3, 2])
            
            with col_b1:
                st.markdown("### Disagreement Type Breakdown")
                
                breakdown_df = pd.DataFrame({
                    "Disagreement Category": [
                        "False Positive (Semantic Pass, Judge Fail)",
                        "False Negative (Semantic Fail, Judge Pass)",
                        "Keyword Fallacy (Keyword Pass, Judge Fail)",
                        "Keyword Omission (Keyword Fail, Judge Pass)"
                    ],
                    "Count": [
                        stats["false_positive_count"],
                        stats["false_negative_count"],
                        stats["keyword_fallacy_count"],
                        stats["keyword_omission_count"]
                    ],
                    "Percentage of Total Valid": [
                        f"{stats['false_positive_rate']:.1f}%",
                        f"{stats['false_negative_rate']:.1f}%",
                        f"{(stats['keyword_fallacy_count'] / stats['total_evaluated']) * 100:.1f}%",
                        f"{(stats['keyword_omission_count'] / stats['total_evaluated']) * 100:.1f}%"
                    ]
                })
                st.dataframe(breakdown_df, use_container_width=True)
            
            with col_b2:
                st.markdown("### Agreement Distribution")
                donut_chart_path = "results/visualizations/disagreement_distribution.png"
                if os.path.exists(donut_chart_path):
                    st.image(donut_chart_path, use_container_width=True)
            
            # Interactive explorer for disagreement cases
            st.markdown("---")
            st.markdown("### 🔍 Interactive Disagreement Explorer")
            
            # Create a dataframe of cases where semantic similarity and judge disagreed
            disagree_cases = filtered_df.copy()
            disagree_cases["semantic_match_dynamic"] = disagree_cases["semantic_score"] >= sim_threshold
            
            # Filter for disagreements
            disagree_df = disagree_cases[disagree_cases["semantic_match_dynamic"] != (disagree_cases["judge_score"] == 1)].dropna(subset=["judge_score"]).reset_index(drop=True)
            
            st.write(f"Found **{len(disagree_df)}** disagreement cases in this category filter.")
            
            if len(disagree_df) > 0:
                disagree_question_list = disagree_df["question"].astype(str).str.replace("\\n", " ", regex=False).tolist()
                selected_disagree_q = st.selectbox("Select Disagreement Case to Inspect", disagree_question_list, key="disagree_explorer_select")
                
                d_idx = disagree_question_list.index(selected_disagree_q)
                d_row = disagree_df.iloc[d_idx]
                
                dc1, dc2 = st.columns([1, 1])
                with dc1:
                    st.markdown("**Question:**")
                    st.info(str(d_row["question"]).replace("\\n", "\n"))
                    st.markdown("**Ground Truth Answer:**")
                    st.success(str(d_row["ground_truth"]).replace("\\n", "\n"))
                with dc2:
                    st.markdown("**Model Response:**")
                    st.warning(str(d_row["response"]).replace("\\n", "\n"))
                    
                    st.markdown("##### Scores & Verdicts")
                    st.markdown(f"""
                    *   **Semantic Score:** `{d_row['semantic_score']:.4f}` (Match: {'✅ Yes' if d_row['semantic_match_dynamic'] else '❌ No'})
                    *   **Keyword Match:** {'✅ Yes' if d_row['keyword_match'] else '❌ No'}
                    *   **LLM Judge Score:** `{'✅ 1.0 (Correct)' if d_row['judge_score'] == 1 else '❌ 0.0 (Incorrect)'}`
                    *   **LLM Judge Reason:** *"{d_row['judge_reason']}"*
                    """)

    with tab4:
        st.subheader("⚠️ Hallucination & Failure Analysis")
        
        # Select analysis mode inside Failure Analysis tab
        analysis_mode = st.selectbox(
            "Select Failure View",
            ["All Semantic Failures", "Disagreements (Semantic Failed but Gemini Judge Passed)"]
        )
        
        if analysis_mode == "All Semantic Failures":
            failures = filtered_df[filtered_df["semantic_score"] < sim_threshold].reset_index(drop=True)
            st.markdown("This section displays evaluation entries where the response fell below the semantic similarity threshold.")
        else:
            if "judge_score" in filtered_df.columns:
                failures = filtered_df[
                    (filtered_df["semantic_score"] < sim_threshold) & 
                    (filtered_df["judge_score"] == 1)
                ].reset_index(drop=True)
                st.markdown("This section displays **false negative disagreements**—cases where semantic similarity threshold was too strict (failed), but the Gemini LLM Judge verified the response as factually correct (passed).")
            else:
                failures = pd.DataFrame()
                st.warning("No Gemini Judge score column found in data.")
                
        total_failures = len(failures)
        
        st.markdown(f"**Identified Cases:** `{total_failures}`")
        
        if total_failures > 0:
            fail_idx = st.number_input("Browse Cases (Index)", min_value=0, max_value=total_failures-1, value=0, step=1, key=f"fail_browse_{analysis_mode.replace(' ', '_')}")
            fail_row = failures.iloc[fail_idx]
            
            f_col1, f_col2 = st.columns([2, 3])
            with f_col1:
                st.markdown("### Case Details")
                st.markdown(f"**Question {fail_idx+1}:**")
                st.warning(str(fail_row['question']).replace("\\n", "\n"))
                
                st.markdown(f"**Category:** `{fail_row['category']}`")
                
                st.markdown("**Expected Ground Truth:**")
                st.success(str(fail_row['ground_truth']).replace("\\n", "\n"))
                
                st.markdown(f"**Metrics status:**")
                st.markdown(f"""
                - **Exact Match:** `{'Yes' if fail_row['exact_match'] else 'No'}`
                - **Keyword Match:** `{'Yes' if fail_row['keyword_match'] else 'No'}`
                - **Semantic Score:** `{fail_row['semantic_score']:.4f}` (Failed to meet `{sim_threshold:.2f}`)
                """)
                
                if "judge_score" in fail_row:
                    if pd.notna(fail_row['judge_score']):
                        st.markdown(f"""
                        - **Gemini Judge Score:** `{fail_row['judge_score']}`
                        - **Judge Reason:** *"{fail_row['judge_reason']}"*
                        """)
                    elif pd.notna(fail_row['judge_reason']) and fail_row['judge_reason'] != "N/A":
                        st.markdown(f"""
                        - **Gemini Judge Score:** `Pending/Skipped`
                        - **Judge Reason:** *"{fail_row['judge_reason']}"*
                        """)
            with f_col2:
                st.markdown("### Actual Model Response")
                formatted_fail_response = str(fail_row['response']).replace("\\n", "\n")
                st.text_area("Response Output", value=formatted_fail_response, height=350, disabled=True, key=f"failure_text_{analysis_mode.replace(' ', '_')}_{fail_idx}")
        else:
            st.success("🎉 No cases found matching the criteria under the current settings!")
 
    with tab5:
        st.subheader("🧩 Evaluator Summary & Insights")
        
        # Layout columns
        insights_col, confusion_col = st.columns([3, 2])
        
        with insights_col:
            st.markdown("### 💡 Automated Project Insights")
            
            if total_q > 0:
                # 1. Best / worst category analysis
                cat_means = filtered_df.groupby("category")["semantic_score"].mean()
                best_cat = cat_means.idxmax()
                best_val = cat_means.max()
                worst_cat = cat_means.idxmin()
                worst_val = cat_means.min()
                
                # 2. Match type discrepancies
                # High term-overlap but wrong meaning (Keyword True, Semantic False)
                keyword_true_semantic_false = len(filtered_df[(filtered_df["keyword_match"] == True) & (filtered_df["semantic_score"] < sim_threshold)])
                pct_overlap_hallucination = (keyword_true_semantic_false / total_q) * 100
                
                # Paraphrasing (Keyword False, Semantic True)
                keyword_false_semantic_true = len(filtered_df[(filtered_df["keyword_match"] == False) & (filtered_df["semantic_score"] >= sim_threshold)])
                pct_paraphrasing = (keyword_false_semantic_true / total_q) * 100
                
                st.markdown(f"""
                <div class="insights-card">
                    <div class="insights-title">🔍 Summary of Findings</div>
                    <ul>
                        <li>🏆 <b>Top Performing Category:</b> <code>{best_cat}</code> with an average similarity score of <b>{best_val:.3f}</b>.</li>
                        <li>📉 <b>Lowest Performing Category:</b> <code>{worst_cat}</code> with an average similarity score of <b>{worst_val:.3f}</b>.</li>
                        <li>🚨 <b>False-Semantic Overlap:</b> <b>{pct_overlap_hallucination:.1f}%</b> of responses (<code>{keyword_true_semantic_false}</code> questions) contain key terms from the answer but do not match the semantic meaning. These represent likely <b>term-overlap hallucinations</b>.</li>
                        <li>📝 <b>Paraphrase Agreement:</b> <b>{pct_paraphrasing:.1f}%</b> of responses (<code>{keyword_false_semantic_true}</code> questions) successfully convey the correct information using different vocabulary (failed keyword check but passed semantic check).</li>
                        <li>📊 <b>Model Generosity:</b> Keyword matching is <b>{((filtered_df['keyword_match'].mean() - (filtered_df['semantic_score'] >= sim_threshold).mean())*100):.1f}%</b> more lenient than Semantic Similarity at threshold <code>{sim_threshold:.2f}</code>.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.write("Run evaluation to compile insights.")
                
        with confusion_col:
            st.markdown("### 🧩 Confusion & Agreement Summary")
            
            if total_q > 0:
                # Calculate counts for Confusion Matrix / Panel
                both_match = len(filtered_df[(filtered_df["keyword_match"] == True) & (filtered_df["semantic_score"] >= sim_threshold)])
                keyword_only = len(filtered_df[(filtered_df["keyword_match"] == True) & (filtered_df["semantic_score"] < sim_threshold)])
                semantic_only = len(filtered_df[(filtered_df["keyword_match"] == False) & (filtered_df["semantic_score"] >= sim_threshold)])
                neither_match = len(filtered_df[(filtered_df["keyword_match"] == False) & (filtered_df["semantic_score"] < sim_threshold)])
                
                confusion_df = pd.DataFrame({
                    "Semantic Match (Pass)": [both_match, semantic_only],
                    "Semantic Match (Fail)": [keyword_only, neither_match]
                }, index=["Keyword Match (Pass)", "Keyword Match (Fail)"])
                
                st.table(confusion_df)
                
                st.markdown("""
                **Key Definitions:**
                *   **Keyword (Pass) & Semantic (Pass):** Full alignment. Response is verified correct and contains target keywords.
                *   **Keyword (Pass) & Semantic (Fail):** *Term-overlap Hallucination.* The response mentions the correct keywords but in a factually incorrect/hallucinated context.
                *   **Keyword (Fail) & Semantic (Pass):** *Paraphrased Correctness.* The response explains the correct concept without using the literal words in the ground truth.
                *   **Keyword (Fail) & Semantic (Fail):** Full mismatch. The model output is completely off-topic or incorrect.
                """)
            else:
                st.write("No data loaded.")
 
    with tab6:
        st.subheader("Raw Results Dataset")
        st.markdown("This table displays the complete evaluations from `evaluation_results.csv`.")
        
        # Display clean df in interactive table
        display_df = filtered_df.copy()
        # Clean formatting newlines for UI spreadsheet rendering
        for col in ["question", "ground_truth", "response"]:
            display_df[col] = display_df[col].astype(str).str.replace("\\n", " ", regex=False)
        st.dataframe(display_df, use_container_width=True)
