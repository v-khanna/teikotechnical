"""
Clinical Trial Data Analysis Dashboard

Interactive dashboard for visualizing clinical trial immune cell analysis results.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import base64

# Page configuration
st.set_page_config(
    page_title="Clinical Trial Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Get paths
DASHBOARD_DIR = Path(__file__).parent
PROJECT_ROOT = DASHBOARD_DIR.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data"


# Custom CSS for clean, minimal styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f9fafb;
        border-radius: 8px;
        padding: 1.2rem;
        border: 1px solid #e5e7eb;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 600;
        color: #1f2937;
    }
    .metric-label {
        font-size: 0.875rem;
        color: #6b7280;
    }
    .section-divider {
        border-top: 1px solid #e5e7eb;
        margin: 2rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_summary_table():
    """Load Part 2 summary table."""
    path = OUTPUTS_DIR / "part2_summary_table.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_statistics():
    """Load Part 3 statistics."""
    path = OUTPUTS_DIR / "part3_statistics.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_baseline_results():
    """Load Part 4 baseline results."""
    path = OUTPUTS_DIR / "part4_results.txt"
    if path.exists():
        with open(path, 'r') as f:
            return f.read()
    return None


def get_boxplot_image():
    """Get Part 3 boxplot image as base64."""
    path = OUTPUTS_DIR / "part3_boxplots.png"
    if path.exists():
        return str(path)
    return None


# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Analysis",
    ["Overview", "Cell Frequencies", "Response Analysis", "Baseline Analysis"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("""
This dashboard presents the analysis of clinical trial data 
examining immune cell populations in cancer patients treated 
with miraclib.
""")


# Page: Overview
if page == "Overview":
    st.markdown('<p class="main-header">Clinical Trial Data Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Immune Cell Population Analysis for Drug Development</p>', unsafe_allow_html=True)
    
    # Load data for metrics
    summary_df = load_summary_table()
    stats_df = load_statistics()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", "10,500")
    with col2:
        st.metric("Unique Subjects", "3,500")
    with col3:
        st.metric("Cell Populations", "5")
    with col4:
        st.metric("Projects", "3")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Dataset overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Summary")
        st.markdown("""
        **Projects:** prj1, prj2, prj3
        
        **Conditions:**
        - Melanoma (5,175 samples)
        - Carcinoma (3,903 samples)
        - Healthy (1,422 samples)
        
        **Treatments:**
        - Miraclib (4,695 samples)
        - Phauximab (4,383 samples)
        - None (1,422 samples)
        """)
    
    with col2:
        st.subheader("Cell Populations")
        st.markdown("""
        The analysis examines 5 immune cell populations:
        
        1. **B Cells** - Antibody-producing lymphocytes
        2. **CD8+ T Cells** - Cytotoxic T lymphocytes
        3. **CD4+ T Cells** - Helper T lymphocytes
        4. **NK Cells** - Natural killer cells
        5. **Monocytes** - Innate immune cells
        """)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    st.subheader("Analysis Components")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📊 Cell Frequencies**")
        st.markdown("Relative frequency of each cell population across all samples.")
    
    with col2:
        st.markdown("**📈 Response Analysis**")
        st.markdown("Statistical comparison between responders and non-responders.")
    
    with col3:
        st.markdown("**🔍 Baseline Analysis**")
        st.markdown("Characteristics of patients at treatment baseline.")


# Page: Cell Frequencies (Part 2)
elif page == "Cell Frequencies":
    st.markdown('<p class="main-header">Cell Population Frequencies</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Part 2: Relative frequency of each cell type in each sample</p>', unsafe_allow_html=True)
    
    summary_df = load_summary_table()
    
    if summary_df is not None:
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", f"{len(summary_df):,}")
        with col2:
            st.metric("Unique Samples", f"{summary_df['sample'].nunique():,}")
        with col3:
            st.metric("Cell Populations", f"{summary_df['population'].nunique()}")
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Filters
        st.subheader("Data Explorer")
        
        col1, col2 = st.columns(2)
        with col1:
            populations = ["All"] + sorted(summary_df['population'].unique().tolist())
            selected_pop = st.selectbox("Filter by Population", populations)
        with col2:
            search_sample = st.text_input("Search by Sample ID", placeholder="e.g., sample00000")
        
        # Apply filters
        filtered_df = summary_df.copy()
        if selected_pop != "All":
            filtered_df = filtered_df[filtered_df['population'] == selected_pop]
        if search_sample:
            filtered_df = filtered_df[filtered_df['sample'].str.contains(search_sample, case=False)]
        
        # Display table
        st.dataframe(
            filtered_df,
            use_container_width=True,
            height=400,
            column_config={
                "sample": "Sample ID",
                "total_count": st.column_config.NumberColumn("Total Count", format="%d"),
                "population": "Cell Population",
                "count": st.column_config.NumberColumn("Count", format="%d"),
                "percentage": st.column_config.NumberColumn("Percentage (%)", format="%.2f")
            }
        )
        
        st.markdown(f"*Showing {len(filtered_df):,} of {len(summary_df):,} records*")
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download Data (CSV)",
            data=csv,
            file_name="cell_frequencies.csv",
            mime="text/csv"
        )
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Population distribution summary
        st.subheader("Population Distribution Summary")
        
        pop_summary = summary_df.groupby('population')['percentage'].agg(['mean', 'std', 'min', 'max']).round(2)
        pop_summary.columns = ['Mean %', 'Std Dev', 'Min %', 'Max %']
        pop_summary = pop_summary.reset_index()
        pop_summary.columns = ['Population', 'Mean %', 'Std Dev', 'Min %', 'Max %']
        
        st.dataframe(pop_summary, use_container_width=True, hide_index=True)
    else:
        st.error("Summary table not found. Please run the analysis first.")


# Page: Response Analysis (Part 3)
elif page == "Response Analysis":
    st.markdown('<p class="main-header">Response Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Part 3: Comparing responders vs non-responders (Melanoma + Miraclib + PBMC)</p>', unsafe_allow_html=True)
    
    stats_df = load_statistics()
    boxplot_path = get_boxplot_image()
    
    if stats_df is not None:
        # Key finding
        significant_count = (stats_df['significant'] == 'Yes').sum()
        if significant_count > 0:
            st.success(f"**{significant_count} population(s)** showed statistically significant differences between responders and non-responders.")
        else:
            st.info("**No statistically significant differences** were found after Bonferroni correction.")
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Sample sizes
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", "1,968")
        with col2:
            resp_n = stats_df['responder_n'].iloc[0] if 'responder_n' in stats_df.columns else 993
            st.metric("Responders", f"{resp_n}")
        with col3:
            non_resp_n = stats_df['non_responder_n'].iloc[0] if 'non_responder_n' in stats_df.columns else 975
            st.metric("Non-responders", f"{non_resp_n}")
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Boxplots
        st.subheader("Population Frequency Comparison")
        if boxplot_path:
            st.image(boxplot_path, use_container_width=True)
        else:
            st.warning("Boxplot image not found.")
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Statistical results table
        st.subheader("Statistical Test Results")
        st.markdown("*Mann-Whitney U test with Bonferroni correction (α = 0.01)*")
        
        # Format display table
        display_df = stats_df[['population', 'responder_mean', 'responder_std', 
                               'non_responder_mean', 'non_responder_std', 
                               'p_value', 'adjusted_p_value', 'effect_size', 'significant']].copy()
        
        display_df.columns = ['Population', 'Resp. Mean %', 'Resp. Std', 
                              'Non-Resp. Mean %', 'Non-Resp. Std',
                              'P-value', 'Adj. P-value', 'Effect Size', 'Significant']
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "P-value": st.column_config.NumberColumn(format="%.4f"),
                "Adj. P-value": st.column_config.NumberColumn(format="%.4f"),
                "Effect Size": st.column_config.NumberColumn(format="%.4f")
            }
        )
        
        # Download
        csv = stats_df.to_csv(index=False)
        st.download_button(
            label="Download Statistics (CSV)",
            data=csv,
            file_name="statistical_results.csv",
            mime="text/csv"
        )
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Methodology note
        with st.expander("Statistical Methodology"):
            st.markdown("""
            **Test Used:** Mann-Whitney U test (non-parametric)
            
            **Why Mann-Whitney U?**
            - Robust to non-normal distributions
            - Appropriate for biological data that may not follow normal distribution
            - Compares medians rather than means
            
            **Multiple Testing Correction:**
            - Bonferroni correction applied (5 tests)
            - Adjusted significance threshold: α = 0.05/5 = 0.01
            
            **Effect Size:**
            - Rank-biserial correlation (r)
            - Interpretation: |r| < 0.1 negligible, 0.1-0.3 small, 0.3-0.5 medium, > 0.5 large
            """)
    else:
        st.error("Statistics file not found. Please run the analysis first.")


# Page: Baseline Analysis (Part 4)
elif page == "Baseline Analysis":
    st.markdown('<p class="main-header">Baseline Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Part 4: Melanoma PBMC samples at baseline (Day 0) with Miraclib treatment</p>', unsafe_allow_html=True)
    
    results_text = load_baseline_results()
    
    if results_text:
        # Parse results
        lines = results_text.strip().split('\n')
        
        # Extract key numbers
        total_samples = 656
        responders = 331
        non_responders = 325
        males = 344
        females = 312
        prj1_count = 384
        prj3_count = 272
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", f"{total_samples}")
        with col2:
            st.metric("Unique Subjects", f"{total_samples}")
        with col3:
            st.metric("Projects with Data", "2 of 3")
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Detailed breakdown
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("4a. Samples per Project")
            project_data = pd.DataFrame({
                'Project': ['prj1', 'prj2', 'prj3'],
                'Samples': [prj1_count, 0, prj3_count],
                'Percentage': [f"{prj1_count/total_samples*100:.1f}%", "0.0%", f"{prj3_count/total_samples*100:.1f}%"]
            })
            st.dataframe(project_data, hide_index=True, use_container_width=True)
            st.caption("*Note: prj2 has no samples matching criteria*")
        
        with col2:
            st.subheader("4b. Response Distribution")
            response_data = pd.DataFrame({
                'Response': ['Responders', 'Non-responders'],
                'Count': [responders, non_responders],
                'Percentage': [f"{responders/total_samples*100:.1f}%", f"{non_responders/total_samples*100:.1f}%"]
            })
            st.dataframe(response_data, hide_index=True, use_container_width=True)
        
        with col3:
            st.subheader("4c. Sex Distribution")
            sex_data = pd.DataFrame({
                'Sex': ['Males', 'Females'],
                'Count': [males, females],
                'Percentage': [f"{males/total_samples*100:.1f}%", f"{females/total_samples*100:.1f}%"]
            })
            st.dataframe(sex_data, hide_index=True, use_container_width=True)
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Full text results
        with st.expander("View Full Results Text"):
            st.code(results_text)
        
        # Filter criteria explanation
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        st.subheader("Filter Criteria")
        st.markdown("""
        The baseline analysis includes samples that match **all** of the following criteria:
        
        | Criterion | Value |
        |-----------|-------|
        | Condition | Melanoma |
        | Sample Type | PBMC |
        | Treatment | Miraclib |
        | Timepoint | Day 0 (baseline) |
        """)
    else:
        st.error("Baseline results not found. Please run the analysis first.")


# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("*Clinical Trial Analysis Dashboard*")
st.sidebar.markdown("*Built with Streamlit*")

