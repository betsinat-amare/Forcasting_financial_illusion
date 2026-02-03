
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# Set page config
st.set_page_config(
    page_title="Ethiopia Financial Inclusion Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- Data Loading Functions ---
@st.cache_data
def load_data():
    file_path = 'data/raw/ethiopia_fi_unified_data.xlsx'
    if not os.path.exists(file_path):
        # Handle relative path if run from root
        file_path = '../data/raw/ethiopia_fi_unified_data.xlsx'
    
    xl = pd.ExcelFile(file_path)
    df = xl.parse('ethiopia_fi_unified_data')
    return df

@st.cache_data
def load_forecasts():
    file_path = 'reports/forecast_results.csv'
    if not os.path.exists(file_path):
        file_path = '../reports/forecast_results.csv'
    return pd.read_csv(file_path)

@st.cache_data
def load_impacts():
    file_path = 'reports/impact_matrix.csv'
    if not os.path.exists(file_path):
        file_path = '../reports/impact_matrix.csv'
    if os.path.exists(file_path):
        return pd.read_csv(file_path, index_name=0)
    return None

# --- App Layout ---
st.title("🇪🇹 Ethiopia Financial Inclusion Dashboard")
st.markdown("Exploring Trends, Impacts, and Forecasts for Financial Inclusion in Ethiopia.")

# Sidebar Navigation
page = st.sidebar.radio("Select Page", ["Overview", "Trends", "Forecasts", "Inclusion Projections"])

try:
    df = load_data()
    forecast_df = load_forecasts()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# --- Shared Utilities ---
def get_indicator_data(indicator_code):
    subset = df[df['indicator_code'] == indicator_code].copy()
    subset['date'] = pd.to_datetime(subset['observation_date'])
    subset['value'] = pd.to_numeric(subset['value_numeric'], errors='coerce')
    return subset.dropna(subset=['value']).sort_values('date')

# --- Page: Overview ---
if page == "Overview":
    st.header("Financial Inclusion Overview")
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    # Latest Account Ownership
    acc_data = get_indicator_data('ACC_OWNERSHIP')
    if not acc_data.empty:
        latest_acc = acc_data.iloc[-1]
        prev_acc = acc_data.iloc[-2] if len(acc_data) > 1 else latest_acc
        diff = latest_acc['value'] - prev_acc['value']
        col1.metric("Account Ownership", f"{latest_acc['value']}%", f"{diff:+.1f}% vs prev")
    
    # Latest Mobile Money
    mm_data = get_indicator_data('ACC_MM_ACCOUNT')
    if not mm_data.empty:
        latest_mm = mm_data.iloc[-1]
        prev_mm = mm_data.iloc[-2] if len(mm_data) > 1 else latest_mm
        mm_diff = latest_mm['value'] - prev_mm['value']
        col2.metric("Mobile Money", f"{latest_mm['value']}%", f"{mm_diff:+.1f}% vs prev")
        
    # Crossover Ratio
    p2p_data = get_indicator_data('USG_P2P_VALUE')
    atm_data = get_indicator_data('USG_ATM_VALUE')
    if not p2p_data.empty and not atm_data.empty:
        crossover = p2p_data.iloc[-1]['value'] / atm_data.iloc[-1]['value']
        col3.metric("P2P/ATM Ratio", f"{crossover:.2f}")
    
    # Growth Highlight (e.g. 4G coverage)
    cov_data = get_indicator_data('ACC_4G_COV')
    if not cov_data.empty:
        col4.metric("4G Coverage", f"{cov_data.iloc[-1]['value']}%")

    st.markdown("---")
    
    # Crossover Visualization
    st.subheader("Growth Momentum & Infrastructure")
    col_a, col_b = st.columns([2, 1])
    
    with col_a:
        if not p2p_data.empty and not atm_data.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=p2p_data['date'], y=p2p_data['value'], name='Digital (P2P)', line=dict(color='green')))
            fig.add_trace(go.Scatter(x=atm_data['date'], y=atm_data['value'], name='Cash (ATM)', line=dict(color='orange')))
            fig.update_layout(title="Transaction Volume: Digital vs Cash", xaxis_title="Year", yaxis_title="Volume")
            st.plotly_chart(fig, use_container_width=True)
    
    with col_b:
        st.info("**Crossover Analysis**: Digital P2P volumes are rapidly approaching traditional ATM volumes, signaling a fundamental shift in user behavior toward mobile-first finance.")

# --- Page: Trends ---
elif page == "Trends":
    st.header("Historical Trends Explorer")
    
    all_codes = df['indicator_code'].dropna().unique().tolist()
    
    col_l, col_r = st.columns([1, 3])
    with col_l:
        selected_indicators = st.multiselect("Select Indicators", all_codes, default=['ACC_OWNERSHIP', 'ACC_MM_ACCOUNT'])
        date_range = st.date_input("Date Range", [df['observation_date'].min(), df['observation_date'].max()])
    
    with col_r:
        if selected_indicators:
            fig = go.Figure()
            for code in selected_indicators:
                data = get_indicator_data(code)
                # Filter by date range
                data = data[(data['date'].dt.date >= date_range[0]) & (data['date'].dt.date <= date_range[1])]
                if not data.empty:
                    fig.add_trace(go.Scatter(x=data['date'], y=data['value'], mode='lines+markers', name=code))
            
            fig.update_layout(title="Multi-Indicator Growth Comparison", xaxis_title="Year", yaxis_title="Value")
            st.plotly_chart(fig, use_container_width=True)
        
    st.subheader("Raw Data Table")
    st.dataframe(df[df['indicator_code'].isin(selected_indicators)])

# --- Page: Forecasts ---
elif page == "Forecasts":
    st.header("Strategic Forecasts (2025-2027)")
    
    target_ind = st.selectbox("Indicator to Forecast", forecast_df['Indicator'].unique())
    model_type = st.radio("Forecast Model", ["Linear Trend", "Log-Trend (Conservative)"], horizontal=True)
    
    sub_f = forecast_df[forecast_df['Indicator'] == target_ind]
    scenario = st.multiselect("Scenarios", ["Base", "Optimistic", "Pessimistic"], default=["Base", "Optimistic"])
    
    fig = go.Figure()
    
    # Historical
    hist = sub_f[sub_f['Scenario'] == 'Historical']
    fig.add_trace(go.Scatter(x=hist['Year'], y=hist['Value'], name='Historical Data', line=dict(color='black', width=3)))
    
    colors = {'Base': 'blue', 'Optimistic': 'green', 'Pessimistic': 'red'}
    
    for s in scenario:
        scen_data = sub_f[sub_f['Scenario'] == s]
        if not scen_data.empty:
            # Join line from last historical point
            last_hist = hist.iloc[-1]
            scen_plot = pd.concat([pd.DataFrame([last_hist]), scen_data])
            
            fig.add_trace(go.Scatter(x=scen_plot['Year'], y=scen_plot['Value'], name=f"{s} Projection", line=dict(color=colors[s], dash='dash')))
            fig.add_trace(go.Scatter(
                x=scen_data['Year'].tolist() + scen_data['Year'].tolist()[::-1],
                y=scen_data['Upper_CI'].tolist() + scen_data['Lower_CI'].tolist()[::-1],
                fill='toself', fillcolor=colors[s], opacity=0.1,
                line=dict(color='rgba(255,255,255,0)'), showlegend=False
            ))
            
    # Milestones Highlights
    milestones = {
        2025: "National Access Goal (60%)",
        2026: "FX Liberalization Maturity",
        2027: "Ecosystem Maturation"
    }
    for year, text in milestones.items():
        fig.add_vline(x=year, line_width=1, line_dash="dot", line_color="gray")
        fig.add_annotation(x=year, y=hist['Value'].max(), text=text, textangle=-90, yshift=10)

    fig.update_layout(title=f"Forecast Analysis for {target_ind}", xaxis_title="Year", yaxis_title="Percentage (%)")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Projected Growth Table")
    st.table(sub_f[sub_f['Scenario'] != 'Historical'][['Year', 'Scenario', 'Value', 'Lower_CI', 'Upper_CI']])

# --- Page: Inclusion Projections ---
elif page == "Inclusion Projections":
    st.header("Financial Inclusion Goal Tracking")
    
    st.markdown("""
    ### Consortium Strategic Questions:
    1. **Will we hit the 60% goal by 2025?** 
    2. **What driving factors matter most?** 
    3. **What is the worst-case scenario?**
    """)
    
    target_value = 60
    acc_f = forecast_df[(forecast_df['Indicator'] == 'ACC_OWNERSHIP') & (forecast_df['Scenario'] != 'Historical')]
    
    st.subheader("Progress Toward 60% National Target")
    fig = px.bar(acc_f[acc_f['Year'] <= 2025], x='Scenario', y='Value', color='Scenario', 
                 title="Projected Inclusion Level in 2025", text_auto='.1f')
    fig.add_hline(y=target_value, line_dash="solid", line_color="green", annotation_text="Target: 60%")
    st.plotly_chart(fig, use_container_width=True)
    
    # Analysis logic
    base_2025 = acc_f[(acc_f['Year'] == 2025) & (acc_f['Scenario'] == 'Base')]['Value'].values[0]
    opt_2025 = acc_f[(acc_f['Year'] == 2025) & (acc_f['Scenario'] == 'Optimistic')]['Value'].values[0]
    
    if base_2025 >= target_value:
        st.success(f"**Base Scenario Result**: Target MET ({base_2025:.1f}%)")
    else:
        st.warning(f"**Base Scenario Result**: Target MISSED ({base_2025:.1f}%) - Policy intensification required.")

    st.info(f"**Optimistic Outlook**: If recent FX reforms and Digital ID adoption accelerate, inclusion could reach **{opt_2025:.1f}%** by year-end 2025.")

# Data Download
st.sidebar.markdown("---")
if st.sidebar.button("Download Full Dataset"):
    csv = df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button("Click to Download", data=csv, file_name="ethiopia_fi_data.csv")
