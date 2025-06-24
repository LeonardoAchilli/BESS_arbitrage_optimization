"""
BESS Hungary - Streamlit Web Application
Calculates optimal sizing (100-1000 kWh) and economic analysis for BESS arbitrage in HUPX market
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from pathlib import Path

# Import our BESS modules
try:
    from bess_model.optimisation import BESSOptimizer
    from bess_model.economics import BESSEconomics
except ImportError:
    st.error("Please ensure bess_model modules are in the correct directory")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="BESS Hungary Calculator",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("🔋 BESS Hungary - HUPX Arbitrage Calculator")
    st.markdown("**Optimal sizing and economic analysis for 100-1000 kWh battery storage systems in Hungarian market**")
    
    # Sidebar inputs
    st.sidebar.header("⚙️ System Configuration")
    
    # Core system parameters
    st.sidebar.subheader("Battery System")
    capacity_kWh = st.sidebar.slider(
        "Capacity (kWh)", 
        min_value=100, 
        max_value=1000, 
        value=500, 
        step=50,
        help="Battery energy capacity in kWh"
    )
    
    # Calculate default power based on duration
    default_duration = st.sidebar.selectbox(
        "System Duration", 
        options=[2, 4, 6, 8], 
        index=0,
        help="Hours of discharge at rated power"
    )
    power_kW = capacity_kWh / default_duration
    
    # Allow manual power override
    power_kW = st.sidebar.number_input(
        "Power Rating (kW)", 
        min_value=50.0, 
        max_value=1000.0, 
        value=float(power_kW),
        step=25.0,
        help="Inverter power rating in kW"
    )
    
    round_trip_eff = st.sidebar.slider(
        "Round-trip Efficiency (%)", 
        min_value=75, 
        max_value=95, 
        value=87,
        help="Battery round-trip efficiency (default 87% from HUPX data)"
    ) / 100
    
    # Economic parameters
    st.sidebar.subheader("Economics")
    capex_eur_per_kwh = st.sidebar.number_input(
        "Battery CAPEX (€/kWh)", 
        min_value=200.0, 
        max_value=800.0, 
        value=420.0,
        step=10.0,
        help="⚠️ VALIDATE - Placeholder value, check current market prices"
    )
    
    capex_power_eur_per_kw = st.sidebar.number_input(
        "Inverter CAPEX (€/kW)",
        min_value=50.0,
        max_value=300.0,
        value=150.0,
        step=10.0
    )
    
    # Additional costs
    st.sidebar.subheader("Additional Costs")
    hupx_license_eur = st.sidebar.number_input(
        "HUPX License (€)",
        min_value=0.0,
        max_value=50000.0,
        value=15000.0,
        step=1000.0,
        help="Market access license fee"
    )
    
    installation_eur = st.sidebar.number_input(
        "Installation & Grid Connection (€)",
        min_value=0.0,
        max_value=100000.0,
        value=10000.0,
        step=1000.0
    )
    
    opex_annual_eur = st.sidebar.number_input(
        "Annual O&M (€/year)",
        min_value=0.0,
        max_value=20000.0,
        value=5000.0,
        step=500.0,
        help="Operation & Maintenance costs per year"
    )
    
    discount_rate = st.sidebar.slider(
        "Discount Rate (%)",
        min_value=5,
        max_value=15,
        value=8,
        help="Weighted Average Cost of Capital (WACC)"
    ) / 100
    
    # Run simulation button
    st.sidebar.markdown("---")
    run_simulation = st.sidebar.button("🚀 Run Simulation", type="primary")
    
    # Main content area
    if run_simulation:
        run_bess_analysis(
            capacity_kWh=capacity_kWh,
            power_kW=power_kW,
            round_trip_efficiency=round_trip_eff,
            capex_eur_per_kwh=capex_eur_per_kwh,
            capex_power_eur_per_kw=capex_power_eur_per_kw,
            hupx_license_eur=hupx_license_eur,
            installation_eur=installation_eur,
            opex_annual_eur=opex_annual_eur,
            discount_rate=discount_rate
        )
    else:
        # Show welcome message and info
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 📊 Based on Real HUPX Market Data
            
            This calculator uses actual Hungarian Power Exchange (HUPX) price patterns:
            - **Solar hours (10:00-15:00)**: 18-40 EUR/MWh ☀️
            - **Evening peak (18:00-21:00)**: 150-210 EUR/MWh ⚡
            - **Average net spread**: 110-170 EUR/MWh
            - **Round-trip efficiency**: 87%
            
            ### 🎯 Arbitrage Strategy
            - **Charge** during 4 cheapest hours of the day
            - **Discharge** during 4 most expensive hours 
            - Account for battery degradation and replacement
            - 10-year economic analysis with Hungarian tax rates
            """)
            
        with col2:
            st.markdown("""
            ### 📋 Quick Start
            1. Adjust system size (100-1000 kWh)
            2. Set economic parameters
            3. Click "Run Simulation"
            4. Review KPI results
            5. Download detailed CSV
            
            ### ⚠️ Important
            - CAPEX values are placeholders
            - Validate with current market prices
            - Results for preliminary analysis only
            """)

def run_bess_analysis(**params):
    """Run complete BESS analysis and display results"""
    
    with st.spinner("Running BESS simulation... Please wait"):
        
        try:
            # Step 1: Run technical simulation
            optimizer = BESSOptimizer(
                capacity_kWh=params['capacity_kWh'],
                power_kW=params['power_kW'],
                round_trip_efficiency=params['round_trip_efficiency']
            )
            
            simulation_results = optimizer.run_simulation()
            
            # Step 2: Run economic analysis
            economics = BESSEconomics(
                capacity_kWh=params['capacity_kWh'],
                power_kW=params['power_kW'],
                capex_eur_per_kwh=params['capex_eur_per_kwh'],
                capex_power_eur_per_kw=params['capex_power_eur_per_kw'],
                hupx_license_eur=params['hupx_license_eur'],
                installation_eur=params['installation_eur'],
                opex_annual_eur=params['opex_annual_eur'],
                discount_rate=params['discount_rate']
            )
            
            economic_results = economics.run_economic_analysis(simulation_results)
            
            # Display results
            display_results(simulation_results, economic_results, params)
            
        except Exception as e:
            st.error(f"Error running simulation: {str(e)}")
            st.error("Please check your input parameters and try again")

def display_results(simulation_results, economic_results, params):
    """Display simulation results in organized layout"""
    
    st.success("✅ Simulation completed successfully!")
    
    # KPI Cards
    st.subheader("📈 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        npv = economic_results['npv_eur']
        npv_color = "normal" if npv >= 0 else "inverse"
        st.metric(
            label="NPV (10 years)",
            value=f"€{npv:,.0f}",
            delta=None,
            help="Net Present Value over 10-year project lifetime"
        )
    
    with col2:
        irr = economic_results['irr']
        if irr is not None:
            st.metric(
                label="IRR",
                value=f"{irr:.1%}",
                delta=f"{irr - params['discount_rate']:.1%} vs WACC",
                help="Internal Rate of Return"
            )
        else:
            st.metric(label="IRR", value="Not viable", help="Project does not achieve positive IRR")
    
    with col3:
        payback = economic_results['payback_years']
        if payback is not None:
            st.metric(
                label="Payback Period",
                value=f"{payback:.1f} years",
                help="Time to recover initial investment"
            )
        else:
            st.metric(label="Payback Period", value="> 10 years", help="Payback exceeds project lifetime")
    
    with col4:
        lcos = economic_results['lcos_eur_per_mwh']
        st.metric(
            label="LCOS",
            value=f"€{lcos:.0f}/MWh",
            help="Levelized Cost of Storage"
        )
    
    # Technical metrics
    st.subheader("🔧 Technical Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Annual Revenue",
            value=f"€{simulation_results['annual_revenue_eur']:,.0f}",
            help="Projected annual arbitrage revenue"
        )
    
    with col2:
        st.metric(
            label="Daily Revenue",
            value=f"€{simulation_results['avg_daily_revenue_eur']:.2f}",
            help="Average daily arbitrage revenue"
        )
    
    with col3:
        st.metric(
            label="Annual Cycles",
            value=f"{simulation_results['total_cycles'] * 365 / simulation_results['simulation_days']:.0f}",
            help="Full charge-discharge cycles per year"
        )
    
    with col4:
        st.metric(
            label="Final SOH",
            value=f"{simulation_results['final_soh']:.1%}",
            help="Battery State of Health after simulation period"
        )
    
    # Cash flow chart
    st.subheader("💰 10-Year Cash Flow Analysis")
    
    # Prepare cash flow data
    cash_flows = economic_results['cash_flow_data']['annual_cash_flows']
    cumulative_cash_flows = np.cumsum(cash_flows)
    years = list(range(len(cash_flows)))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot cumulative cash flow
    ax.plot(years, cumulative_cash_flows, 'b-', linewidth=3, marker='o', markersize=6, label='Cumulative Cash Flow')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Break-even')
    ax.fill_between(years, cumulative_cash_flows, 0, alpha=0.3, where=(np.array(cumulative_cash_flows) >= 0), color='green', label='Positive')
    ax.fill_between(years, cumulative_cash_flows, 0, alpha=0.3, where=(np.array(cumulative_cash_flows) < 0), color='red', label='Negative')
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Cumulative Cash Flow (EUR)')
    ax.set_title('BESS Project - Cumulative Cash Flow')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Format y-axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'€{x/1000:.0f}K'))
    
    st.pyplot(fig)
    
    # Detailed breakdown
    with st.expander("📊 Detailed Annual Breakdown"):
        
        # Prepare detailed data
        annual_details = economic_results['cash_flow_data']['annual_details']
        df_details = pd.DataFrame(annual_details)
        df_details = df_details.round(0)
        
        # Add year 0 (initial investment)
        year_0 = {
            'year': 0,
            'revenue_eur': 0,
            'opex_eur': 0,
            'depreciation_eur': 0,
            'replacement_cost_eur': economic_results['cash_flow_data']['initial_capex']['total_capex_eur'],
            'ebt_eur': -economic_results['cash_flow_data']['initial_capex']['total_capex_eur'],
            'tax_eur': 0,
            'net_income_eur': -economic_results['cash_flow_data']['initial_capex']['total_capex_eur'],
            'cash_flow_eur': cash_flows[0]
        }
        
        df_complete = pd.concat([pd.DataFrame([year_0]), df_details], ignore_index=True)
        df_complete['cumulative_cash_flow_eur'] = df_complete['cash_flow_eur'].cumsum()
        
        st.dataframe(df_complete, use_container_width=True)
    
    # Download section
    st.subheader("💾 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Create summary CSV
        summary_data = {
            'Parameter': [
                'Capacity (kWh)', 'Power (kW)', 'Round-trip Efficiency',
                'Annual Revenue (EUR)', 'NPV (EUR)', 'IRR (%)', 
                'Payback (years)', 'LCOS (EUR/MWh)'
            ],
            'Value': [
                params['capacity_kWh'], params['power_kW'], f"{params['round_trip_efficiency']:.1%}",
                f"{simulation_results['annual_revenue_eur']:,.0f}", f"{economic_results['npv_eur']:,.0f}",
                f"{economic_results['irr']:.1%}" if economic_results['irr'] else "N/A",
                f"{economic_results['payback_years']:.1f}" if economic_results['payback_years'] else "> 10",
                f"{economic_results['lcos_eur_per_mwh']:,.0f}"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        csv_summary = summary_df.to_csv(index=False)
        
        st.download_button(
            label="📄 Download Summary CSV",
            data=csv_summary,
            file_name=f"bess_summary_{params['capacity_kWh']}kWh.csv",
            mime="text/csv"
        )
    
    with col2:
        # Create detailed CSV
        detailed_df = pd.DataFrame(annual_details)
        csv_detailed = detailed_df.to_csv(index=False)
        
        st.download_button(
            label="📊 Download Detailed CSV",
            data=csv_detailed,
            file_name=f"bess_detailed_{params['capacity_kWh']}kWh.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
