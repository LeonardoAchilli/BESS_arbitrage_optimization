from .optimisation import BESSOptimizer, run_arbitrage_simulation
from .economics import BESSEconomics, calculate_npv, calculate_irr, calculate_payback, calculate_lcos

def run(capacity_kWh=500, power_kW=None, **kwargs):
    """Main entry point for BESS simulation with economic analysis"""
    
    if power_kW is None:
        power_kW = capacity_kWh / 2  # Default 2h duration
    
    # Run technical simulation
    optimizer = BESSOptimizer(capacity_kWh, power_kW, **kwargs)
    simulation_results = optimizer.run_simulation()
    
    # Run economic analysis
    economics = BESSEconomics(capacity_kWh, power_kW, **kwargs)
    economic_results = economics.run_economic_analysis(simulation_results)
    
    # Combine results
    combined_results = {
        'simulation': simulation_results,
        'economics': economic_results,
        'NPV': economic_results['npv_eur'],
        'IRR': economic_results['irr'],
        'payback_years': economic_results['payback_years'],
        'LCOS': economic_results['lcos_eur_per_mwh']
    }
    
    return combined_results
