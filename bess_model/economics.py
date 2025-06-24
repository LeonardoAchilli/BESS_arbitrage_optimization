"""
BESS Economics Module for Hungarian Market
Calculates NPV, IRR, Payback, and LCOS for 10-year project lifecycle
Includes CAPEX, OPEX, battery replacement, and Hungarian-specific costs
"""

import numpy as np
from scipy.optimize import fsolve
from typing import Dict, List, Optional, Tuple

class BESSEconomics:
    """Economic analysis for BESS projects in Hungarian market"""
    
    def __init__(
        self,
        capacity_kWh: float,
        power_kW: float,
        capex_eur_per_kwh: float = 420,  # Placeholder as mentioned in requirements
        capex_power_eur_per_kw: float = 150,  # Inverter costs
        opex_annual_eur: float = 5000,  # Annual O&M
        hupx_license_eur: float = 15000,  # HUPX market access license
        installation_eur: float = 10000,  # Grid connection and installation
        project_lifetime_years: int = 10,
        discount_rate: float = 0.08,  # 8% WACC
        battery_replacement_soh_threshold: float = 0.70,  # Replace at 70% SOH
        battery_replacement_cost_factor: float = 0.7,  # Future battery cost reduction
        tax_rate: float = 0.09,  # Hungarian corporate tax rate
        depreciation_years: int = 10
    ):
        self.capacity_kWh = capacity_kWh
        self.power_kW = power_kW
        self.capex_eur_per_kwh = capex_eur_per_kwh
        self.capex_power_eur_per_kw = capex_power_eur_per_kw
        self.opex_annual_eur = opex_annual_eur
        self.hupx_license_eur = hupx_license_eur
        self.installation_eur = installation_eur
        self.project_lifetime_years = project_lifetime_years
        self.discount_rate = discount_rate
        self.battery_replacement_soh_threshold = battery_replacement_soh_threshold
        self.battery_replacement_cost_factor = battery_replacement_cost_factor
        self.tax_rate = tax_rate
        self.depreciation_years = depreciation_years
    
    def calculate_initial_capex(self) -> Dict[str, float]:
        """Calculate initial capital expenditure breakdown"""
        
        battery_capex = self.capacity_kWh * self.capex_eur_per_kwh
        inverter_capex = self.power_kW * self.capex_power_eur_per_kw
        
        capex_breakdown = {
            'battery_system_eur': battery_capex,
            'inverter_system_eur': inverter_capex,
            'installation_eur': self.installation_eur,
            'hupx_license_eur': self.hupx_license_eur,
            'total_capex_eur': battery_capex + inverter_capex + self.installation_eur + self.hupx_license_eur
        }
        
        return capex_breakdown
    
    def calculate_annual_opex(self, year: int = 1) -> float:
        """Calculate annual operational expenditure with inflation"""
        inflation_rate = 0.03  # 3% annual inflation
        return self.opex_annual_eur * (1 + inflation_rate) ** (year - 1)
    
    def calculate_depreciation(self, total_capex: float) -> List[float]:
        """Calculate annual depreciation (straight-line method)"""
        annual_depreciation = total_capex / self.depreciation_years
        return [annual_depreciation if year <= self.depreciation_years else 0 
                for year in range(1, self.project_lifetime_years + 1)]
    
    def determine_battery_replacement_year(self, simulation_results: Dict) -> Optional[int]:
        """Determine if/when battery replacement is needed based on SOH"""
        
        # Estimate SOH degradation over project lifetime
        annual_degradation = simulation_results.get('total_cycles', 300) / simulation_results.get('simulation_days', 365) * 365 * 0.02 / 365
        
        for year in range(1, self.project_lifetime_years + 1):
            projected_soh = 1.0 - (annual_degradation * year)
            if projected_soh <= self.battery_replacement_soh_threshold:
                return year
        
        return None  # No replacement needed
    
    def calculate_cash_flows(self, simulation_results: Dict) -> Dict:
        """Calculate 10-year cash flow projection"""
        
        # Initial costs
        capex = self.calculate_initial_capex()
        initial_investment = -capex['total_capex_eur']
        
        # Annual revenues and costs
        annual_revenue = simulation_results.get('annual_revenue_eur', 0)
        
        # Determine battery replacement
        replacement_year = self.determine_battery_replacement_year(simulation_results)
        replacement_cost = (self.capacity_kWh * self.capex_eur_per_kwh * 
                          self.battery_replacement_cost_factor) if replacement_year else 0
        
        # Calculate depreciation
        depreciation_schedule = self.calculate_depreciation(capex['total_capex_eur'])
        
        # Year-by-year cash flows
        cash_flows = [initial_investment]  # Year 0
        annual_details = []
        
        for year in range(1, self.project_lifetime_years + 1):
            # Revenue (might degrade with battery degradation)
            degradation_factor = 1.0 - (0.02 * year)  # 2% annual degradation
            revenue = annual_revenue * max(0.7, degradation_factor)  # Min 70% of original
            
            # Operating expenses
            opex = self.calculate_annual_opex(year)
            
            # Battery replacement cost
            replacement = -replacement_cost if year == replacement_year else 0
            
            # Depreciation (tax shield)
            depreciation = depreciation_schedule[year - 1]
            
            # Earnings before tax
            ebt = revenue - opex - depreciation
            
            # Tax
            tax = max(0, ebt * self.tax_rate)
            
            # Net income
            net_income = ebt - tax
            
            # Cash flow = Net income + Depreciation + Replacement cost
            annual_cash_flow = net_income + depreciation + replacement
            
            cash_flows.append(annual_cash_flow)
            
            annual_details.append({
                'year': year,
                'revenue_eur': revenue,
                'opex_eur': opex,
                'depreciation_eur': depreciation,
                'replacement_cost_eur': -replacement,
                'ebt_eur': ebt,
                'tax_eur': tax,
                'net_income_eur': net_income,
                'cash_flow_eur': annual_cash_flow
            })
        
        return {
            'initial_capex': capex,
            'annual_cash_flows': cash_flows,
            'annual_details': annual_details,
            'replacement_year': replacement_year,
            'replacement_cost_eur': replacement_cost
        }
    
    def calculate_npv(self, cash_flows: List[float]) -> float:
        """Calculate Net Present Value"""
        npv = 0
        for year, cash_flow in enumerate(cash_flows):
            npv += cash_flow / (1 + self.discount_rate) ** year
        return npv
    
    def calculate_irr(self, cash_flows: List[float]) -> Optional[float]:
        """Calculate Internal Rate of Return"""
        
        def npv_func(rate):
            return sum(cf / (1 + rate) ** year for year, cf in enumerate(cash_flows))
        
        try:
            # Initial guess based on discount rate
            irr = fsolve(npv_func, self.discount_rate)[0]
            
            # Validate IRR makes sense
            if -1 < irr < 2:  # Between -100% and 200%
                return irr
            else:
                return None
        except:
            return None
    
    def calculate_payback(self, cash_flows: List[float]) -> Optional[float]:
        """Calculate payback period in years"""
        
        cumulative_cash_flow = 0
        
        for year, cash_flow in enumerate(cash_flows):
            cumulative_cash_flow += cash_flow
            
            if cumulative_cash_flow > 0 and year > 0:
                # Linear interpolation for exact payback time
                prev_cumulative = cumulative_cash_flow - cash_flow
                payback = year - 1 + abs(prev_cumulative) / cash_flow
                return payback
        
        return None  # Payback not achieved within project lifetime
    
    def calculate_lcos(self, simulation_results: Dict, cash_flow_data: Dict) -> float:
        """Calculate Levelized Cost of Storage (EUR/MWh)"""
        
        # Total lifecycle costs (present value)
        total_costs_pv = 0
        for year, details in enumerate(cash_flow_data['annual_details']):
            costs = details['opex_eur'] + abs(details['replacement_cost_eur'])
            total_costs_pv += costs / (1 + self.discount_rate) ** (year + 1)
        
        # Add initial CAPEX
        total_costs_pv += cash_flow_data['initial_capex']['total_capex_eur']
        
        # Total energy throughput (present value)
        annual_energy_mwh = simulation_results.get('annual_revenue_eur', 0) / 100  # Rough estimate
        if annual_energy_mwh == 0:
            annual_energy_mwh = self.capacity_kWh * 300 / 1000  # Fallback: 300 cycles/year
        
        total_energy_pv = 0
        for year in range(1, self.project_lifetime_years + 1):
            degradation_factor = 1.0 - (0.02 * year)
            energy = annual_energy_mwh * max(0.7, degradation_factor)
            total_energy_pv += energy / (1 + self.discount_rate) ** year
        
        # LCOS = Total costs PV / Total energy PV
        if total_energy_pv > 0:
            return total_costs_pv / total_energy_pv
        else:
            return float('inf')
    
    def run_economic_analysis(self, simulation_results: Dict) -> Dict:
        """Run complete economic analysis"""
        
        # Calculate cash flows
        cash_flow_data = self.calculate_cash_flows(simulation_results)
        cash_flows = cash_flow_data['annual_cash_flows']
        
        # Calculate financial metrics
        npv = self.calculate_npv(cash_flows)
        irr = self.calculate_irr(cash_flows)
        payback = self.calculate_payback(cash_flows)
        lcos = self.calculate_lcos(simulation_results, cash_flow_data)
        
        return {
            'npv_eur': npv,
            'irr': irr,
            'payback_years': payback,
            'lcos_eur_per_mwh': lcos,
            'cash_flow_data': cash_flow_data,
            'project_summary': {
                'capacity_kWh': self.capacity_kWh,
                'power_kW': self.power_kW,
                'initial_capex_eur': cash_flow_data['initial_capex']['total_capex_eur'],
                'annual_revenue_eur': simulation_results.get('annual_revenue_eur', 0),
                'project_lifetime_years': self.project_lifetime_years,
                'discount_rate': self.discount_rate
            }
        }

# Convenience functions for standalone use
def calculate_npv(cash_flows: List[float], discount_rate: float = 0.08) -> float:
    """Standalone NPV calculation"""
    return sum(cf / (1 + discount_rate) ** year for year, cf in enumerate(cash_flows))

def calculate_irr(cash_flows: List[float]) -> Optional[float]:
    """Standalone IRR calculation"""
    def npv_func(rate):
        return sum(cf / (1 + rate) ** year for year, cf in enumerate(cash_flows))
    
    try:
        irr = fsolve(npv_func, 0.08)[0]
        return irr if -1 < irr < 2 else None
    except:
        return None

def calculate_payback(cash_flows: List[float]) -> Optional[float]:
    """Standalone payback calculation"""
    cumulative = 0
    for year, cf in enumerate(cash_flows):
        cumulative += cf
        if cumulative > 0 and year > 0:
            prev_cumulative = cumulative - cf
            return year - 1 + abs(prev_cumulative) / cf
    return None

def calculate_lcos(total_costs: float, total_energy_mwh: float) -> float:
    """Standalone LCOS calculation"""
    return total_costs / total_energy_mwh if total_energy_mwh > 0 else float('inf')

# Example usage
if __name__ == "__main__":
    # Example simulation results
    example_results = {
        'annual_revenue_eur': 50000,
        'total_cycles': 300,
        'simulation_days': 365,
        'capacity_kWh': 500
    }
    
    # Run economic analysis
    economics = BESSEconomics(capacity_kWh=500, power_kW=250)
    analysis = economics.run_economic_analysis(example_results)
    
    print("BESS Economic Analysis Results:")
    print(f"NPV: {analysis['npv_eur']:,.0f} EUR")
    print(f"IRR: {analysis['irr']:.1%}" if analysis['irr'] else "IRR: Not viable")
    print(f"Payback: {analysis['payback_years']:.1f} years" if analysis['payback_years'] else "Payback: > 10 years")
    print(f"LCOS: {analysis['lcos_eur_per_mwh']:.0f} EUR/MWh")
