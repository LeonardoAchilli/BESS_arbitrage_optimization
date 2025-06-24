"""
BESS Optimization Module for Hungarian HUPX Market
Implements simple heuristic arbitrage strategy:
- Charge during 4 cheapest hours of the day (typically 10:00-15:00)  
- Discharge during 4 most expensive hours (typically 18:00-21:00)
- Account for round-trip efficiency, degradation, and grid constraints
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class BESSOptimizer:
    """BESS optimization for HUPX arbitrage operations"""
    
    def __init__(
        self,
        capacity_kWh: float,
        power_kW: float,
        round_trip_efficiency: float = 0.87,  # From PDF: 87% efficiency
        cycles_per_day: float = 1.0,
        degradation_yearly: float = 0.02,  # 2% per year
        min_soc: float = 0.1,  # 10% minimum SOC
        max_soc: float = 0.9,  # 90% maximum SOC  
        data_path: str = "data/hu_spot_2024.csv"
    ):
        self.capacity_kWh = capacity_kWh
        self.power_kW = power_kW
        self.round_trip_efficiency = round_trip_efficiency
        self.cycles_per_day = cycles_per_day
        self.degradation_yearly = degradation_yearly
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.data_path = data_path
        
        # Load price data
        self.price_data = self._load_price_data()
        
        # Initialize simulation state
        self.soc = 0.5  # Start at 50% SOC
        self.soh = 1.0  # Start at 100% SOH (State of Health)
        
    def _load_price_data(self) -> pd.DataFrame:
        """Load Hungarian spot price data"""
        try:
            df = pd.read_csv(self.data_path)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime').reset_index(drop=True)
            return df
        except FileNotFoundError:
            # Create sample data if file doesn't exist
            return self._create_sample_data()
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample price data based on HUPX patterns"""
        np.random.seed(42)
        
        # Create 1 week of sample data
        date_range = pd.date_range(start='2024-01-01', periods=24*7, freq='H')
        
        prices = []
        for dt in date_range:
            hour = dt.hour
            
            # Base pattern from PDF data
            if 10 <= hour <= 15:  # Solar hours
                base_price = np.random.uniform(18, 40)
            elif 18 <= hour <= 21:  # Evening peak
                base_price = np.random.uniform(150, 210)
            else:  # Other hours
                base_price = np.random.uniform(60, 120)
                
            prices.append({
                'datetime': dt,
                'hour': hour,
                'price_eur_mwh': round(base_price, 2)
            })
        
        return pd.DataFrame(prices)
    
    def _get_daily_arbitrage_strategy(self, daily_prices: pd.DataFrame) -> Tuple[List[int], List[int]]:
        """
        Simple heuristic: charge during 4 cheapest hours, discharge during 4 most expensive
        Returns: (charge_hours, discharge_hours)
        """
        # Sort by price to find cheapest and most expensive hours
        sorted_prices = daily_prices.sort_values('price_eur_mwh')
        
        # Get 4 cheapest hours for charging (but respect power constraint)
        max_charge_hours = min(4, int(self.capacity_kWh * (self.max_soc - self.min_soc) / self.power_kW))
        charge_hours = sorted_prices.head(max_charge_hours)['hour'].tolist()
        
        # Get 4 most expensive hours for discharging  
        max_discharge_hours = min(4, int(self.capacity_kWh * (self.max_soc - self.min_soc) / self.power_kW))
        discharge_hours = sorted_prices.tail(max_discharge_hours)['hour'].tolist()
        
        return charge_hours, discharge_hours
    
    def _simulate_day(self, daily_prices: pd.DataFrame, day_num: int) -> Dict:
        """Simulate one day of BESS operation"""
        
        # Get arbitrage strategy for this day
        charge_hours, discharge_hours = self._get_daily_arbitrage_strategy(daily_prices)
        
        daily_revenue = 0
        daily_energy_charged = 0
        daily_energy_discharged = 0
        hourly_operations = []
        
        for _, hour_data in daily_prices.iterrows():
            hour = hour_data['hour']
            price = hour_data['price_eur_mwh']
            
            operation = 'idle'
            energy_flow = 0
            revenue = 0
            
            # Charging logic
            if hour in charge_hours and self.soc < self.max_soc:
                # Calculate how much we can charge
                available_capacity = (self.max_soc - self.soc) * self.capacity_kWh * self.soh
                max_charge = min(self.power_kW, available_capacity)
                
                if max_charge > 0.1:  # Only if meaningful charge
                    energy_flow = max_charge
                    cost = energy_flow * price / 1000  # Convert to EUR
                    revenue = -cost  # Negative because we're buying
                    
                    # Update SOC (with charging efficiency)
                    self.soc += (energy_flow * np.sqrt(self.round_trip_efficiency)) / (self.capacity_kWh * self.soh)
                    self.soc = min(self.soc, self.max_soc)
                    
                    daily_energy_charged += energy_flow
                    operation = 'charge'
            
            # Discharging logic  
            elif hour in discharge_hours and self.soc > self.min_soc:
                # Calculate how much we can discharge
                available_energy = (self.soc - self.min_soc) * self.capacity_kWh * self.soh
                max_discharge = min(self.power_kW, available_energy)
                
                if max_discharge > 0.1:  # Only if meaningful discharge
                    energy_flow = max_discharge
                    # Account for discharge efficiency
                    sellable_energy = energy_flow * np.sqrt(self.round_trip_efficiency)
                    revenue = sellable_energy * price / 1000  # Convert to EUR
                    
                    # Update SOC
                    self.soc -= energy_flow / (self.capacity_kWh * self.soh)
                    self.soc = max(self.soc, self.min_soc)
                    
                    daily_energy_discharged += sellable_energy
                    operation = 'discharge'
            
            daily_revenue += revenue
            
            hourly_operations.append({
                'hour': hour,
                'price_eur_mwh': price,
                'operation': operation,
                'energy_kWh': energy_flow,
                'revenue_eur': revenue,
                'soc': self.soc,
                'soh': self.soh
            })
        
        # Apply daily degradation
        daily_cycles = daily_energy_discharged / (self.capacity_kWh * self.soh)
        degradation = daily_cycles * (self.degradation_yearly / 365)
        self.soh = max(0.7, self.soh - degradation)  # Min 70% SOH
        
        return {
            'day': day_num,
            'revenue_eur': daily_revenue,
            'energy_charged_kWh': daily_energy_charged,
            'energy_discharged_kWh': daily_energy_discharged,
            'cycles': daily_cycles,
            'soh_end': self.soh,
            'hourly_operations': hourly_operations
        }
    
    def run_simulation(self, simulation_days: Optional[int] = None) -> Dict:
        """Run full BESS arbitrage simulation"""
        
        if simulation_days is None:
            simulation_days = len(self.price_data) // 24
        
        # Group price data by day
        self.price_data['date'] = self.price_data['datetime'].dt.date
        daily_groups = self.price_data.groupby('date')
        
        simulation_results = []
        total_revenue = 0
        total_cycles = 0
        
        for day_num, (date, daily_data) in enumerate(daily_groups):
            if day_num >= simulation_days:
                break
                
            day_result = self._simulate_day(daily_data, day_num)
            simulation_results.append(day_result)
            
            total_revenue += day_result['revenue_eur']
            total_cycles += day_result['cycles']
        
        # Calculate summary statistics
        avg_daily_revenue = total_revenue / len(simulation_results) if simulation_results else 0
        annual_revenue = avg_daily_revenue * 365
        
        return {
            'simulation_days': len(simulation_results),
            'total_revenue_eur': total_revenue,
            'avg_daily_revenue_eur': avg_daily_revenue,
            'annual_revenue_eur': annual_revenue,
            'total_cycles': total_cycles,
            'final_soh': self.soh,
            'capacity_kWh': self.capacity_kWh,
            'power_kW': self.power_kW,
            'round_trip_efficiency': self.round_trip_efficiency,
            'daily_results': simulation_results
        }

def run_arbitrage_simulation(capacity_kWh: float = 500, **kwargs) -> Dict:
    """Convenience function to run BESS arbitrage simulation"""
    
    # Set default power rating to 2h duration if not specified
    power_kW = kwargs.get('power_kW', capacity_kWh / 2)
    
    optimizer = BESSOptimizer(capacity_kWh, power_kW, **kwargs)
    return optimizer.run_simulation()

# Example usage and testing
if __name__ == "__main__":
    # Test with 500 kWh system
    results = run_arbitrage_simulation(capacity_kWh=500, power_kW=250)
    
    print(f"BESS Simulation Results:")
    print(f"Capacity: {results['capacity_kWh']} kWh")
    print(f"Power: {results['power_kW']} kW") 
    print(f"Simulation days: {results['simulation_days']}")
    print(f"Annual revenue: {results['annual_revenue_eur']:.2f} EUR")
    print(f"Average daily revenue: {results['avg_daily_revenue_eur']:.2f} EUR")
    print(f"Total cycles: {results['total_cycles']:.1f}")
    print(f"Final SOH: {results['final_soh']:.1%}")
