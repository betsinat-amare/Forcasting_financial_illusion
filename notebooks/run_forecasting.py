
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
import os

# Configure plotting
plt.rcParams['figure.figsize'] = (12, 6)
sns.set_theme(style="whitegrid")
os.makedirs('reports', exist_ok=True)

# 1. Load Data
# --------------------------------------------------------------------------------
file_path = 'data/raw/ethiopia_fi_unified_data.xlsx'
print(f"Loading data from {file_path}...")
df = pd.read_excel(file_path, sheet_name='ethiopia_fi_unified_data')

# 2. Extract Target Indicators
# --------------------------------------------------------------------------------
target_indicators = ['ACC_OWNERSHIP', 'ACC_MM_ACCOUNT']
print(f"Filtering for indicators: {target_indicators}")

forecast_data = df[df['indicator_code'].isin(target_indicators)].copy()
forecast_data['date'] = pd.to_datetime(forecast_data['observation_date'])
forecast_data['year'] = forecast_data['date'].dt.year
forecast_data['value'] = pd.to_numeric(forecast_data['value_numeric'], errors='coerce')
forecast_data = forecast_data.dropna(subset=['value', 'year']).sort_values('year')

print(forecast_data[['year', 'indicator_code', 'value']].head(10))

# 3. Define Models
# --------------------------------------------------------------------------------
def linear_model(x, a, b):
    return a * x + b

def log_model(x, a, b):
    return a * np.log(x - 2010) + b # Offset year to keep log positive

# 4. Forecast Function
# --------------------------------------------------------------------------------
def generate_forecasts(data, indicator, years_to_forecast=[2025, 2026, 2027]):
    subset = data[data['indicator_code'] == indicator]
    if subset.empty:
        print(f"No data for {indicator}")
        return None
    
    X = subset['year'].values
    y = subset['value'].values
    
    # Fit Linear Model
    popt_lin, pcov_lin = curve_fit(linear_model, X, y)
    perr_lin = np.sqrt(np.diag(pcov_lin))
    
    # Generate Baseline Forecast
    X_future = np.array(years_to_forecast)
    y_future_lin = linear_model(X_future, *popt_lin)
    
    # Calculate Confidence Intervals (Simple approximation)
    # Using the standard error of the fit to estimate uncertainty
    residuals = y - linear_model(X, *popt_lin)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    std_err = np.std(residuals)
    
    print(f"Indicator: {indicator}, R-squared: {r_squared:.2f}")
    
    # Scenarios formulation
    # Base: Linear Trend
    # Optimistic: +1.5 * StdDev growth (simulating accelerated adoption due to reforms)
    # Pessimistic: -0.5 * StdDev growth (simulating stagnation)
    
    results = []
    
    # Historical Data for Plotting
    for x_val, y_val in zip(X, y):
        results.append({
            'Indicator': indicator,
            'Year': int(x_val),
            'Scenario': 'Historical',
            'Value': y_val,
            'Lower_CI': np.nan,
            'Upper_CI': np.nan
        })
        
    # Future Forecasts
    for i, year in enumerate(years_to_forecast):
        base_val = y_future_lin[i]
        
        # Uncertainty widens with time
        uncertainty = std_err * (1 + 0.2 * (i + 1)) 
        
        # Scenario Adjustments (cumulative effect)
        optimistic_boost = uncertainty * 1.5 
        pessimistic_drag = uncertainty * 0.5
        
        scenarios = {
            'Base': base_val,
            'Optimistic': base_val + optimistic_boost,
            'Pessimistic': base_val - pessimistic_drag
        }
        
        for name, val in scenarios.items():
            # Cap at 100% and floor at 0%
            val = max(0, min(100, val))
            
            results.append({
                'Indicator': indicator,
                'Year': int(year),
                'Scenario': name,
                'Value': val,
                'Lower_CI': max(0, val - uncertainty), # Simple CI
                'Upper_CI': min(100, val + uncertainty)
            })
            
    return pd.DataFrame(results)

# 5. Execute Forecasting
# --------------------------------------------------------------------------------
all_results = []
for ind in target_indicators:
    print(f"\nProcessing {ind}...")
    res = generate_forecasts(forecast_data, ind)
    if res is not None:
        all_results.append(res)

final_df = pd.concat(all_results, ignore_index=True)
print("\nForecast Results:")
print(final_df.tail())

# Save Results
final_df.to_csv('reports/forecast_results.csv', index=False)
print("\nSaved forecasts to reports/forecast_results.csv")

# 6. Visualization
# --------------------------------------------------------------------------------
def plot_forecast(df, indicator):
    subset = df[df['Indicator'] == indicator]
    
    plt.figure(figsize=(10, 6))
    
    # Historical
    hist = subset[subset['Scenario'] == 'Historical']
    plt.plot(hist['Year'].values, hist['Value'].values, 'ko-', label='Historical', linewidth=2)
    
    # Scenarios
    scenarios = ['Base', 'Optimistic', 'Pessimistic']
    colors = {'Base': 'blue', 'Optimistic': 'green', 'Pessimistic': 'red'}
    styles = {'Base': '--', 'Optimistic': '-.', 'Pessimistic': ':'}
    
    for scen in scenarios:
        data = subset[subset['Scenario'] == scen]
        if not data.empty:
            plt.plot(data['Year'].values, data['Value'].values, color=colors[scen], linestyle=styles[scen], marker='x', label=scen)
            plt.fill_between(data['Year'].values, data['Lower_CI'].values, data['Upper_CI'].values, color=colors[scen], alpha=0.1)
    
    plt.title(f"Forecast: {indicator} (2025-2027)")
    plt.xlabel("Year")
    plt.ylabel("Percent (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100) # Percentage scale
    
    filename = f"reports/forecast_{indicator}.png"
    plt.savefig(filename)
    print(f"Saved plot to {filename}")

for ind in target_indicators:
    plot_forecast(final_df, ind)
