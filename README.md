# Ethiopia Financial Inclusion Forecasting

Analyzing and forecasting financial inclusion indicators in Ethiopia (2011-2027).

## Features
- **Data Exploration**: Historical Findex and local data analysis.
- **Event Modeling**: Quantitative impact assessment of key events (e.g., Telebirr, Safaricom, FX reforms).
- **Forecasting**: Scenario-based projections (Base, Optimistic, Pessimistic) for 2025-2027.
- **Interactive Dashboard**: stakeholders can explore data, trends, and forecasts visually.

## Installation

1.  **Clone the repository**
2.  **Install dependencies**:
    ```bash
    pip install pandas openpyxl streamlit plotly nbformat scipy matplotlib seaborn
    ```
3.  **Run the Dashboard**:
    ```bash
    streamlit run dashboard/app.py
    ```

## Project Structure
- `data/raw/`: Enriched dataset in Excel format.
- `notebooks/`: Sequential process from EDA to Forecasting.
- `reports/`: Generated impact matrix, forecast results, and visualizations.
- `dashboard/`: Streamlit application code.
