
import pandas as pd

file_path = 'data/raw/ethiopia_fi_unified_data.xlsx'
df = pd.read_excel(file_path, sheet_name='ethiopia_fi_unified_data')

print("Unique Record Types:", df['record_type'].unique())
print("Unique Indicator Codes:", df['indicator_code'].unique())

# Also print columns to be sure
print("Columns:", df.columns.tolist())

# Peek at rows that might be relevant
print("\nSample rows with 'Account' in indicator name:")
print(df[df['indicator'].fillna('').str.contains('Account', case=False)][['record_type', 'indicator', 'indicator_code']].head())
