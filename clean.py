import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_excel('dataset.xlsx', sheet_name='Data')

# Convert Time to datetime and sort properly
df['Time'] = pd.to_datetime(df['Time'])
df = df.sort_values(['Country', 'Time'])

print(f"Original data shape: {df.shape}")
print(f"Missing values before cleaning: {df['Amount'].isna().sum()}")

# Strategy: Multi-step imputation for maximum accuracy
def advanced_imputation(group):
    """Advanced imputation strategy for time-series fiscal data"""
    amount_series = group['Amount']
    
    # Step 1: Try linear interpolation (best for time-series)
    interpolated = amount_series.interpolate(method='linear', limit_direction='both')
    
    # Step 2: For any remaining gaps, use forward/backward fill
    filled = interpolated.ffill().bfill()
    
    # Step 3: If still missing, use rolling mean of same period
    if filled.isna().any():
        # Use 3-period rolling mean within the same country
        rolling_mean = amount_series.rolling(window=3, min_periods=1).mean()
        filled = filled.fillna(rolling_mean)
    
    group['Amount'] = filled
    return group

# Apply the advanced imputation
df = df.groupby('Country', group_keys=False).apply(advanced_imputation)

print(f"Missing values after cleaning: {df['Amount'].isna().sum()}")
print(f"Final data shape: {df.shape}")

# Additional cleaning steps
df = df.dropna(subset=['Amount'])  # Remove any stubborn missing values
df = df.drop_duplicates()  # Remove duplicates

# Standardize units (convert billions to millions)
df.loc[df['Unit'].str.lower() == 'billion', 'Amount'] *= 1000
df.loc[df['Unit'].str.lower() == 'billion', 'Unit'] = 'Million'

# Save cleaned dataset
df.to_csv('cleaned_budget_data.csv', index=False)
df.to_excel('cleaned_budget_data.xlsx', index=False)

print("✅ Cleaning complete! Files saved:")
print("- cleaned_budget_data.csv")
print("- cleaned_budget_data.xlsx")
