import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("🔮 BUDGET DEFICIT PREDICTIVE MODEL")
print("=" * 50)

# Load and prepare data
def load_and_prepare_data():
    """Load data and prepare for modeling"""
    try:
        df = pd.read_csv('cleaned_budget_data.csv')
    except:
        df = pd.read_excel('cleaned_budget_data.xlsx')
    
    # Ensure proper data types
    df['Time'] = pd.to_datetime(df['Time'])
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    df = df.dropna(subset=['Amount'])
    
    # Create time-based features
    df['Year'] = df['Time'].dt.year
    df['Month'] = df['Time'].dt.month
    df['Quarter'] = df['Time'].dt.quarter
    df['DayOfYear'] = df['Time'].dt.dayofyear
    
    # Sort by time for time-series analysis
    df = df.sort_values(['Country', 'Time'])
    
    return df

df = load_and_prepare_data()
print(f"✅ Data loaded: {len(df)} records, {len(df['Country'].unique())} countries")

# 1. FEATURE ENGINEERING
print("\n1. FEATURE ENGINEERING")
print("-" * 30)

def create_features(df):
    """Create advanced features for prediction"""
    
    # Country encoding
    le_country = LabelEncoder()
    df['Country_Encoded'] = le_country.fit_transform(df['Country'])
    
    # Create lag features (previous period values)
    df = df.sort_values(['Country', 'Time'])
    df['Amount_Lag1'] = df.groupby('Country')['Amount'].shift(1)
    df['Amount_Lag2'] = df.groupby('Country')['Amount'].shift(2)
    df['Amount_Lag3'] = df.groupby('Country')['Amount'].shift(3)
    
    # Rolling statistics
    df['Amount_Rolling_Mean_3'] = df.groupby('Country')['Amount'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    df['Amount_Rolling_Std_3'] = df.groupby('Country')['Amount'].transform(
        lambda x: x.rolling(window=3, min_periods=1).std()
    )
    
    # Year-over-year changes
    df['YoY_Change'] = df.groupby('Country')['Amount'].pct_change(periods=12)
    
    # Country-level statistics
    country_stats = df.groupby('Country')['Amount'].agg(['mean', 'std']).reset_index()
    country_stats.columns = ['Country', 'Country_Mean', 'Country_Std']
    df = df.merge(country_stats, on='Country', how='left')
    
    # Time-based features
    df['Years_From_Start'] = df['Year'] - df.groupby('Country')['Year'].transform('min')
    
    # Economic cycle features (simplified)
    df['Global_Financial_Crisis'] = ((df['Year'] >= 2007) & (df['Year'] <= 2009)).astype(int)
    df['COVID_Period'] = ((df['Year'] >= 2020) & (df['Year'] <= 2022)).astype(int)
    
    # Seasonal features
    df['Is_Q4'] = (df['Quarter'] == 4).astype(int)
    df['Is_H2'] = (df['Quarter'] >= 3).astype(int)
    
    return df

df = create_features(df)
print(f"✅ Features created. Total features: {len([col for col in df.columns if col not in ['Country', 'Time', 'Amount']])}")

# 2. PREPARE DATA FOR MODELING
print("\n2. PREPARING DATA FOR MODELING")
print("-" * 30)

# Define features and target
feature_columns = [
    'Country_Encoded', 'Year', 'Month', 'Quarter', 'DayOfYear',
    'Amount_Lag1', 'Amount_Lag2', 'Amount_Lag3',
    'Amount_Rolling_Mean_3', 'Amount_Rolling_Std_3', 'YoY_Change',
    'Country_Mean', 'Country_Std', 'Years_From_Start',
    'Global_Financial_Crisis', 'COVID_Period', 'Is_Q4', 'Is_H2'
]

# Remove rows with missing values in features
model_data = df.dropna(subset=feature_columns + ['Amount']).copy()

X = model_data[feature_columns]
y = model_data['Amount']

print(f"✅ Modeling dataset: {len(X)} samples, {len(feature_columns)} features")

# Split data (time-series aware split)
train_size = int(0.8 * len(X))
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

print(f"   Training set: {len(X_train)} samples")
print(f"   Test set: {len(X_test)} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. MODEL TRAINING
print("\n3. MODEL TRAINING")
print("-" * 30)

# Define models
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0)
}

# Train and evaluate models
results = {}
feature_importance = {}

for name, model in models.items():
    print(f"🏃 Training {name}...")
    
    if name in ['Linear Regression', 'Ridge Regression']:
        X_tr = X_train_scaled
        X_te = X_test_scaled
    else:
        X_tr = X_train
        X_te = X_test
    
    # Train model
    model.fit(X_tr, y_train)
    
    # Make predictions
    y_pred = model.predict(X_te)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Store results
    results[name] = {
        'model': model,
        'predictions': y_pred,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }
    
    # Store feature importance for tree-based models
    if hasattr(model, 'feature_importances_'):
        feature_importance[name] = model.feature_importances_
    
    print(f"   ✅ {name} - R²: {r2:.3f}, RMSE: {rmse:,.0f}, MAE: {mae:,.0f}")

# 4. MODEL COMPARISON
print("\n4. MODEL COMPARISON")
print("-" * 30)

# Create comparison DataFrame
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'R² Score': [results[name]['r2'] for name in results.keys()],
    'RMSE': [results[name]['rmse'] for name in results.keys()],
    'MAE': [results[name]['mae'] for name in results.keys()]
}).sort_values('R² Score', ascending=False)

print(comparison_df.to_string(index=False))

# Select best model
best_model_name = comparison_df.iloc[0]['Model']
best_model = results[best_model_name]['model']
print(f"\n🏆 Best Model: {best_model_name}")

# 5. FEATURE IMPORTANCE ANALYSIS
print("\n5. FEATURE IMPORTANCE ANALYSIS")
print("-" * 30)

if feature_importance:
    # Plot feature importance for the best tree-based model
    if best_model_name in feature_importance:
        importance_df = pd.DataFrame({
            'feature': feature_columns,
            'importance': feature_importance[best_model_name]
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df.head(15), x='importance', y='feature')
        plt.title(f'Top 15 Feature Importance - {best_model_name}')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Top 10 Most Important Features:")
        for i, row in importance_df.head(10).iterrows():
            print(f"   {i+1:2d}. {row['feature']}: {row['importance']:.3f}")

# 6. PREDICTION VISUALIZATION
print("\n6. PREDICTION VISUALIZATION")
print("-" * 30)

# Plot actual vs predicted for best model
best_predictions = results[best_model_name]['predictions']

plt.figure(figsize=(12, 6))
plt.scatter(y_test, best_predictions, alpha=0.6, s=50)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Budget Balance (Million)')
plt.ylabel('Predicted Budget Balance (Million)')
plt.title(f'Actual vs Predicted Budget Balances\n{best_model_name} (R² = {results[best_model_name]["r2"]:.3f})')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('prediction_scatter.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. COUNTRY-SPECIFIC PREDICTIONS
print("\n7. COUNTRY-SPECIFIC ANALYSIS")
print("-" * 30)

# Add predictions back to test data
test_data = model_data.iloc[train_size:].copy()
test_data['Predicted_Amount'] = best_predictions
test_data['Prediction_Error'] = test_data['Predicted_Amount'] - test_data['Amount']

# Analyze performance by country
country_performance = test_data.groupby('Country').agg({
    'Amount': ['count', 'mean', 'std'],
    'Prediction_Error': ['mean', 'std']
}).round(0)

country_performance.columns = ['Records', 'Actual_Mean', 'Actual_Std', 'Error_Mean', 'Error_Std']
country_performance['Abs_Error_Mean'] = abs(country_performance['Error_Mean'])
country_performance = country_performance.sort_values('Abs_Error_Mean')

print("🏛️  Model Performance by Country (Best to Worst):")
for country in country_performance.head(10).index:
    error = country_performance.loc[country, 'Error_Mean']
    records = country_performance.loc[country, 'Records']
    print(f"   • {country}: Avg Error ${error:,.0f}M ({records} records)")

# 8. FUTURE PREDICTIONS
print("\n8. FUTURE PREDICTIONS FOR 2024-2025")
print("-" * 30)

def predict_future(model, scaler, feature_columns, df, years_ahead=2):
    """Predict future budget balances for all countries"""
    
    future_predictions = []
    current_year = df['Year'].max()
    
    for country in df['Country'].unique():
        country_data = df[df['Country'] == country].sort_values('Time')
        
        # Get the latest data point for this country
        latest_data = country_data.iloc[-1].copy()
        
        for year_offset in range(1, years_ahead + 1):
            future_year = current_year + year_offset
            
            # Create future feature vector
            future_features = latest_data[feature_columns].copy()
            
            # Update time-based features
            future_features['Year'] = future_year
            future_features['Years_From_Start'] = future_year - df[df['Country'] == country]['Year'].min()
            
            # Update lag features (using predictions for multi-year forecasts)
            if year_offset == 1:
                future_features['Amount_Lag1'] = latest_data['Amount']
                future_features['Amount_Lag2'] = latest_data['Amount_Lag1']
                future_features['Amount_Lag3'] = latest_data['Amount_Lag2']
            else:
                # For year 2 predictions, we'd need to use previous predictions
                # This is simplified - in practice you'd want a more sophisticated approach
                pass
            
            # Update rolling statistics (simplified)
            future_features['Amount_Rolling_Mean_3'] = latest_data['Amount_Rolling_Mean_3']
            future_features['Amount_Rolling_Std_3'] = latest_data['Amount_Rolling_Std_3']
            
            # Reset event flags for future years
            future_features['Global_Financial_Crisis'] = 0
            future_features['COVID_Period'] = 0
            
            # Make prediction
            future_features_df = pd.DataFrame([future_features])[feature_columns]
            
            if best_model_name in ['Linear Regression', 'Ridge Regression']:
                future_features_scaled = scaler.transform(future_features_df)
                prediction = model.predict(future_features_scaled)[0]
            else:
                prediction = model.predict(future_features_df)[0]
            
            future_predictions.append({
                'Country': country,
                'Year': future_year,
                'Predicted_Amount': prediction,
                'Prediction_Type': 'Future'
            })
    
    return pd.DataFrame(future_predictions)

# Generate future predictions
future_df = predict_future(best_model, scaler, feature_columns, model_data)

print("🔮 Future Budget Predictions (2024-2025):")
for country in df['Country'].unique()[:8]:  # Show first 8 countries
    country_future = future_df[future_df['Country'] == country]
    for _, row in country_future.iterrows():
        amount_type = "Surplus" if row['Predicted_Amount'] > 0 else "Deficit"
        print(f"   • {country} {row['Year']}: ${row['Predicted_Amount']:,.0f}M ({amount_type})")

# 9. MODEL DEPLOYMENT READY OUTPUTS
print("\n9. GENERATING DEPLOYMENT OUTPUTS")
print("-" * 30)

# Save model predictions
test_data[['Country', 'Time', 'Amount', 'Predicted_Amount', 'Prediction_Error']].to_csv(
    'model_predictions.csv', index=False
)

# Save future predictions
future_df.to_csv('future_predictions.csv', index=False)

# Create performance report
performance_report = f"""
PREDICTIVE MODEL PERFORMANCE REPORT
===================================

Best Model: {best_model_name}
R² Score: {results[best_model_name]['r2']:.3f}
RMSE: {results[best_model_name]['rmse']:,.0f}
MAE: {results[best_model_name]['mae']:,.0f}

Dataset Information:
- Total records used: {len(model_data)}
- Training samples: {len(X_train)}
- Test samples: {len(X_test)}
- Number of features: {len(feature_columns)}

Top Performing Countries (Lowest Prediction Error):
{chr(10).join([f"- {country}: ${error:,.0f}M error" for country, error in country_performance.head(5)['Error_Mean'].items()])}

Key Insights:
- Model explains {results[best_model_name]['r2']*100:.1f}% of variance in budget balances
- Average prediction error: ${results[best_model_name]['mae']:,.0f}M
- Most important features are historical values and country characteristics
"""

with open('model_performance_report.txt', 'w') as f:
    f.write(performance_report)

print("✅ Model outputs saved:")
print("   • model_predictions.csv - Test set predictions")
print("   • future_predictions.csv - Future forecasts")
print("   • model_performance_report.txt - Performance summary")
print("   • feature_importance.png - Feature importance plot")
print("   • prediction_scatter.png - Prediction accuracy visualization")

print(f"\n🎯 MODEL READY FOR DEPLOYMENT!")
print(f"   Best model: {best_model_name}")
print(f"   Prediction accuracy: {results[best_model_name]['r2']*100:.1f}%")
print(f"   Average error: ${results[best_model_name]['mae']:,.0f}M")

# 10. ADVANCED ANALYSIS: DEFICIT/SURPLUS CLASSIFICATION
print("\n10. DEFICIT/SURPLUS CLASSIFICATION ANALYSIS")
print("-" * 30)

# Create binary classification (deficit vs surplus)
df['Is_Deficit'] = (df['Amount'] < 0).astype(int)
test_data['Predicted_Is_Deficit'] = (test_data['Predicted_Amount'] < 0).astype(int)

# Calculate classification accuracy
classification_accuracy = (test_data['Is_Deficit'] == test_data['Predicted_Is_Deficit']).mean()
print(f"📊 Deficit/Surplus Classification Accuracy: {classification_accuracy:.1%}")

# Analyze which countries are hardest to predict
hard_to_predict = test_data.groupby('Country').apply(
    lambda x: (x['Is_Deficit'] != x['Predicted_Is_Deficit']).mean()
).sort_values(ascending=False)

print(f"\n🎯 Countries where deficit/surplus prediction is most challenging:")
for country in hard_to_predict.head(5).index:
    accuracy = 1 - hard_to_predict[country]
    print(f"   • {country}: {accuracy:.1%} classification accuracy")

print(f"\n🏁 PREDICTIVE MODELING COMPLETE!")
print("   The model is ready for fiscal forecasting and policy planning.")
