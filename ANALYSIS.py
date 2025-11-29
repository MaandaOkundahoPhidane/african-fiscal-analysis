import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load and preprocess your cleaned data
def load_and_preprocess_data():
    """Load the cleaned dataset and ensure proper data types"""
    try:
        df = pd.read_csv('cleaned_budget_data.csv')
    except:
        df = pd.read_excel('cleaned_budget_data.xlsx')
    
    # Convert Amount to numeric, handling any non-numeric values
    print("Converting Amount column to numeric...")
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    
    # Check for any remaining non-numeric values
    if df['Amount'].isna().any():
        print(f"Warning: {df['Amount'].isna().sum()} non-numeric values found and converted to NaN")
        df = df.dropna(subset=['Amount'])
    
    # Ensure proper datetime format
    df['Time'] = pd.to_datetime(df['Time'])
    df['Year'] = df['Time'].dt.year
    df['Month'] = df['Time'].dt.month
    df['Quarter'] = df['Time'].dt.quarter
    
    return df

# Load the data
df = load_and_preprocess_data()
print("✅ Data loaded and preprocessed successfully!")
print(f"Dataset shape: {df.shape}")
print(f"Countries: {', '.join(df['Country'].unique())}")
print(f"Time range: {df['Time'].min().strftime('%Y-%m-%d')} to {df['Time'].max().strftime('%Y-%m-%d')}")
print(f"Total records: {len(df):,}")
print(f"Amount range: ${df['Amount'].min():,.0f} to ${df['Amount'].max():,.0f} million")

# 1. EXECUTIVE SUMMARY
print("\n" + "="*60)
print("1. EXECUTIVE SUMMARY")
print("="*60)

# Key metrics
total_deficit = df[df['Amount'] < 0]['Amount'].sum()
total_surplus = df[df['Amount'] > 0]['Amount'].sum()
deficit_percentage = (df['Amount'] < 0).mean() * 100
avg_deficit_size = df[df['Amount'] < 0]['Amount'].mean()
avg_surplus_size = df[df['Amount'] > 0]['Amount'].mean()

print(f"📊 OVERALL FISCAL HEALTH:")
print(f"   • Deficit Periods: {deficit_percentage:.1f}% of all records")
print(f"   • Total Cumulative Deficit: ${abs(total_deficit):,.0f} million")
print(f"   • Total Cumulative Surplus: ${total_surplus:,.0f} million")
print(f"   • Average Deficit Size: ${abs(avg_deficit_size):,.0f} million")
print(f"   • Average Surplus Size: ${avg_surplus_size:,.0f} million")

# 2. COUNTRY PERFORMANCE RANKINGS
print("\n" + "="*60)
print("2. COUNTRY PERFORMANCE RANKINGS")
print("="*60)

country_stats = df.groupby('Country').agg({
    'Amount': ['mean', 'min', 'max', 'std', 'count'],
    'Year': ['min', 'max']
}).round(0)

country_stats.columns = ['Avg_Balance', 'Worst_Balance', 'Best_Balance', 'Volatility', 'Records', 'Start_Year', 'End_Year']
country_stats['Deficit_Frequency'] = df[df['Amount'] < 0].groupby('Country').size() / df.groupby('Country').size() * 100
country_stats['Deficit_Frequency'] = country_stats['Deficit_Frequency'].fillna(0).round(1)

print("🏆 COUNTRY RANKINGS:")
print("\nA. By Average Balance (Best to Worst):")
ranking_balance = country_stats.sort_values('Avg_Balance', ascending=False)[['Avg_Balance', 'Deficit_Frequency']]
for country, row in ranking_balance.iterrows():
    balance_type = "Surplus" if row['Avg_Balance'] > 0 else "Deficit"
    print(f"   • {country}: ${row['Avg_Balance']:,.0f}M ({balance_type}) - Deficit {row['Deficit_Frequency']}% of time")

print("\nB. By Deficit Frequency (Most Chronic to Least):")
ranking_frequency = country_stats.sort_values('Deficit_Frequency', ascending=False)[['Deficit_Frequency', 'Avg_Balance']]
for country, row in ranking_frequency.iterrows():
    print(f"   • {country}: {row['Deficit_Frequency']}% deficit frequency - Avg: ${row['Avg_Balance']:,.0f}M")

print("\nC. By Volatility (Most Unstable to Most Stable):")
ranking_volatility = country_stats.sort_values('Volatility', ascending=False)[['Volatility', 'Avg_Balance']]
for country, row in ranking_volatility.iterrows():
    print(f"   • {country}: ${row['Volatility']:,.0f}M volatility - Avg: ${row['Avg_Balance']:,.0f}M")

# 3. TREND ANALYSIS
print("\n" + "="*60)
print("3. TREND ANALYSIS")
print("="*60)

# Yearly trends
yearly_data = df.groupby(['Country', 'Year'])['Amount'].mean().reset_index()

def analyze_country_trends(country_data):
    """Analyze trends for a specific country"""
    if len(country_data) < 3:
        return None
    
    # Calculate trends
    x = np.arange(len(country_data))
    y = country_data['Amount'].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # Recent vs historical
    recent_years = 3
    if len(country_data) >= recent_years:
        recent_avg = country_data.tail(recent_years)['Amount'].mean()
        historical_avg = country_data.head(len(country_data) - recent_years)['Amount'].mean()
        recent_vs_historical = recent_avg - historical_avg
    else:
        recent_vs_historical = 0
    
    return {
        'trend': 'improving' if slope > 0 else 'worsening',
        'trend_strength': abs(slope),
        'r_squared': r_value**2,
        'recent_vs_historical': recent_vs_historical,
        'current_status': 'deficit' if country_data.iloc[-1]['Amount'] < 0 else 'surplus'
    }

print("📈 TREND ANALYSIS BY COUNTRY:")
meaningful_trends = 0
for country in df['Country'].unique():
    country_yearly = yearly_data[yearly_data['Country'] == country].sort_values('Year')
    trend_info = analyze_country_trends(country_yearly)
    
    if trend_info and trend_info['r_squared'] > 0.1:  # Meaningful trend
        meaningful_trends += 1
        trend_icon = "📈" if trend_info['trend'] == 'improving' else "📉"
        status_icon = "🔴" if trend_info['current_status'] == 'deficit' else "🟢"
        change_icon = "⬆️" if trend_info['recent_vs_historical'] > 0 else "⬇️"
        
        print(f"   {trend_icon} {status_icon} {country}:")
        print(f"      Trend: {trend_info['trend'].upper()} (R²={trend_info['r_squared']:.2f})")
        print(f"      Recent vs Historical: {change_icon} ${trend_info['recent_vs_historical']:,.0f}M")
        print(f"      Current: {'Deficit' if trend_info['current_status'] == 'deficit' else 'Surplus'}")

print(f"\n   Found {meaningful_trends} countries with meaningful trends (R² > 0.1)")

# 4. COMPREHENSIVE VISUALIZATIONS
print("\n" + "="*60)
print("4. GENERATING COMPREHENSIVE VISUALIZATIONS")
print("="*60)

# Create a professional dashboard
fig = plt.figure(figsize=(20, 16))
fig.suptitle('FISCAL ANALYSIS DASHBOARD: Budget Deficits & Surpluses\nAfrican Countries Analysis', 
             fontsize=16, fontweight='bold', y=0.98)

# Define grid for subplots
gs = plt.GridSpec(3, 3, figure=fig)

# Plot 1: Time series trends (top, full width)
ax1 = fig.add_subplot(gs[0, :])
for country in df['Country'].unique():
    country_data = df[df['Country'] == country]
    yearly_avg = country_data.groupby('Year')['Amount'].mean()
    if len(yearly_avg) > 1:  # Only plot countries with multiple years
        line_style = '-' if country_data['Amount'].mean() >= 0 else '--'
        ax1.plot(yearly_avg.index, yearly_avg.values, marker='o', linewidth=2, 
                 label=country, linestyle=line_style, markersize=4)

ax1.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
ax1.set_title('Budget Balance Trends by Country (Yearly Averages)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Year', fontweight='bold')
ax1.set_ylabel('Budget Balance (Million USD)', fontweight='bold')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.ticklabel_format(style='plain', axis='y')

# Plot 2: Recent performance comparison
ax2 = fig.add_subplot(gs[1, 0])
recent_data = df[df['Year'] >= df['Year'].max() - 2]  # Last 3 years
if not recent_data.empty:
    recent_avg = recent_data.groupby('Country')['Amount'].mean().sort_values()
    colors = ['#d62728' if x < 0 else '#2ca02c' for x in recent_avg.values]
    bars = ax2.barh(recent_avg.index, recent_avg.values, color=colors, alpha=0.8)
    ax2.axvline(x=0, color='black', linestyle='-', alpha=0.8)
    ax2.set_title('Recent Performance\n(Last 3 Years Average)', fontweight='bold')
    ax2.set_xlabel('Average Balance (Million USD)', fontweight='bold')
    
    # Add value labels on bars
    for bar in bars:
        width = bar.get_width()
        label_pos = width if width >= 0 else width - (ax2.get_xlim()[1] * 0.01)
        ax2.text(label_pos, bar.get_y() + bar.get_height()/2, 
                 f'{width:,.0f}', ha='left' if width >= 0 else 'right', 
                 va='center', fontweight='bold', fontsize=8)

# Plot 3: Deficit frequency
ax3 = fig.add_subplot(gs[1, 1])
deficit_freq = df[df['Amount'] < 0].groupby('Country').size() / df.groupby('Country').size() * 100
deficit_freq = deficit_freq.sort_values(ascending=True)
colors = plt.cm.RdYlGn_r(1 - deficit_freq.values/100)
bars = ax3.barh(deficit_freq.index, deficit_freq.values, color=colors, alpha=0.8)
ax3.set_title('Deficit Frequency\n(% of Time in Deficit)', fontweight='bold')
ax3.set_xlabel('Percentage of Periods', fontweight='bold')
ax3.set_xlim(0, 100)

# Add percentage labels
for bar in bars:
    width = bar.get_width()
    ax3.text(width + 1, bar.get_y() + bar.get_height()/2, 
             f'{width:.1f}%', va='center', fontweight='bold', fontsize=8)

# Plot 4: Volatility analysis
ax4 = fig.add_subplot(gs[1, 2])
volatility = df.groupby('Country')['Amount'].std().sort_values(ascending=True)
colors = plt.cm.viridis(volatility.values / volatility.max())
bars = ax4.barh(volatility.index, volatility.values, color=colors, alpha=0.8)
ax4.set_title('Budget Volatility\n(Standard Deviation)', fontweight='bold')
ax4.set_xlabel('Volatility (Std Dev)', fontweight='bold')

# Add volatility values
for bar in bars:
    width = bar.get_width()
    ax4.text(width + ax4.get_xlim()[1] * 0.01, bar.get_y() + bar.get_height()/2, 
             f'{width:,.0f}', va='center', fontweight='bold', fontsize=8)

# Plot 5: Distribution analysis
ax5 = fig.add_subplot(gs[2, :])
plotted_countries = 0
for country in df['Country'].unique():
    country_data = df[df['Country'] == country]['Amount']
    if len(country_data) > 10:  # Only plot countries with sufficient data
        # Use kernel density estimate for smoother distributions
        from scipy.stats import gaussian_kde
        try:
            kde = gaussian_kde(country_data)
            x_range = np.linspace(country_data.min(), country_data.max(), 100)
            ax5.plot(x_range, kde(x_range), linewidth=2, label=country, alpha=0.7)
            plotted_countries += 1
        except:
            continue

ax5.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Balance Point')
ax5.set_title(f'Distribution of Budget Balances by Country ({plotted_countries} countries with sufficient data)', 
              fontsize=14, fontweight='bold')
ax5.set_xlabel('Budget Balance (Million USD)', fontweight='bold')
ax5.set_ylabel('Density', fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comprehensive_fiscal_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# 5. KEY INSIGHTS AND RECOMMENDATIONS
print("\n" + "="*60)
print("5. KEY INSIGHTS AND STRATEGIC RECOMMENDATIONS")
print("="*60)

# Categorize countries for recommendations
chronic_deficit = country_stats[country_stats['Deficit_Frequency'] > 75]
large_deficit = country_stats[country_stats['Avg_Balance'] < -50000]
improving_trends = []
deteriorating_trends = []

# Analyze trends for categorization
for country in df['Country'].unique():
    country_yearly = yearly_data[yearly_data['Country'] == country].sort_values('Year')
    trend_info = analyze_country_trends(country_yearly)
    if trend_info:
        if trend_info['trend'] == 'improving' and trend_info['r_squared'] > 0.3:
            improving_trends.append(country)
        elif trend_info['trend'] == 'worsening' and trend_info['r_squared'] > 0.3:
            deteriorating_trends.append(country)

print("🎯 STRATEGIC RECOMMENDATIONS BY CATEGORY:")

if not chronic_deficit.empty:
    print("\n🔴 CHRONIC DEFICIT COUNTRIES (Structural Reform Needed):")
    for country in chronic_deficit.index:
        print(f"   • {country}: Implement comprehensive tax reform and expenditure rationalization")

if not large_deficit.empty:
    print("\n🟡 LARGE DEFICIT COUNTRIES (Fiscal Consolidation Priority):")
    for country in large_deficit.index:
        print(f"   • {country}: Focus on debt sustainability and medium-term fiscal framework")

if improving_trends:
    print("\n🟢 IMPROVING TREND COUNTRIES (Maintain Momentum):")
    for country in improving_trends:
        print(f"   • {country}: Continue current policies and build fiscal buffers")

if deteriorating_trends:
    print("\n🔴 DETERIORATING TREND COUNTRIES (Urgent Action Needed):")
    for country in deteriorating_trends:
        print(f"   • {country}: Immediate fiscal review and corrective measures")

# 6. EXPORT RESULTS
print("\n" + "="*60)
print("6. EXPORTING ANALYSIS RESULTS")
print("="*60)

# Export detailed analysis
try:
    with pd.ExcelWriter('comprehensive_fiscal_analysis.xlsx') as writer:
        country_stats.to_excel(writer, sheet_name='Country_Statistics')
        
        # Yearly trends
        yearly_pivot = yearly_data.pivot(index='Year', columns='Country', values='Amount')
        yearly_pivot.to_excel(writer, sheet_name='Yearly_Trends')
        
        # Create executive summary
        summary_data = []
        for country in df['Country'].unique():
            country_data = df[df['Country'] == country]
            stats = country_stats.loc[country]
            summary_data.append({
                'Country': country,
                'Avg_Balance_Million': stats['Avg_Balance'],
                'Deficit_Frequency_Pct': stats['Deficit_Frequency'],
                'Volatility': stats['Volatility'],
                'Records': stats['Records'],
                'Period': f"{int(stats['Start_Year'])}-{int(stats['End_Year'])}",
                'Risk_Level': 'High' if stats['Deficit_Frequency'] > 70 else 'Medium' if stats['Deficit_Frequency'] > 40 else 'Low'
            })
        
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Executive_Summary', index=False)
    
    print("✅ ANALYSIS COMPLETE! Generated Files:")
    print("   📊 comprehensive_fiscal_analysis.png - Main dashboard visualization")
    print("   📈 comprehensive_fiscal_analysis.xlsx - Detailed data and analysis")
    print("   📋 Strategic recommendations provided above")
    
    print(f"\n💡 KEY FINDINGS SUMMARY:")
    print(f"   • Analyzed {len(df['Country'].unique())} African countries")
    print(f"   • {len(chronic_deficit)} countries have chronic deficit issues (>75% frequency)")
    print(f"   • {len(large_deficit)} countries face large deficit challenges (>$50M avg deficit)")
    print(f"   • Overall fiscal health: {deficit_percentage:.1f}% of periods in deficit")
    
except Exception as e:
    print(f"⚠️  Could not export Excel file: {e}")
    print("✅ Analysis completed successfully - results displayed above")

print("\n" + "="*60)
print("ANALYSIS COMPLETE - Ready for Policy Decision Making!")
print("="*60)
