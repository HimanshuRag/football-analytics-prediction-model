#!/usr/bin/env python3
"""
Football Analytics - Initial Data Analysis
Author: Himanshu Raghav
Date: December 2025

This script performs initial exploratory data analysis on football match data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 60)
print("Football Analytics - Initial Analysis")
print("=" * 60)
print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*60)

# ==========================================
# SECTION 1: SAMPLE DATA CREATION
# ==========================================
print("\n[1] Creating Sample Dataset...")

# Create sample data for demonstration
np.random.seed(42)
teams = ['Arsenal', 'Chelsea', 'Liverpool', 'Man City', 'Man United', 
         'Tottenham', 'Newcastle', 'Brighton', 'Aston Villa', 'West Ham']

data = {
    'Team': teams,
    'Matches_Played': [20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
    'Wins': [14, 12, 15, 16, 10, 11, 9, 8, 7, 6],
    'Draws': [3, 4, 2, 2, 6, 5, 7, 8, 9, 8],
    'Losses': [3, 4, 3, 2, 4, 4, 4, 4, 4, 6],
    'Goals_Scored': [45, 38, 48, 52, 35, 38, 32, 30, 28, 25],
    'Goals_Conceded': [18, 22, 20, 15, 25, 23, 26, 28, 30, 35],
    'Home_Wins': [9, 8, 10, 10, 6, 7, 6, 5, 4, 4],
    'Away_Wins': [5, 4, 5, 6, 4, 4, 3, 3, 3, 2]
}

df = pd.DataFrame(data)

# Calculate additional metrics
df['Points'] = (df['Wins'] * 3) + df['Draws']
df['Goal_Difference'] = df['Goals_Scored'] - df['Goals_Conceded']
df['Win_Percentage'] = (df['Wins'] / df['Matches_Played'] * 100).round(2)
df['Goals_Per_Match'] = (df['Goals_Scored'] / df['Matches_Played']).round(2)
df['Points_Per_Match'] = (df['Points'] / df['Matches_Played']).round(2)

print(f"✓ Dataset created with {len(df)} teams")
print(f"✓ Total features: {len(df.columns)}")

# ==========================================
# SECTION 2: DATA EXPLORATION
# ==========================================
print("\n[2] Data Overview...")
print("\nDataset Shape:", df.shape)
print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe().round(2))

# ==========================================
# SECTION 3: KEY INSIGHTS
# ==========================================
print("\n[3] Key Performance Insights...")

# Top performers
top_team = df.nlargest(1, 'Points')['Team'].values[0]
top_scorer = df.nlargest(1, 'Goals_Scored')['Team'].values[0]
best_defense = df.nsmallest(1, 'Goals_Conceded')['Team'].values[0]

print(f"\n🏆 League Leader: {top_team} ({df[df['Team']==top_team]['Points'].values[0]} points)")
print(f"⚽ Top Scorer: {top_scorer} ({df[df['Team']==top_scorer]['Goals_Scored'].values[0]} goals)")
print(f"🛡️ Best Defense: {best_defense} ({df[df['Team']==best_defense]['Goals_Conceded'].values[0]} goals conceded)")

# Performance metrics
avg_goals = df['Goals_Scored'].mean()
avg_win_pct = df['Win_Percentage'].mean()
home_advantage = ((df['Home_Wins'].sum() / df['Wins'].sum()) * 100)

print(f"\n📊 League Averages:")
print(f"   • Average Goals per Team: {avg_goals:.1f}")
print(f"   • Average Win Percentage: {avg_win_pct:.1f}%")
print(f"   • Home Win Percentage: {home_advantage:.1f}%")

# ==========================================
# SECTION 4: CORRELATION ANALYSIS
# ==========================================
print("\n[4] Correlation Analysis...")

corr_features = ['Wins', 'Goals_Scored', 'Goals_Conceded', 'Goal_Difference', 'Points']
corr_matrix = df[corr_features].corr()

print("\nCorrelation Matrix:")
print(corr_matrix.round(3))

# Key correlations
print("\n🔍 Strong Correlations Found:")
print(f"   • Points vs Wins: {corr_matrix.loc['Points', 'Wins']:.3f}")
print(f"   • Points vs Goal Difference: {corr_matrix.loc['Points', 'Goal_Difference']:.3f}")
print(f"   • Goals Scored vs Points: {corr_matrix.loc['Goals_Scored', 'Points']:.3f}")

# ==========================================
# SECTION 5: VISUALIZATIONS
# ==========================================
print("\n[5] Creating Visualizations...")

# Create figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Football Analytics Dashboard - Premier League 2024/25', fontsize=16, fontweight='bold')

# Plot 1: Team Points
ax1 = axes[0, 0]
df_sorted = df.sort_values('Points', ascending=False)
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(df_sorted)))
ax1.barh(df_sorted['Team'], df_sorted['Points'], color=colors)
ax1.set_xlabel('Points', fontsize=12, fontweight='bold')
ax1.set_ylabel('Team', fontsize=12, fontweight='bold')
ax1.set_title('League Table - Points Distribution', fontsize=14, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# Plot 2: Goals Scored vs Conceded
ax2 = axes[0, 1]
scatter = ax2.scatter(df['Goals_Scored'], df['Goals_Conceded'], 
                     s=df['Points']*10, alpha=0.6, c=df['Points'], cmap='RdYlGn')
for idx, row in df.iterrows():
    ax2.annotate(row['Team'], (row['Goals_Scored'], row['Goals_Conceded']), 
                fontsize=8, alpha=0.7)
ax2.set_xlabel('Goals Scored', fontsize=12, fontweight='bold')
ax2.set_ylabel('Goals Conceded', fontsize=12, fontweight='bold')
ax2.set_title('Offensive vs Defensive Performance', fontsize=14, fontweight='bold')
ax2.grid(alpha=0.3)
plt.colorbar(scatter, ax=ax2, label='Points')

# Plot 3: Win Percentage Distribution
ax3 = axes[1, 0]
ax3.hist(df['Win_Percentage'], bins=8, color='steelblue', edgecolor='black', alpha=0.7)
ax3.axvline(df['Win_Percentage'].mean(), color='red', linestyle='--', 
           linewidth=2, label=f'Mean: {df["Win_Percentage"].mean():.1f}%')
ax3.set_xlabel('Win Percentage (%)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Number of Teams', fontsize=12, fontweight='bold')
ax3.set_title('Win Percentage Distribution', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# Plot 4: Home vs Away Performance
ax4 = axes[1, 1]
x = np.arange(len(df))
width = 0.35
ax4.bar(x - width/2, df['Home_Wins'], width, label='Home Wins', color='forestgreen', alpha=0.8)
ax4.bar(x + width/2, df['Away_Wins'], width, label='Away Wins', color='crimson', alpha=0.8)
ax4.set_xlabel('Team', fontsize=12, fontweight='bold')
ax4.set_ylabel('Number of Wins', fontsize=12, fontweight='bold')
ax4.set_title('Home vs Away Win Comparison', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(df['Team'], rotation=45, ha='right')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
print("✓ Dashboard created successfully")

# Save figure
try:
    plt.savefig('../visualizations/initial_analysis_dashboard.png', dpi=300, bbox_inches='tight')
    print("✓ Dashboard saved to: visualizations/initial_analysis_dashboard.png")
except:
    print("ℹ️ Could not save figure (folder may not exist yet)")

plt.show()

# ==========================================
# SECTION 6: EXPORT INSIGHTS
# ==========================================
print("\n[6] Exporting Results...")

# Create insights summary
insights_summary = f"""
FOOTBALL ANALYTICS - KEY INSIGHTS SUMMARY
==========================================
Analysis Date: {datetime.now().strftime('%Y-%m-%d')}

TOP PERFORMERS:
• League Leader: {top_team}
• Top Scorer: {top_scorer}
• Best Defense: {best_defense}

LEAGUE STATISTICS:
• Average Goals per Team: {avg_goals:.1f}
• Average Win Percentage: {avg_win_pct:.1f}%
• Home Advantage: {home_advantage:.1f}% of wins are at home

CORRELATION INSIGHTS:
• Strong positive correlation between Points and Wins (r={corr_matrix.loc['Points', 'Wins']:.3f})
• Strong positive correlation between Goal Difference and Points (r={corr_matrix.loc['Points', 'Goal_Difference']:.3f})
• Goals Scored correlates with Points (r={corr_matrix.loc['Goals_Scored', 'Points']:.3f})

KEY FINDINGS:
1. Teams with better goal difference tend to have more points
2. Home advantage is significant ({home_advantage:.1f}% of wins)
3. Defensive performance is as important as offensive output
4. Top teams maintain consistency in both home and away matches

NEXT STEPS:
- Collect real match data from football-data.co.uk
- Build predictive models for match outcomes
- Analyze betting odds correlation
- Create interactive dashboard with Streamlit
"""

print(insights_summary)

# Try to save insights
try:
    with open('../data/processed/insights_summary.txt', 'w') as f:
        f.write(insights_summary)
    print("\n✓ Insights saved to: data/processed/insights_summary.txt")
except:
    print("\nℹ️ Could not save insights file (folder may not exist yet)")

print("\n" + "="*60)
print("Analysis Complete!")
print("="*60)
print("\n📊 Next Steps:")
print("   1. Download real data from football-data.co.uk")
print("   2. Replace sample data with actual match results")
print("   3. Expand analysis with more advanced metrics")
print("   4. Build machine learning prediction models")
print("\n🔗 Project Repository: https://github.com/HimanshuRag/football-analytics-prediction-model")
print("="*60)
