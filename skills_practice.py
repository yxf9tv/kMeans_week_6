# %%
# ============================================================
# NBA Salary Clustering Analysis
# Goal: Identify underpaid, high-performing players for the Wizards
# ============================================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ============================================================
# SECTION 1: LOAD & CLEAN DATA
# ============================================================

# %%
# Load salary data — header is on row 1 (row 0 is a label), use latin-1 for special chars
salary_data = pd.read_csv('2025_salaries.csv', header=1, encoding='latin-1')

# Load per-season player stats
stats_data = pd.read_csv('nba_2025.txt', sep=",", encoding='latin-1')

# %%
# Merge on Player name — inner join keeps only players present in both datasets
merged_data = pd.merge(salary_data, stats_data, on='Player')

# Rename salary column for clarity
merged_data.rename(columns={'2025-26': 'Salary'}, inplace=True)

# %%
# Clean Salary: strip "$" and commas, then convert to numeric float
merged_data['Salary'] = (merged_data['Salary']
                         .str.replace('$', '', regex=False)
                         .str.replace(',', '', regex=False)
                         .astype(float))

# %%
# Handle duplicate players (traded mid-season appear multiple times)
# Strategy: for players with a 'TOT' (total) row, keep that; otherwise keep first appearance
tot_players = merged_data[merged_data['Tm'] == 'TOT']['Player'].unique()
non_tot = merged_data[
    ~((merged_data['Player'].isin(tot_players)) & (merged_data['Tm'] != 'TOT'))
]
merged_data = non_tot.drop_duplicates(subset='Player', keep='first').copy()

print(f"Players after deduplication: {len(merged_data)}")

# %%
# Drop columns that are not useful for clustering:
# - 'Rk': ranking index
# - 'Tm': team (categorical, not predictive of skill value)
# - 'Pos': position (categorical)
# - 'Awards': categorical text
# - 'Player-additional': duplicate identifier
# - 'Trp-Dbl': very sparse stat
# - Duplicate percentage columns (FG%, 2P%, eFG% are derivatives of raw counts)
drop_cols = ['Rk', 'Tm', 'Pos', 'Awards', 'Player-additional',
             'Trp-Dbl', 'FG%', '2P%', 'eFG%']
merged_data.drop(columns=[c for c in drop_cols if c in merged_data.columns], inplace=True)

# %%
# Fill NaN in percentage columns with 0
# (e.g., 3P% is NaN when a player has 0 three-point attempts)
merged_data.fillna(0, inplace=True)

# %%
# Remove players with zero or implausibly low salary — these are data anomalies
# (two-way / Exhibit-10 players whose salary didn't load correctly)
# NBA minimum for 2025-26 is ~$1.1M; we keep any player with a recorded salary
pre_filter = len(merged_data)
merged_data = merged_data[merged_data['Salary'] > 0].copy()
print(f"Removed {pre_filter - len(merged_data)} players with $0 salary")

print("Data shape after cleaning:", merged_data.shape)
print(merged_data.head())

# ============================================================
# SECTION 2: CORRELATION ANALYSIS & HEATMAP
# ============================================================

# %%
# Compute correlation of all numeric features with Salary
numeric_cols = merged_data.select_dtypes(include=np.number).columns.tolist()
correlations = merged_data[numeric_cols].corr()['Salary'].drop('Salary').sort_values(ascending=False)

print("\nTop features correlated with Salary:")
print(correlations.head(15))

# %%
# NOTE ON MULTICOLLINEARITY: Auto-selecting the top correlated features produces
# a redundant set (FT, FTA, FG, 2P, 2PA, PTS correlate 0.87-0.99 with each other).
# Feeding K-means near-duplicate columns distorts cluster distances.
# Instead we manually select four semantically distinct, low-overlap stats:
#   PTS  — scoring output (highest correlation with salary: 0.61)
#   AST  — playmaking / creation
#   TRB  — rebounding / interior presence
#   MP   — playing time (proxy for coach trust / availability)
# These capture the four key dimensions scouts use and are meaningfully
# less collinear than the raw shooting-breakdown columns.
# Takeaway: Sticking with the multicolinear top features (PTS, FT, FTA, FG, 2P, 2PA) would 
# have given us a cluster that basically just segments players by scoring volume and shooting attempts 
# — not a nuanced skill-based clustering. This is more practical and interpretable for the Wizards' needs.
cluster_features = ['PTS', 'AST', 'TRB', 'MP']

# Build heatmap over these four features + salary to confirm lower inter-feature correlation
heatmap_features = ['Salary'] + cluster_features
plt.figure(figsize=(8, 6))
sns.heatmap(
    merged_data[heatmap_features].corr(),
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    center=0,
    linewidths=0.5,
    annot_kws={'size': 11}
)
plt.title('Correlation Heatmap: Selected Features vs. Salary\n'
          '(chosen to minimise multicollinearity)',
          fontsize=13, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('heatmap_correlations.png', dpi=150)
plt.show()
print("Heatmap saved to heatmap_correlations.png")
print("\nFeatures selected for clustering:", cluster_features)

# %%
# Composite performance score: PTS + AST + TRB
# Computed here so it's available for all scatter plots and player categorization.
# This captures the three dimensions scouts care most about (scoring, creation, rebounding)
# and serves as the VALUE numerator when compared against Salary.
merged_data['PerfScore'] = merged_data['PTS'] + merged_data['AST'] + merged_data['TRB']

# ============================================================
# SECTION 3: FEATURE SCALING
# ============================================================

# %%
# Store player names separately before scaling (names are non-numeric)
player_names = merged_data['Player'].copy()

# Add explicit value ratio so K-means can separate "high perf / low salary" players
# from "high perf / high salary" players — without this, both groups look similar
# in raw feature space and collapse into the same cluster.
# np.log1p dampens the extreme skew from low-salary high-performers.
merged_data['ValueRatio'] = np.log1p(merged_data['PerfScore'] / merged_data['Salary'])

# Include Salary and ValueRatio in clustering alongside performance features
features_for_scaling = cluster_features + ['Salary', 'ValueRatio']

# StandardScaler: zero mean, unit variance — required for distance-based K-means
scaler = StandardScaler()
scaled_array = scaler.fit_transform(merged_data[features_for_scaling])
scaled_df = pd.DataFrame(scaled_array, columns=features_for_scaling, index=merged_data.index)

print("Scaled feature stats (should be ~0 mean, ~1 std):")
print(scaled_df.describe().loc[['mean', 'std']].round(2))

# ============================================================
# SECTION 4: INITIAL K-MEANS CLUSTERING (K=3)
# ============================================================

# %%
# Initial guess: K=3 (underpaid / fair / overpaid)
kmeans_init = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_init.fit(scaled_df)

# Attach cluster labels to the merged (unscaled) data for interpretation
merged_data = merged_data.copy()
merged_data['Cluster'] = kmeans_init.labels_

# Map numeric cluster IDs → descriptive names based on value ratio (PerfScore / Salary)
# Highest ratio = Underpaid, lowest = Overpaid, everything in between = Fair Value
_profile = merged_data.groupby('Cluster')[['PerfScore', 'Salary']].mean()
_profile['value_ratio'] = _profile['PerfScore'] / _profile['Salary']
_profile = _profile.sort_values('value_ratio', ascending=False)
_names = ['Underpaid'] + ['Fair Value'] * (len(_profile) - 2) + ['Overpaid']
cluster_name_map = dict(zip(_profile.index, _names))
merged_data['ClusterLabel'] = merged_data['Cluster'].map(cluster_name_map)

print("\nInitial K=3 cluster sizes:")
print(merged_data['ClusterLabel'].value_counts())

# %%
# Scatter plot: PerfScore vs Salary colored by named cluster
# PerfScore (PTS+AST+TRB) on the x-axis reflects production value directly —
# players to the upper-left of the trend line are underpaid targets.
label_colors = {'Underpaid': 'seagreen', 'Fair Value': 'steelblue', 'Overpaid': 'tomato'}
plt.figure(figsize=(10, 6))
for label in ['Underpaid', 'Fair Value', 'Overpaid']:
    if label not in merged_data['ClusterLabel'].values:
        continue
    mask = merged_data['ClusterLabel'] == label
    plt.scatter(
        merged_data.loc[mask, 'PerfScore'],
        merged_data.loc[mask, 'Salary'] / 1e6,
        label=label,
        alpha=0.7,
        color=label_colors[label]
    )
plt.xlabel('Performance Score (PTS + AST + TRB)', fontsize=12)
plt.ylabel('Salary ($ Millions)', fontsize=12)
plt.title('Initial K-Means (K=3): Performance Score vs. Salary', fontsize=13, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('kmeans_initial.png', dpi=150)
plt.show()

# ============================================================
# SECTION 5: ELBOW METHOD & SILHOUETTE ANALYSIS
# ============================================================

# %%
# Test K from 2 to 10 and record inertia and silhouette score
k_range = range(2, 11)
inertias = []
silhouette_scores = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(scaled_df)
    inertias.append(km.inertia_)
    silhouette_scores.append(silhouette_score(scaled_df, km.labels_))

# %%
# Plot elbow curve and silhouette scores side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Elbow curve
axes[0].plot(list(k_range), inertias, marker='o', color='steelblue', linewidth=2)
axes[0].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[0].set_ylabel('Inertia (Within-cluster Sum of Squares)', fontsize=11)
axes[0].set_title('Elbow Method', fontsize=13, fontweight='bold')
axes[0].set_xticks(list(k_range))
axes[0].grid(alpha=0.3)

# Silhouette scores
axes[1].plot(list(k_range), silhouette_scores, marker='s', color='tomato', linewidth=2)
axes[1].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[1].set_ylabel('Silhouette Score', fontsize=11)
axes[1].set_title('Silhouette Score by K', fontsize=13, fontweight='bold')
axes[1].set_xticks(list(k_range))
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('elbow_silhouette.png', dpi=150)
plt.show()

# %%
# Identify optimal K: highest silhouette score
# Enforce a minimum of 3 clusters so we always have three player categories
raw_optimal = list(k_range)[silhouette_scores.index(max(silhouette_scores))]
optimal_k = max(3, raw_optimal)
print(f"\nSilhouette scores by K: {dict(zip(k_range, [round(s,3) for s in silhouette_scores]))}")
print(f"Silhouette-optimal K: {raw_optimal}  →  Using K={optimal_k} (minimum 3 for three categories)")

# ============================================================
# SECTION 6: RETRAIN WITH OPTIMAL K & EVALUATE
# ============================================================

# %%
# Retrain K-means with the optimal K found above
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_final.fit(scaled_df)
merged_data['Cluster'] = kmeans_final.labels_

# Remap names after final model (cluster numbers may differ from initial run)
_profile = merged_data.groupby('Cluster')[['PerfScore', 'Salary']].mean()
_profile['value_ratio'] = _profile['PerfScore'] / _profile['Salary']
_profile = _profile.sort_values('value_ratio', ascending=False)
_names = ['Underpaid'] + ['Fair Value'] * (len(_profile) - 2) + ['Overpaid']
cluster_name_map = dict(zip(_profile.index, _names))
merged_data['ClusterLabel'] = merged_data['Cluster'].map(cluster_name_map)

print(f"\nFinal K={optimal_k} cluster sizes:")
print(merged_data['ClusterLabel'].value_counts())

# %%
# Evaluate clustering quality
# Variance explained: 1 - (inertia / total sum of squares)
total_ss = np.sum((scaled_df.values - scaled_df.values.mean(axis=0)) ** 2)
variance_explained = 1 - (kmeans_final.inertia_ / total_ss)
final_silhouette = silhouette_score(scaled_df, kmeans_final.labels_)

print(f"\nVariance Explained: {variance_explained:.3f} ({variance_explained*100:.1f}%)")
print(f"Silhouette Score:   {final_silhouette:.3f}")

# %%
# Scatter plot: PerfScore vs Salary with final clusters + fair-value trend line
# The trend line is a linear fit of Salary ~ PerfScore across all players.
# Players above the line are underpaid (high production, low pay) — targets.
# Players below the line are overpaid (low production, high pay) — avoid.
label_colors = {'Underpaid': 'seagreen', 'Fair Value': 'steelblue', 'Overpaid': 'tomato'}
cluster_means = merged_data.groupby('ClusterLabel')[['PerfScore', 'Salary']].mean()

fig, ax = plt.subplots(figsize=(12, 8))
for label in ['Underpaid', 'Fair Value', 'Overpaid']:
    if label not in merged_data['ClusterLabel'].values:
        continue
    mask = merged_data['ClusterLabel'] == label
    ax.scatter(
        merged_data.loc[mask, 'PerfScore'],
        merged_data.loc[mask, 'Salary'] / 1e6,
        label=label,
        alpha=0.65,
        color=label_colors[label],
        s=60
    )
    # Annotate cluster center with its name, perf score, and salary
    cx = cluster_means.loc[label, 'PerfScore']
    cy = cluster_means.loc[label, 'Salary'] / 1e6
    ax.annotate(f'{label}\n(perf={cx:.0f}, ${cy:.1f}M)',
                xy=(cx, cy), fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

# Fair-value trend line: OLS fit of Salary ~ PerfScore
# Above this line = underpaid; below = overpaid
trend_x = np.linspace(merged_data['PerfScore'].min(), merged_data['PerfScore'].max(), 200)
slope, intercept = np.polyfit(merged_data['PerfScore'], merged_data['Salary'] / 1e6, 1)
ax.plot(trend_x, slope * trend_x + intercept,
        color='black', linewidth=1.5, linestyle='--', label='Fair-value trend')
ax.text(trend_x[-1] * 0.95, slope * trend_x[-1] + intercept + 1,
        'Fair value', fontsize=9, color='black')

ax.set_xlabel('Performance Score (PTS + AST + TRB)', fontsize=12)
ax.set_ylabel('Salary ($ Millions)', fontsize=12)
ax.set_title(f'Final K-Means (K={optimal_k}): Performance Score vs. Salary\n'
             'Players above trend line = underpaid (targets)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('kmeans_final.png', dpi=150)
plt.show()

# ============================================================
# SECTION 7: PLAYER CATEGORIZATION
# ============================================================

# %%
# Summarise each named cluster to confirm assignments
# (ClusterLabel was set in Section 6 based on value ratio)
cluster_profile = merged_data.groupby('ClusterLabel').agg(
    mean_salary=('Salary', 'mean'),
    mean_perf=('PerfScore', 'mean'),
    count=('Player', 'count')
).reset_index()
cluster_profile['value_ratio'] = cluster_profile['mean_perf'] / cluster_profile['mean_salary']

print("\nCluster Profiles:")
print(cluster_profile.to_string(index=False))

# %%
# Good Choices: Underpaid cluster — sort by value ratio (perf / salary), top 10
good_df = merged_data[merged_data['ClusterLabel'] == 'Underpaid'].copy()
good_df['ValueRatio'] = good_df['PerfScore'] / good_df['Salary']
good_choices = good_df.nlargest(10, 'ValueRatio')[['Player', 'Salary', 'PTS', 'AST', 'TRB', 'PerfScore', 'ValueRatio']]

# Not Good Choices: Overpaid cluster — sort by salary per unit performance, top 10
bad_df = merged_data[merged_data['ClusterLabel'] == 'Overpaid'].copy()
bad_df['SalaryPerPerf'] = bad_df['Salary'] / (bad_df['PerfScore'] + 1)  # +1 avoids div-by-zero
not_good = bad_df.nlargest(10, 'SalaryPerPerf')[['Player', 'Salary', 'PTS', 'AST', 'TRB', 'PerfScore', 'SalaryPerPerf']]

# Fallback Options: Fair Value cluster — sort by value ratio, top 10
fallback_df = merged_data[merged_data['ClusterLabel'] == 'Fair Value'].copy()
fallback_df['ValueRatio'] = fallback_df['PerfScore'] / fallback_df['Salary']
fallbacks = fallback_df.nlargest(10, 'ValueRatio')[['Player', 'Salary', 'PTS', 'AST', 'TRB', 'PerfScore', 'ValueRatio']]

# %%
# Print results
print("\n" + "="*65)
print("PLAYER RECOMMENDATIONS FOR THE WIZARDS")
print("="*65)

print("\n--- GOOD CHOICES (Underpaid / High Value) — Top 10 ---")
good_choices['Salary_M'] = (good_choices['Salary'] / 1e6).round(2)
good_choices['PerfPerM'] = (good_choices['PerfScore'] / (good_choices['Salary'] / 1e6)).round(1)
print(good_choices[['Player', 'Salary_M', 'PTS', 'AST', 'TRB', 'PerfScore', 'PerfPerM']].to_string(index=False))
print("  (PerfPerM = performance score per $1M salary — higher is better value)")

print("\n--- NOT GOOD CHOICES (Overpaid / Low Value) — Top 10 ---")
not_good['Salary_M'] = (not_good['Salary'] / 1e6).round(2)
not_good['PerfPerM'] = (not_good['PerfScore'] / (not_good['Salary'] / 1e6 + 0.001)).round(1)
print(not_good[['Player', 'Salary_M', 'PTS', 'AST', 'TRB', 'PerfScore', 'PerfPerM']].to_string(index=False))
print("  (PerfPerM = performance score per $1M salary — lower means worse value)")

print("\n--- FALLBACK OPTIONS (Moderate Value) — Top 10 ---")
fallbacks['Salary_M'] = (fallbacks['Salary'] / 1e6).round(2)
fallbacks['PerfPerM'] = (fallbacks['PerfScore'] / (fallbacks['Salary'] / 1e6)).round(1)
print(fallbacks[['Player', 'Salary_M', 'PTS', 'AST', 'TRB', 'PerfScore', 'PerfPerM']].to_string(index=False))
print("  (PerfPerM = performance score per $1M salary)")

# %%
# Save all players in each cluster to a CSV for full review
# Each cluster gets its own sheet in an Excel file, and a flat CSV is also saved
output_cols = ['Player', 'Salary', 'PTS', 'AST', 'TRB', 'PerfScore', 'ClusterLabel']

# Flat CSV: all players with their cluster label, sorted by cluster then value ratio
all_players_out = merged_data[output_cols].copy()
all_players_out['Salary_M'] = (all_players_out['Salary'] / 1e6).round(2)
all_players_out['PerfPerM'] = (
    all_players_out['PerfScore'] / (all_players_out['Salary_M'] + 0.001)
).round(1)
all_players_out = all_players_out.sort_values(
    ['ClusterLabel', 'PerfPerM'], ascending=[True, False]
).drop(columns=['Salary'])
all_players_out.to_csv('cluster_all_players.csv', index=False)

# Excel: one sheet per cluster
with pd.ExcelWriter('cluster_all_players.xlsx', engine='openpyxl') as writer:
    for label in ['Underpaid', 'Fair Value', 'Overpaid']:
        sheet_df = all_players_out[all_players_out['ClusterLabel'] == label].copy()
        sheet_df.to_excel(writer, sheet_name=label, index=False)

print("\n" + "="*65)
print("Full cluster tables saved:")
print("  cluster_all_players.csv  — all players, flat file")
print("  cluster_all_players.xlsx — one sheet per cluster")
print("Charts saved: heatmap_correlations.png, kmeans_initial.png,")
print("              elbow_silhouette.png, kmeans_final.png")
print("="*65)
