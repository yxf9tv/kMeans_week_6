# NBA Player Value Analysis — K-Means Clustering

## Objective

The Washington Wizards' scouting department wants to identify **high-performing, underpaid NBA players** they can target to improve the roster. This project uses K-means clustering on 2024-25 season data to classify all NBA players into three value tiers — **Underpaid**, **Fair Value**, and **Overpaid** — and surface specific player recommendations in each category.

---

## Data Sources

| File | Contents |
|---|---|
| `nba_2025.txt` | Per-season player statistics (33 columns, ~470 players) |
| `2025_salaries.csv` | 2025-26 contract salaries for NBA players |

The two datasets are merged on player name. Players who were traded mid-season appear multiple times in the stats file; we resolve this by keeping the `TOT` (season total) row for traded players and dropping team-specific splits.

---

## Methodology

### 1. Data Cleaning

- **Salary parsing:** The salary column is stored as a formatted string (`$59,020,000`). It is stripped of `$` and commas and converted to a float.
- **Duplicate players:** Players traded mid-season appear once per team plus a `TOT` row. We keep only `TOT` rows for these players, then apply `drop_duplicates` for anyone without a `TOT` entry.
- **Zero-salary players:** Three players had `$0` salary after the merge (data anomalies from two-way or Exhibit-10 contracts). These are removed before analysis.
- **NaN filling:** Percentage columns (e.g., `3P%`) are `NaN` when a player has zero attempts. These are filled with `0`.
- **Dropped columns:** `Rk`, `Tm`, `Pos`, `Awards`, `Player-additional`, `Trp-Dbl`, `FG%`, `2P%`, `eFG%` are removed as non-informative or redundant.

**Final dataset:** 411 unique players.

---

### 2. Feature Selection & Correlation Analysis

We compute the Pearson correlation of every numeric column with `Salary` and print the top 15. A **correlation heatmap** (`heatmap_correlations.png`) shows the inter-feature relationships of the selected features.

**Why not just pick the top correlated features?**

The top correlates of salary are `FT`, `FTA`, `FG`, `2P`, `2PA`, `PTS` — but these are all shooting-volume stats that correlate **0.87–0.99 with each other**. Feeding K-means near-duplicate columns distorts cluster distances: the algorithm treats the same information as if it were six independent signals, pulling clusters toward high-volume scorers regardless of their other contributions.

Instead, we manually select **four semantically distinct stats** with lower inter-correlation:

| Feature | What it measures | Salary correlation |
|---|---|---|
| `PTS` | Scoring output | 0.61 |
| `AST` | Playmaking / creation | 0.46 |
| `TRB` | Rebounding / interior presence | 0.36 |
| `MP` | Minutes played (proxy for coach trust and availability) | 0.43 |

These four dimensions capture the key skills scouts evaluate and are meaningfully less collinear than raw shooting-breakdown columns.

---

### 3. Performance Score

We define a composite **Performance Score**:

```
PerfScore = PTS + AST + TRB
```

This is a single number representing a player's total season contribution across the three most impactful box-score dimensions. It is used in scatter plots and the value-ratio calculation throughout the project. It is computed before scaling so it is available at every stage.

---

### 4. Feature Scaling & Value Ratio

K-means uses Euclidean distance, so all features must be on the same scale. We apply `StandardScaler` (zero mean, unit variance) to the clustering features.

**The multicollinearity problem revisited — in clustering:**

When we clustered on `[PTS, AST, TRB, MP, Salary]` alone, all high-performing players clustered together regardless of salary. A player earning `$3M` with 1,400 `PerfScore` looked identical to a `$30M` player with the same stats in raw feature space — both would land in "Fair Value."

**Fix: add `ValueRatio` as a clustering feature.**

```python
ValueRatio = log1p(PerfScore / Salary)
```

The `log1p` transform dampens the extreme right skew caused by low-salary players with any meaningful production (e.g., a `$130K` player with 500 `PerfScore` would otherwise have an enormous ratio that overwhelms the scaler). With `ValueRatio` included, K-means has an explicit signal separating "high production, low pay" from "high production, high pay."

**Final clustering features:** `PTS`, `AST`, `TRB`, `MP`, `Salary`, `ValueRatio`

---

### 5. Initial K-Means (K = 3)

We begin with `K = 3` as an informed guess — we expect the data to contain three natural tiers: underpaid, fairly paid, and overpaid. The initial model is trained on the scaled features and used to produce the first scatter plot (`kmeans_initial.png`).

---

### 6. Elbow Method & Silhouette Analysis

We test `K = 2` through `K = 10`, recording **inertia** (within-cluster sum of squares) and **silhouette score** (a measure of cluster separation, ranging from -1 to 1) for each. Results are plotted side-by-side (`elbow_silhouette.png`).

The silhouette score peaks at `K = 2`, but two clusters cannot support three player recommendation categories. We enforce a **minimum of K = 3** to ensure we always have Underpaid, Fair Value, and Overpaid groups. `K = 3` has the second-highest silhouette score and is the natural choice given the problem structure.

---

### 7. Final Model & Cluster Naming

We retrain with the chosen `K` and assign **descriptive names** to clusters rather than numeric labels. Naming is done automatically by ranking clusters on their mean `PerfScore / Salary` ratio:

- **Highest ratio** → `Underpaid` (high production per dollar)
- **Lowest ratio** → `Overpaid` (low production per dollar)
- **Middle** → `Fair Value`

This mapping is recomputed after both the initial and final model in case cluster numbering shifts between runs.

**Final model quality:**
- Variance explained: ~56–62% (reasonable for 6 features)
- Silhouette score: ~0.37

The final scatter plot (`kmeans_final.png`) shows all players colored by cluster, with cluster center annotations and a **fair-value trend line** (OLS fit of `Salary ~ PerfScore`). Players above the trend line are underpaid relative to their production; players below are overpaid.

---

### 8. Player Recommendations

Within each cluster, players are ranked by **PerfPerM** (performance score per $1M salary):

```
PerfPerM = PerfScore / (Salary in $M)
```

The top 4 from each cluster are selected as recommendations.

**Sample results (2024-25 season):**

| Category | Player | Salary | PerfScore | PerfPerM |
|---|---|---|---|---|
| Good Choice | Russell Westbrook | $2.3M | 1,434 | 624 |
| Good Choice | Toumani Camara | $2.2M | 1,178 | 530 |
| Not Good Choice | Bradley Beal | $59M | 64 | 1.1 |
| Not Good Choice | Matisse Thybulle | $11.6M | 27 | 2.3 |
| Fallback | Dyson Daniels | $7.7M | 1,283 | 167 |
| Fallback | Amen Thompson | $9.7M | 1,592 | 164 |

---

## Output Files

| File | Description |
|---|---|
| `heatmap_correlations.png` | Correlation matrix of selected clustering features vs. Salary |
| `kmeans_initial.png` | Scatter plot of initial K=3 clusters (PerfScore vs. Salary) |
| `elbow_silhouette.png` | Elbow curve and silhouette scores for K=2–10 |
| `kmeans_final.png` | Final cluster scatter with fair-value trend line |

---

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python skills_practice.py
```

All charts are saved as `.png` files in the working directory. Player recommendations print to stdout.

---

## Key Design Decisions Summary

| Decision | Choice | Reason |
|---|---|---|
| Duplicate players | Keep `TOT` row | Complete season totals; avoids double-counting |
| Feature selection | Manual (PTS, AST, TRB, MP) | Avoids multicollinearity from redundant shooting stats |
| Value signal | `log1p(PerfScore / Salary)` as a feature | Forces K-means to separate high-perf/low-pay from high-perf/high-pay |
| Optimal K | K=3 (enforced minimum) | Two clusters cannot support three recommendation categories |
| Cluster naming | Automatic via `PerfScore / Salary` ratio | Deterministic; survives random seed changes |
| Salary scaling | `StandardScaler` | Required for Euclidean distance; better silhouette than MinMaxScaler |
