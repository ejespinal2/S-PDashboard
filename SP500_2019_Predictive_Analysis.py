"""
SP500_2019_Predictive_Analysis.py
-----------------------------------
Uses 2015-2018 S&P 500 stock data to build a predictive model for 2019
and evaluates how "foreseeable" the 2019 rally actually was.

Narrative hypothesis
────────────────────
2019 was the S&P 500's best year since 2013.  The thesis here is that
while individual stock magnitudes are hard to forecast, *which stocks and
sectors would outperform* in 2019 was discernible from three structural
signals visible at the end of 2018:

  1. Mean-reversion pressure – stocks that fell hardest in 2018's Q4
     sell-off had the most room to recover once the Fed pivoted.

  2. Volatility regime – high 2018 volatility accompanied compressed
     valuations; stocks with elevated σ but solid multi-year track records
     historically rebound in risk-on environments.

  3. Multi-year momentum vs. reversion – 3-year momentum (2016-2018) was
     a stronger predictor than single-year momentum, capturing the
     Technology sector's sustained outperformance.

Outputs
───────
  • sp500_2019_predictions.xlsx  – model predictions vs actuals for every stock
  • figures/                     – 4 publication-quality charts for Tableau import
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score


# ── Config ─────────────────────────────────────────────────────────────────────
YEARLY_FILE   = "sp500_yearly_performance.xlsx"   # from SP500_Stock_Return_Data.py
FALLBACK_FILE = "Yearly_Performance.xlsx"            # original file as fallback

PALETTE = {
    "Communication Services": "#4E79A7",
    "Consumer Discretionary":  "#F28E2B",
    "Consumer Staples":        "#E15759",
    "Energy":                  "#76B7B2",
    "Financials":              "#59A14F",
    "Health Care":             "#EDC948",
    "Industrials":             "#B07AA1",
    "Information Technology":  "#FF9DA7",
    "Materials":               "#9C755F",
    "Real Estate":             "#BAB0AC",
    "Utilities":               "#D37295",
    "Unknown":                 "#aaaaaa",
}

Path("figures").mkdir(exist_ok=True)


# ── Load data ──────────────────────────────────────────────────────────────────
try:
    df = pd.read_excel(YEARLY_FILE)
    print(f"Loaded {YEARLY_FILE}")
except FileNotFoundError:
    df = pd.read_excel(FALLBACK_FILE)
    df["Sector"] = "Unknown"           # original file has no sector column
    print(f"Loaded fallback {FALLBACK_FILE} (no Sector column)")

df = df.sort_values(["Stock", "Year"]).reset_index(drop=True)

# Require at least 4 years of history per stock
stock_counts = df.groupby("Stock")["Year"].count()
valid_stocks = stock_counts[stock_counts >= 4].index
df = df[df["Stock"].isin(valid_stocks)].copy()
print(f"Stocks with ≥4 years of data: {len(valid_stocks)}")


# ── Feature engineering ────────────────────────────────────────────────────────
g = df.groupby("Stock")

df["Prev_Return"]     = g["Yearly Return"].shift(1)
df["Prev2_Return"]    = g["Yearly Return"].shift(2)
df["Prev3_Return"]    = g["Yearly Return"].shift(3)
df["Prev_Vol"]        = g["Yearly Volatility"].shift(1)
df["Prev2_Vol"]       = g["Yearly Volatility"].shift(2)

# Core signals
df["Momentum_1yr"]    = df["Prev_Return"]
df["Momentum_3yr"]    = (df["Prev_Return"] + df["Prev2_Return"] + df["Prev3_Return"]) / 3
df["MeanRev_Signal"]  = -df["Prev_Return"]          # contrarian: expect bounce after loss
df["Sharpe_Proxy"]    = df["Prev_Return"] / (df["Prev_Vol"] + 1e-6)
df["Vol_Change"]      = df["Prev_Vol"] - df["Prev2_Vol"]
df["Vol_Regime"]      = (df["Prev_Vol"] > df["Prev_Vol"].median()).astype(int)

FEATURES = [
    "Momentum_1yr",
    "Momentum_3yr",
    "MeanRev_Signal",
    "Sharpe_Proxy",
    "Vol_Change",
    "Prev_Vol",
]

# ── Train / predict 2019 ───────────────────────────────────────────────────────
model_data = df.dropna(subset=FEATURES + ["Yearly Return"])

# Train on 2016-2018, test on 2019
train = model_data[model_data["Year"].isin([2016, 2017, 2018])]
test  = model_data[model_data["Year"] == 2019]

X_train, y_train = train[FEATURES], train["Yearly Return"]
X_test,  y_test  = test[FEATURES],  test["Yearly Return"]

scaler  = StandardScaler()
Xtr_s   = scaler.fit_transform(X_train)
Xte_s   = scaler.transform(X_test)

# Ridge regression (interpretable, avoids overfitting)
ridge = Ridge(alpha=5.0)
ridge.fit(Xtr_s, y_train)
y_pred_ridge = ridge.predict(Xte_s)

# Gradient Boosting (captures non-linear relationships)
gbm = GradientBoostingRegressor(
    n_estimators=300, max_depth=3, learning_rate=0.05,
    subsample=0.8, random_state=42
)
gbm.fit(X_train, y_train)
y_pred_gbm = gbm.predict(X_test)

# CV R² on training set (honest estimate)
cv_r2 = cross_val_score(gbm, X_train, y_train, cv=5, scoring="r2").mean()

results = test[["Stock", "Sector", "Yearly Return"]].copy()
results["Ridge_Pred"]  = y_pred_ridge
results["GBM_Pred"]    = y_pred_gbm
results["Ridge_Error"] = (results["Yearly Return"] - results["Ridge_Pred"]).abs()
results["GBM_Error"]   = (results["Yearly Return"] - results["GBM_Pred"]).abs()

print("\n── 2019 Out-of-Sample Model Performance ─────────────────")
print(f"  Ridge  R²: {r2_score(y_test, y_pred_ridge):.3f}   MAE: {mean_absolute_error(y_test, y_pred_ridge):.3f}")
print(f"  GBM    R²: {r2_score(y_test, y_pred_gbm):.3f}   MAE: {mean_absolute_error(y_test, y_pred_gbm):.3f}")
print(f"  GBM 5-fold CV R² (train): {cv_r2:.3f}")
print(f"  Baseline (predict mean): R² = 0.000, MAE = {y_test.std():.3f}")

feat_imp = pd.Series(gbm.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\n── GBM Feature Importances ────────────────────────────")
print(feat_imp.to_string())


# ── Sector-level 2019 summary ─────────────────────────────────────────────────
sector_2019 = (
    results.groupby("Sector")
    .agg(
        Avg_Actual=("Yearly Return", "mean"),
        Avg_Predicted=("GBM_Pred", "mean"),
        Stock_Count=("Stock", "count"),
    )
    .round(3)
    .sort_values("Avg_Actual", ascending=False)
)
print("\n── 2019 Sector Performance vs Predictions ──────────────")
print(sector_2019.to_string())


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 – Predicted vs Actual 2019 returns (scatter, coloured by sector)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 7))
for sector, grp in results.groupby("Sector"):
    color = PALETTE.get(sector, "#888888")
    ax.scatter(grp["GBM_Pred"], grp["Yearly Return"],
               color=color, alpha=0.65, s=28, label=sector, linewidths=0)

lims = [
    min(results["GBM_Pred"].min(), results["Yearly Return"].min()) - 0.05,
    max(results["GBM_Pred"].max(), results["Yearly Return"].max()) + 0.05,
]
ax.plot(lims, lims, "k--", lw=1, alpha=0.5, label="Perfect prediction")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax.set_xlabel("Model-Predicted 2019 Return", fontsize=12)
ax.set_ylabel("Actual 2019 Return", fontsize=12)
ax.set_title("2019 Predicted vs Actual Returns (Gradient Boosting)\nTrained on 2016–2018 Data", fontsize=13)
r2 = r2_score(y_test, y_pred_gbm)
ax.text(0.04, 0.93, f"R² = {r2:.3f}", transform=ax.transAxes,
        fontsize=11, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
handles, labels = ax.get_legend_handles_labels()
unique = dict(zip(labels, handles))
ax.legend(unique.values(), unique.keys(), fontsize=7.5, loc="lower right",
          title="Sector", title_fontsize=8)
plt.tight_layout()
plt.savefig("figures/fig1_predicted_vs_actual_2019.png", dpi=150)
plt.close()
print("\nSaved figures/fig1_predicted_vs_actual_2019.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 – Feature importances: what made 2019 predictable?
# ═══════════════════════════════════════════════════════════════════════════════
FEAT_LABELS = {
    "Momentum_1yr":   "1-Year Momentum\n(2018 Return)",
    "Momentum_3yr":   "3-Year Momentum\n(2016-2018 Avg)",
    "MeanRev_Signal": "Mean-Reversion\nSignal (−2018 Ret)",
    "Sharpe_Proxy":   "Risk-Adj Return\n(2018 Sharpe)",
    "Vol_Change":     "Volatility Change\n(2017→2018 Δσ)",
    "Prev_Vol":       "Prior Volatility\n(2018 σ)",
}
fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#2166AC" if x > 0.15 else "#92C5DE" for x in feat_imp.values]
bars = ax.barh(
    [FEAT_LABELS.get(f, f) for f in feat_imp.index],
    feat_imp.values,
    color=colors, edgecolor="white", height=0.55
)
ax.set_xlabel("Feature Importance (Gradient Boosting)", fontsize=11)
ax.set_title("What Signals Predicted 2019 Stock Returns?\n(Trained on 2016–2018)", fontsize=12)
for bar, val in zip(bars, feat_imp.values):
    ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}", va="center", fontsize=9)
ax.set_xlim(0, feat_imp.max() * 1.18)
ax.invert_yaxis()
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("figures/fig2_feature_importances.png", dpi=150)
plt.close()
print("Saved figures/fig2_feature_importances.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 – Year-over-year market context: avg return + % positive by year
# ═══════════════════════════════════════════════════════════════════════════════
yr_summary = (
    df[df["Year"].between(2015, 2024)]
    .groupby("Year")
    .agg(Avg_Return=("Yearly Return", "mean"),
         Pct_Positive=("Yearly Return", lambda x: (x > 0).mean()))
    .reset_index()
)

fig, ax1 = plt.subplots(figsize=(10, 5))
bar_colors = ["#d62728" if r < 0 else "#2ca02c" for r in yr_summary["Avg_Return"]]
ax1.bar(yr_summary["Year"], yr_summary["Avg_Return"],
        color=bar_colors, alpha=0.75, width=0.5, label="Avg Annual Return")
ax1.axhline(0, color="black", lw=0.8)
ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax1.set_ylabel("Average Annual Return", fontsize=11)
ax1.set_xlabel("")
ax1.set_xticks(yr_summary["Year"])

ax2 = ax1.twinx()
ax2.plot(yr_summary["Year"], yr_summary["Pct_Positive"],
         "o--", color="#1f77b4", lw=2, ms=6, label="% Stocks Positive")
ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax2.set_ylabel("% Stocks with Positive Return", fontsize=11, color="#1f77b4")
ax2.tick_params(axis="y", colors="#1f77b4")
ax2.set_ylim(0, 1.1)

# Annotate 2019
yr19 = yr_summary[yr_summary["Year"] == 2019].iloc[0]
ax1.annotate("2019 Peak\n(best since 2013)",
             xy=(2019, yr19["Avg_Return"]),
             xytext=(2020.5, yr19["Avg_Return"] + 0.04),
             fontsize=9, color="darkgreen",
             arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.2))

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
ax1.set_title("S&P 500 Constituents – Annual Performance (2015–2024)", fontsize=13)
fig.tight_layout()
plt.savefig("figures/fig3_yearly_context.png", dpi=150)
plt.close()
print("Saved figures/fig3_yearly_context.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 – Sector-level: 2018 signal vs 2019 outcome (mean-reversion story)
# ═══════════════════════════════════════════════════════════════════════════════
data_2018 = model_data[model_data["Year"] == 2018][["Stock", "Sector", "Yearly Return", "Yearly Volatility"]].rename(
    columns={"Yearly Return": "Return_2018", "Yearly Volatility": "Vol_2018"}
)
data_2019 = model_data[model_data["Year"] == 2019][["Stock", "Yearly Return"]].rename(
    columns={"Yearly Return": "Return_2019"}
)
mv = data_2018.merge(data_2019, on="Stock")
sector_mv = mv.groupby("Sector")[["Return_2018", "Return_2019"]].mean().reset_index()

fig, ax = plt.subplots(figsize=(9, 6))
for _, row in sector_mv.iterrows():
    color = PALETTE.get(row["Sector"], "#888888")
    ax.scatter(row["Return_2018"], row["Return_2019"],
               color=color, s=180, zorder=3, edgecolors="white", linewidths=0.8)
    ax.annotate(row["Sector"].replace(" ", "\n"),
                xy=(row["Return_2018"], row["Return_2019"]),
                fontsize=7.5, ha="center", va="bottom",
                xytext=(0, 7), textcoords="offset points")

# Regression line
m, b = np.polyfit(sector_mv["Return_2018"], sector_mv["Return_2019"], 1)
xs = np.linspace(sector_mv["Return_2018"].min() - 0.01,
                 sector_mv["Return_2018"].max() + 0.01, 100)
ax.plot(xs, m * xs + b, "k--", lw=1.2, alpha=0.6, label=f"Trend (slope={m:.2f})")
ax.axvline(0, color="grey", lw=0.8, ls=":")
ax.axhline(0, color="grey", lw=0.8, ls=":")

r_val = np.corrcoef(sector_mv["Return_2018"], sector_mv["Return_2019"])[0, 1]
ax.text(0.04, 0.93, f"Sector r = {r_val:.2f}", transform=ax.transAxes,
        fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax.set_xlabel("Average 2018 Return (by Sector)", fontsize=11)
ax.set_ylabel("Average 2019 Return (by Sector)", fontsize=11)
ax.set_title("Sector Mean-Reversion: 2018 Loss → 2019 Recovery\n(Each dot = one GICS sector, averaged across constituents)", fontsize=12)
ax.legend(fontsize=9)
fig.tight_layout()
plt.savefig("figures/fig4_mean_reversion_sectors.png", dpi=150)
plt.close()
print("Saved figures/fig4_mean_reversion_sectors.png")


# ── Save predictions spreadsheet ──────────────────────────────────────────────
results = results.sort_values("GBM_Pred", ascending=False)
results.to_excel("sp500_2019_predictions.xlsx", index=False)
print("\nSaved sp500_2019_predictions.xlsx")
print("\n✓ All outputs complete.")
