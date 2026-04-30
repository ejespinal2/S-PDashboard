# S&P 500 2019 Stock Performance Analysis

Was the 2019 S&P 500 rally predictable? This project investigates that question by building a machine learning model trained exclusively on 2016–2018 data and testing it against actual 2019 outcomes. The model achieves an R² of 0.37, suggesting that a meaningful portion of 2019's cross-sectional stock performance was foreseeable from signals that existed before the year began.

**[View the Tableau Dashboard →](https://public.tableau.com/views/SP5002019StockData/Dashboard2?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)**

---

## Background

2019 was the S&P 500's best year since 2013, with the average constituent returning approximately 27%. This followed a sharp Q4 2018 sell-off driven by Federal Reserve rate hike fears. When the Fed pivoted to a pause in January 2019, a broad recovery followed — but not all stocks recovered equally. This project examines which stocks led the recovery, why, and whether those outcomes could have been anticipated.

---

## Project Structure

```
├── SP_Return_Data.py                  # Downloads S&P 500 index-level data
├── SP500_Stock_Return_Data.py         # Downloads individual constituent data
├── SP500_2019_Predictive_Analysis.py  # Predictive model + chart generation
├── sp500_returns.xlsx                 # Output of SP_Return_Data.py
├── sp500_stock_returns.xlsx           # Output of SP500_Stock_Return_Data.py
├── Yearly_Performance.xlsx            # Per-stock yearly returns & volatility (2015–2025)
├── Monthly_Returns.xlsx               # Per-stock monthly returns (2015–2025)
└── figures/
    ├── fig1_predicted_vs_actual.png   # Model predictions vs 2019 actuals
    ├── fig2_feature_importances.png   # What signals drove the model
    ├── fig3_yearly_context.png        # Annual return & breadth (2015–2024)
    └── fig4_mean_reversion_deciles.png # 2018 losers becoming 2019 winners
```

---

## Python Files

All price data is sourced from the [yfinance](https://github.com/ranaroussi/yfinance) library.

### `SP_Return_Data.py`
Downloads S&P 500 index (`^GSPC`) data from 2015 to 2025 and computes:
- Compound monthly returns
- Compound yearly returns
- Annualised yearly volatility
- Yearly Sharpe ratio
- Yearly max drawdown

Outputs `sp500_returns.xlsx`.

### `SP500_Stock_Return_Data.py`
Downloads historical price data for every S&P 500 constituent (sourced from Wikipedia) and computes the same metrics as above at the individual stock level, with GICS sector labels attached to every row.

Outputs `sp500_stock_returns.xlsx`. Any tickers that fail to download are logged to `download_errors.csv`.

### `SP500_2019_Predictive_Analysis.py`
The core analysis script. Uses Ridge Regression trained on 2016–2018 stock data to predict 2019 returns out-of-sample. The three key predictive signals are:

- **3-Year Momentum** — sustained multi-year performance (2016–2018) was a stronger predictor than single-year momentum, capturing sectors like Information Technology whose structural tailwinds outlasted short-term macro noise.
- **Prior Volatility** — stocks with high 2018 volatility had the most compressed valuations going into 2019. When fear subsided, these had the most room to recover.
- **Mean-Reversion** — the Q4 2018 sell-off was macro-driven, not fundamental. Stocks that fell hardest in 2018 snapped back strongest once the Fed paused rate hikes.

Outputs `sp500_2019_predictions.xlsx` and four chart images in `figures/`.

---

## Excel Files

| File | Description |
|---|---|
| `sp500_returns.xlsx` | S&P 500 index monthly returns, yearly returns, volatility, Sharpe, and max drawdown (2015–2025) |
| `sp500_stock_returns.xlsx` | Same metrics for individual S&P 500 constituents |
| `Yearly_Performance.xlsx` | Per-stock yearly returns and volatility (2015–2025), used as model input |
| `Monthly_Returns.xlsx` | Per-stock monthly returns (2015–2025) |
| `sp500_2019_predictions.xlsx` | Model-predicted vs actual 2019 returns for each stock |

---

## Requirements

```
yfinance
pandas
numpy
scikit-learn
matplotlib
seaborn
openpyxl
```

Install with:

```bash
pip install yfinance pandas numpy scikit-learn matplotlib seaborn openpyxl
```

---

## How to Run

Run the scripts in this order:

```bash
python SP500_Stock_Return_Data.py     # ~10-15 min, downloads 500 stocks
python SP_Return_Data.py              # ~1 min, downloads index data
python SP500_2019_Predictive_Analysis.py  # runs instantly on saved Excel files
```

The analysis script reads from `Yearly_Performance.xlsx` by default. If you have regenerated the data yourself, update the `YEARLY_FILE` variable at the top of the script to point to `sp500_yearly_performance_v2.xlsx`.
