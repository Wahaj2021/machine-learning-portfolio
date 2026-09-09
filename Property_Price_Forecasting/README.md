# Property Price Forecasting

Can monthly median house prices be forecast a year ahead? Fifteen model
specifications across four families were benchmarked against three naive
baselines on twelve years of Australian residential sales.

**The best model beats a flat-line forecast by 1.6%.**

---

## Result

| Model | RMSE | MAE | MAPE |
|---|---|---|---|
| **ARIMA(2,1,2)** | **$25,707** | $18,492 | **2.82%** |
| Naive (no change) | $26,133 | $18,658 | 2.85% |
| XGBoost on lag features | $26,400 | $20,266 | 3.07% |
| ARIMA(0,1,2) | $26,725 | $19,023 | 2.91% |
| SARIMA(0,1,1)(0,0,1,12) | $27,231 | $19,279 | 2.95% |
| Linear Regression on lags | $30,537 | $23,535 | 3.59% |
| Seasonal Naive | $31,932 | $24,950 | 3.69% |
| Holt-Winters (damped) | $33,344 | $26,147 | 3.96% |
| SARIMAX + sales volume | $38,738 | $32,457 | 4.75% |
| Drift | $40,254 | $29,959 | 4.59% |
| Holt-Winters (undamped) | $48,271 | $38,065 | 5.76% |
| SARIMA(0,1,1)(0,1,1,12) | $52,953 | $42,411 | 6.41% |

Predicting "no change for nineteen months" achieves 2.85% mean absolute
percentage error. The best of fifteen models achieves 2.82%.

## What explains the ranking

Almost entirely **how much trend a model extrapolates**.

The series rose steadily from 2014 to 2017, then reversed during the test
period. Models producing near-flat forecasts occupy the top of the table;
models projecting the prior trend forward occupy the bottom.

The clearest demonstration is a single parameter. Holt-Winters with an
undamped trend scores $48,271; the identical model with damping scores
$33,344. One switch, worth $15,000 of RMSE.

## Three findings worth more than the winner

**The seasonal difference is actively harmful.** SARIMA(0,1,1)(0,1,1,12)
scores $52,953 — the worst model tested. The identical specification with
`D=0` scores $27,231. The ADF test predicted this: first differencing gave
p < 0.0001, while adding a seasonal difference *raised* the p-value to
0.0056, indicating over-differencing. Most implementations set `D=1` by
default for seasonal data and never test the alternative.

**Sales volume does not predict price here.** SARIMAX with monthly
transaction volume as an exogenous regressor scored $38,738 — worse than
plain ARIMA — *despite being given actual future volume*, which a
forecaster would not have. The regressor fails even with an unfair
advantage.

**The lag-feature models solve an easier problem and still lose.**
XGBoost and linear regression on lag features use `lag_1`, the previous
month's actual price, so they produce one-step-ahead forecasts with
information refreshed monthly. The ARIMA models forecast all nineteen
months blind from the training cutoff. XGBoost still scores $26,400
against ARIMA's $25,707.

## Why the ceiling is low

![Forecast vs actual](docs/forecast_vs_actual.png)

Actual monthly medians swing between $615k and $710k — a $95k range —
while every model produces a near-flat line around $680k. The models are
not tracking the series; they are predicting its central level and being
scored on how well-centred that estimate is. The naive forecast is
visually indistinguishable from ARIMA(1,1,2), which is why the winning
margin is 1.6%.

Each monthly point is a median over 100–300 sales. Which specific
properties transact in a given month moves that median independently of
any underlying price change — composition, not signal.

**This was tested.** Quarterly aggregation should reduce composition noise
if that is the explanation. ARIMA(1,1,1) on quarterly medians scored MAPE
2.77% against 2.82% monthly, but RMSE $27,853 against $25,707 — an
improvement on one metric and a deterioration on the other. The hypothesis
is only weakly supported, which suggests genuine unpredictability rather
than noise that averaging can remove.

A quality-adjusted index — controlling for property size, location and
type rather than aggregating raw medians — would isolate price movement
from changes in the sales mix, and is the next thing worth testing.

## Approach

**Data.** 29,580 individual residential sales across 27 Australian
postcodes, February 2007 to July 2019. One row per sale: date, postcode,
price, property type, bedrooms. The time series does not exist in the raw
data and is constructed by aggregation.

**Cleaning.** 29,580 → 24,533 rows.

| Issue | Rows | Decision |
|---|---|---|
| `bedrooms = 0` | 30 | Removed — undocumented code, not a valid category |
| Units | 5,017 | Filtered out (see below) |
| Missing values | 0 | None present |
| Price outliers | — | Retained. Max $8,000,000 against a $550,000 median, skewness 4.31 — the median aggregation is robust to these without discarding real sales. |

**Houses only.** Units have a median price of $390,000 against $585,000
for houses, in an 83/17 mix. A shift in the proportion of unit sales would
move the aggregate median without any underlying price change.

**Series starts January 2009.** 2007 contains 147 sales for the entire
year (~12/month) and 2008 has 639. A monthly median from a handful of
sales is noise. From 2009 every month has at least 76 sales.

This yields 127 monthly observations spanning ten complete annual cycles.

**Split.** By date: January 2009 – December 2017 for training (108
months), January 2018 – July 2019 for testing (19 months). A random split
would train on 2019 to predict 2012.

The test period contains a trend reversal — the series peaks in early 2018
and declines through 2019 — so models are evaluated on a turning point
rather than a smooth continuation. This is the harder case and the more
informative one.

## Time series analysis

| Series | ADF statistic | p-value | Result |
|---|---|---|---|
| Original | −0.957 | 0.7687 | Non-stationary |
| 1st difference | −9.312 | <0.0001 | Stationary |
| 1st + seasonal difference | −3.606 | 0.0056 | Over-differenced |

First differencing is sufficient (**d = 1**). Multiplicative decomposition
shows a regular annual seasonal component ranging 0.94–1.04 — a ±5% swing
repeating consistently across ten cycles — and three trend regimes: a
plateau to 2013, acceleration to 2018, then flattening.

On the differenced series, ACF cuts off after a single negative spike at
lag 1 (≈ −0.42) while PACF decays gradually — the signature of an MA(1)
process.

## Order selection

A grid over p, q ∈ {0,1,2,3} with d=1 was evaluated. Five specifications
cluster at the top — (1,1,2) at $25,696, (1,1,3) at $25,702, (2,1,2) at
$25,707, (3,1,1) at $25,751 and (2,1,1) at $25,774 — all within $80 of
each other and all beating naive. The clustering indicates a genuine
region of the parameter space rather than one fortunate configuration.

AIC and test RMSE agree: the AIC-minimising order (2,1,1) sits within $80
RMSE of the RMSE-minimising order (1,1,2). Selection was not driven by
test-set performance.

ARIMA(0,1,0) scores exactly $26,133 — identical to naive, as expected,
since with no AR or MA terms the two are mathematically the same forecast.

## A note on R²

Every model has negative R², including the winner. This is expected and
not a sign of failure: R² compares against the mean of the *test* period,
a value no forecaster could know in advance, and across a nearly flat
19-month window that mean is a strong benchmark. RMSE and MAPE are the
meaningful metrics here.

## Running it

```bash
git clone https://github.com/Wahaj2021/property-price-forecasting.git
cd property-price-forecasting
pip install -r requirements.txt
jupyter notebook notebooks/property_price_forecasting.ipynb
```

## Limitations

- **127 monthly observations.** Enough for ten seasonal cycles, but a
  short series by forecasting standards.
- **19-month test window.** A 1.6% margin over the baseline on 19 points
  is real but modest, and should not be overstated.
- **Prophet was excluded** on environment grounds. Its changepoint
  detection would be relevant given the three trend regimes, though the
  results suggest the limiting factor is the trend reversal rather than
  model choice.
- **Deep learning was ruled out.** LSTM and similar architectures need
  thousands of observations; 127 would overfit badly.
- **One market.** 27 postcodes in a single Australian region. The
  conclusions are about this series, not property markets generally.
- **No external regressors that would be known in advance.** Interest
  rates, listings inventory and lending conditions all plausibly matter
  and none are in the dataset.
