# Customer Value Prediction

Can a retailer predict how much each customer will spend next quarter, and
who is about to stop buying? Ten model specifications across six families
were benchmarked against a naive baseline on UK e-commerce transaction
data.

**Headline: the spend prediction fails. The churn prediction works.**

---

## Result

| Task | Outcome |
|---|---|
| Predict forward **spend** (£) | No model beat a naive baseline |
| Predict **whether** a customer returns | ROC AUC 0.7313 — usable |

The distinction only became visible because the regression was benchmarked
against a baseline. Most published work on this dataset reports R² around
0.5 without establishing what a one-line heuristic achieves.

## The regression finding

The baseline is *"forward spend equals prior spend, scaled by the ratio of
window lengths"* — one line of arithmetic, no model.

| Approach | RMSE | MAE | R² |
|---|---|---|---|
| **Baseline (prior spend × 13/39)** | **£4,480** | **£751** | **0.5088** |
| Ridge, raw target | £4,757 | £850 | 0.4462 |
| Ridge, log target | £5,521 | £828 | 0.2541 |
| Two-stage (logistic + ridge) | £5,566 | £810 | 0.2417 |
| Random Forest, raw target | £5,812 | £1,059 | 0.1733 |

**Why.** Prior spend alone correlates 0.818 with forward spend, so the
baseline already captures most of the available signal. The remaining
fourteen features add little, and the transformations needed to fit a
target skewed at 18.3 introduce more error on inversion than they recover:
an error of 1.0 in log space near £100,000 becomes a £170,000 error in
pounds, and the wholesale customers who dominate squared error sit exactly
there.

The two-stage zero-inflated model was tested because 43.6% of targets are
exactly zero. Its classifier component works in isolation (ROC AUC 0.73),
but multiplying a return probability by a predicted amount shrinks
high-value predictions and worsens RMSE overall.

## The churn finding

The same features predict *whether* a customer returns.

| Metric | Value |
|---|---|
| ROC AUC | 0.7313 |
| Average precision | 0.8014 |
| Accuracy | 67.0% (base rate 57.1%) |

The practical value is concentrated in the tail:

| Targeting | Actually returned | vs overall |
|---|---|---|
| Lowest 10% by predicted return probability | 31.8% | 57.1% |
| Lowest 20% | 36.4% | 57.1% |
| Lowest 30% | 35.7% | 57.1% |

A campaign aimed at the bottom decile reaches customers roughly **1.8×
more likely to lapse** than a random selection. The advantage narrows as
targeting widens, so the model supports a focused retention campaign
rather than a broad one.

## Why the two tasks diverge

Feature correlations rank almost inversely depending on which question is
being asked:

| Feature | vs raw spend | vs log spend |
|---|---|---|
| `total_spend` | **0.818** | 0.238 |
| `frequency` | 0.449 | 0.358 |
| `single_purchase` | −0.109 | **−0.351** |
| `recency_days` | −0.131 | **−0.340** |

Raw correlation is dominated by a small number of wholesale customers, for
whom prior spend is the strongest signal. On the log scale, where the
43.6% of customers with zero forward spend carry proportionate weight,
recency and single-purchase status dominate instead.

Confirming this, XGBoost assigned `single_purchase` an importance of
**exactly zero** despite its −0.351 correlation with the log target. A
single regressor optimises for high-value customers and does not learn the
churn question at all — which is why splitting the two tasks was worth
testing, and why the classifier succeeds where the regressor does not.

## Approach

**Data.** UCI Online Retail — 541,909 transaction lines from a UK online
giftware retailer, 1 Dec 2010 to 9 Dec 2011.
[Source](https://archive.ics.uci.edu/dataset/352/online+retail). One row
per line item, not per customer.

**Cleaning.** 541,909 rows → 392,692.

| Issue | Rows | Decision |
|---|---|---|
| Exact duplicates | 5,268 | Removed — would double measured spend |
| Cancellations (`C` prefix) | 9,288 | Removed |
| Negative quantity (returns) | 10,624 | Removed |
| Non-positive unit price | 2,517 | Removed — adjustments and samples |
| Missing `CustomerID` | 135,080 | Removed — cannot attribute to a customer |

Administrative stock codes (`POST`, `DOT`, `M`, `C2`, `D`, `S`, `CRUK`,
`PADS`, `B`, `m`) were **retained**: postage and carriage are real charges
paid by the customer and form part of customer value. Note that `M` and
`m` are the same code entered inconsistently.

**Temporal design.** This is the part that determines whether the project
is valid at all.

```
2010-12-01 ── observation (39 wks) ──► 2011-08-31 │ target (13 wks) ► 2011-11-30
              features built here                   label measured here
```

392,692 line items aggregate to 3,317 customers. Features come only from
the observation window; the target only from the forward window. A random
split would allow a feature such as purchase frequency to include
transactions from the period being predicted — target leakage, and the
most common error in customer-value notebooks.

December 2011 is excluded entirely: the data stops on the 9th, and a
partial month would understate every customer's target.

Customers who purchased before the cutoff and never returned receive a
target of £0. They are 43.6% of the population, and excluding them would
train on survivors only and cause systematic overprediction.

`recency_days` is measured from the last date in the observation window
rather than from today, since the data is fifteen years old.

**Features.** Fifteen, engineered from transaction history: recency,
tenure, frequency, total spend, total units, distinct products, line
count, average unit price, average order value, average basket size, lines
per order, spend per day, purchase rate, UK flag, single-purchase flag.

## Model comparison

Selection on 5-fold cross-validated R² in log space, training data only.

| Model | Train R² | CV R² | CV std | Gap |
|---|---|---|---|---|
| Random Forest | 0.3395 | 0.2779 | 0.0381 | 0.0616 |
| XGBoost | 0.3253 | 0.2757 | 0.0376 | 0.0497 |
| Gradient Boosting | 0.3294 | 0.2753 | 0.0364 | 0.0541 |
| Ridge | 0.2834 | 0.2704 | 0.0358 | 0.0130 |
| ElasticNet | 0.2830 | 0.2704 | 0.0355 | 0.0126 |
| Linear | 0.2836 | 0.2699 | 0.0366 | 0.0137 |
| Lasso | 0.2833 | 0.2699 | 0.0358 | 0.0134 |
| KNN | 0.2995 | 0.2680 | 0.0317 | 0.0315 |
| Decision Tree | 0.3177 | 0.2368 | 0.0337 | 0.0809 |
| Polynomial (deg 2) | 0.3361 | 0.1668 | 0.0934 | 0.1693 |

Eight of ten fall between 0.268 and 0.278 — a spread of 0.010 against fold
standard deviations of ~0.036. The ranking among them is not
statistically meaningful, and the convergence across six model families
indicates the ceiling is set by the information in the features rather
than by algorithm choice.

Polynomial expansion is the clearest overfit: 135 interaction terms on
2,653 rows, a train–CV gap of 0.169 and the highest fold variance.

All three regularised linear models selected the **smallest penalty in
their grid** (α = 0.001) and Lasso eliminated no features — with 15
predictors and 2,653 rows there is nothing for regularisation to shrink.

**Ridge was selected**: it matches the best CV score within noise while
carrying a train–CV gap of 0.013 against Random Forest's 0.062, and is the
simplest and fastest specification. Where cross-validation cannot
distinguish models, the simpler one is preferred.

Linear coefficients are **not individually interpretable** here:
`total_units` (+1.90) and `total_spend` (−1.77) carry large offsetting
weights because the monetary features are arithmetically related.

## Limitations

- **39 weeks of history.** Too short to separate customer behaviour from
  seasonality, or to establish stable purchase cycles.
- **The target window is the Q4 peak.** Revenue roughly doubles from
  August (£760k) to November (£1.5m). With one year of data, seasonality
  cannot be separated from customer effects, so the model predicts
  *peak-season* spend rather than typical-quarter spend.
- **24.9% of transactions have no customer ID** and were excluded. These
  are real sales, so revenue figures here understate the retailer's
  turnover.
- **No marketing, browsing or competitor data.** Whether a customer
  returns depends heavily on factors absent from a transaction log.
- **3,317 customers** after aggregation — small, which is why fold
  standard deviations run at 0.036 and why small differences between
  models cannot be resolved.

## What would change the conclusion

Longer history, or features the transaction log does not contain: email
and campaign contact, site visits without purchase, product category
affinity, and competitor pricing. The regression ceiling here is a data
limitation, not a modelling one — which is a conclusion only reachable
with a baseline in place.
