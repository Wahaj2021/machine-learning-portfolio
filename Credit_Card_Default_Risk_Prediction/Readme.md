# Credit Card Default Risk

A deployed model that ranks credit card accounts by their probability of
defaulting on next month's payment, with a decision threshold derived from
the cost of each error type and constrained by how many accounts a
collections team can realistically contact.

**[Live demo](https://credit-risk-model-n721.onrender.com)** · **[API docs](https://credit-risk-model-n721.onrender.com/docs)**

---

## The problem

A card issuer wants to intervene on accounts that are about to default —
a phone call, a payment plan, a credit line review — before the loss
happens. The constraint is that a collections team can only contact a
small fraction of the book each month.

So the question is not "who will default?" It is **"which 5% of accounts
should we call, and what is that worth?"**

## Result

At the deployed threshold, on a held-out test set of 6,000 accounts:

| | Value |
|---|---|
| Accounts flagged | 294 (4.9% of the book) |
| Of those, genuinely defaulted | 226 |
| Precision | 76.9% |
| Lift over random targeting | **3.5×** |
| Expected saving | ~NT$1.1m per 1,000 accounts |

Contacting a random 5% of customers would reach roughly 66 defaulters.
The model reaches 226 for the same collections effort. On a
30,000-account book that is approximately NT$33m of avoided losses per
year.

## The finding that matters more than the model

Sweeping the decision threshold against the cost matrix produces two very
different answers:

- **Cost-minimising threshold: 0.07**, which flags 84% of the book.
  Arithmetically correct at a 15:1 cost ratio, and completely
  unimplementable.
- **Capacity-constrained threshold: 0.71**, which flags 4.9%.

The gap between them is **NT$97m**. For comparison, the spread between
the best and worst of six candidate models was 0.056 average precision.

The operational constraint dominates model quality by roughly an order of
magnitude. What each additional percentage point of contact capacity is
worth:

| Contact capacity | Threshold | Expected cost |
|---|---|---|
| 2% | 0.77 | NT$148.0m |
| 5% | 0.71 | NT$132.9m |
| 10% | 0.57 | NT$110.0m |
| 20% | 0.33 | NT$81.8m |
| 30% | 0.23 | NT$65.6m |

Doubling collections capacity from 5% to 10% is worth **NT$22.9m** — more
than any achievable improvement to the model. That is a resourcing
recommendation, and it is the most useful output of the project.

## Cost assumptions

Every threshold decision follows from two numbers, both stated explicitly
so they can be challenged:

- **Missed default: NT$30,000.** A typical defaulting balance of ~NT$40,000
  at a loss-given-default of 75%, standard for unsecured card debt.
- **Unnecessary intervention: NT$2,000.** Operational cost of contact plus
  attrition risk from approaching a customer who would have paid.

These are estimates, not the issuer's actuals. They are the single largest
source of uncertainty in the analysis, which is why they live in one
constant block rather than scattered through the code.

## Approach

**Data.** UCI *Default of Credit Card Clients* — 30,000 Taiwanese
customers, April–September 2005, 23 features, 22.1% default rate.
[Source](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).
Not included in this repo; download it to `data/`.

**Feature engineering.** Six domain features replace what PCA would have
extracted, because credit decisions must be explainable to the customer
and to a regulator. `months_delinquent` alone separates the book into
groups defaulting at 11.7% to 70.3%:

| Months late in last 6 | Default rate | Accounts |
|---|---|---|
| 0 | 11.7% | 19,918 |
| 1 | 29.9% | 4,405 |
| 2 | 38.8% | 1,899 |
| 3 | 50.9% | 1,154 |
| 4 | 57.3% | 951 |
| 5 | 57.4% | 298 |
| 6 | 70.3% | 1,340 |

**Model selection.** Six families compared by 5-fold cross-validated
average precision on training data only. Accuracy is not used: the
majority-class baseline is 77.9%.

| Model | CV avg. precision | Train–CV gap |
|---|---|---|
| Logistic regression | 0.5073 | 0.004 |
| Decision tree | 0.5234 | 0.041 |
| KNN | 0.5248 | 0.032 |
| AdaBoost | 0.5414 | 0.010 |
| Random forest | 0.5611 | 0.088 |
| **XGBoost** | **0.5637** | 0.063 |

SVM was excluded on computational grounds — `SVC` scales approximately
O(n²), and it produces uncalibrated scores, which this project depends on.

**Tie-break on calibration.** The top three ensembles are statistically
tied (0.5611–0.5637 against a fold standard deviation of 0.022), so the
ranking between them is noise. XGBoost was selected because its predicted
probabilities deviate from observed default rates by at most **0.0097**,
against 0.0135 for gradient boosting and 0.0269 for random forest.

Calibration is the deciding criterion because the threshold is derived
from a cost matrix, and that arithmetic is only valid if a predicted 0.71
corresponds to a real 71% default rate. It is also why SMOTE was rejected:
resampling distorts predicted probabilities and would have invalidated
the cost calculation. Class imbalance here is moderate (22.1%, 6,636
minority examples) and is handled by threshold selection rather than by
altering the training distribution.

**Held-out performance.** Test average precision 0.5548 against a
cross-validated 0.5637 — a gap of 0.009, and meaningful because model and
threshold were both selected without touching the test set. ROC AUC 0.778,
consistent with published results on this dataset.

## Why recall is 17%

The model catches 226 of 1,327 defaulters. This is a consequence of the
capacity constraint, not a modelling failure: 1,327 accounts default and
the team may contact 294. **A perfect model caps out at 22.2% recall
under the same limit**, so the model captures roughly three quarters of
what is achievable.

Reporting recall without the constraint attached would be misleading in
either direction.

## Fairness

`SEX` and `MARRIAGE` were retained to permit measurement of outcome
disparity, which exclusion alone would not reveal.

| | n | Actual default rate | Flag rate |
|---|---|---|---|
| Male | 2,332 | 24.4% | 5.9% |
| Female | 3,661 | 20.7% | 4.7% |

Flag rates differ by a factor of 1.14 against an underlying default ratio
of 1.18, so the model tracks the observed difference slightly
conservatively rather than amplifying it.

Under equalised odds, the meaningful test is whether error rates match.
**False negative rates are identical (0.824 for both groups)**; false
positive rates are 0.021 and 0.014, on rates already suppressed by the
capacity constraint.

In a live UK deployment both features would nonetheless be excluded: the
Equality Act 2010 prohibits their use in credit decisions regardless of
measured fairness, and their predictive contribution is marginal. Removing
them would not remove the disparity, since `LIMIT_BAL` and `EDUCATION` act
as proxies.

## What the model responds to

Sensitivity analysis on a representative account:

| Feature varied | Range of predicted probability |
|---|---|
| `PAY_0` (−1 to 4) | 0.277 → 0.689 |
| `LIMIT_BAL` (10k to 500k) | 0.644 → 0.694 |
| `BILL_AMT1` (1k to 100k) | 0.628 → 0.687 |

Repayment history moves the prediction roughly eight times more than
balance or credit limit, and the response to `PAY_0` plateaus beyond two
months delinquent — matching the flattening observed during exploration.
The model responds to whether a customer paid on time, not to how much
they owe.

## Running it

```bash
git clone https://github.com/Wahaj2021/credit-risk-model.git
cd credit-risk-model
pip install -r requirements.txt

# download UCI_Credit_Card.csv into data/ if retraining
python src/train.py          # optional; a trained model is included

cd src
uvicorn app:app --reload
```

Then open <http://localhost:8000> for the scoring form, or
<http://localhost:8000/docs> for the API.

With Docker:

```bash
docker build -t credit-risk .
docker run -p 8000:8000 credit-risk
```

### API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"LIMIT_BAL":50000,"SEX":2,"EDUCATION":2,"MARRIAGE":1,"AGE":29,
       "PAY_0":2,"PAY_2":2,"PAY_3":2,"PAY_4":0,"PAY_5":0,"PAY_6":0,
       "BILL_AMT1":48000,"BILL_AMT2":47000,"BILL_AMT3":46000,
       "BILL_AMT4":45000,"BILL_AMT5":44000,"BILL_AMT6":43000,
       "PAY_AMT1":1500,"PAY_AMT2":1500,"PAY_AMT3":1200,
       "PAY_AMT4":1000,"PAY_AMT5":1000,"PAY_AMT6":1000}'
```

```json
{
  "probability": 0.6846,
  "threshold": 0.71,
  "decision": "NO_ACTION",
  "reasons": [
    "Late in 3 of the last 6 months",
    "Worst delinquency: 2 months behind",
    "Credit utilisation at 96%"
  ]
}
```

This account is predicted to default — 68.5% is well above the 50%
statistical cutoff — but is not contacted, because 294 accounts rank
higher and the team has capacity for 294. The `reasons` field exists
because a collections officer can act on "late in 3 of the last 6 months"
and cannot act on "PAY_3 = 2".

Endpoints: `GET /` form · `GET /health` · `GET /metadata` ·
`POST /predict` · `POST /predict/batch` (returns accounts ranked by risk,
which is what a capacity-constrained team actually consumes).

## Structure

```
├── notebooks/credit_risk.ipynb   Exploration, modelling, evaluation
├── src/
│   ├── preprocessing.py          Stateless cleaning + feature engineering
│   ├── train.py                  Split, pipeline, selection, threshold, save
│   ├── app.py                    FastAPI service
│   └── static/index.html         Scoring form
├── model/
│   ├── model.pkl                 Full sklearn Pipeline
│   └── metadata.json             Threshold, costs, versions, performance
├── requirements.txt              Pinned to the versions the model was built with
└── Dockerfile
```

`preprocessing.py` contains only stateless operations — no statistic is
learned from the data — so it is safe to apply before the train/test split
and it is imported by both `train.py` and `app.py`, which guarantees
training and serving apply identical logic.

Everything that learns from data (imputation, Yeo-Johnson transform,
scaling, one-hot encoding) lives inside the `Pipeline`, so it is fitted on
training data only, refitted within each cross-validation fold, and
serialised together with the estimator. `model.pkl` therefore accepts raw
customer values; no preprocessing is reimplemented at serving time.

Yeo-Johnson rather than Box-Cox: 590 accounts carry a negative
`BILL_AMT1`, being in credit, and Box-Cox requires strictly positive input.

## Limitations

- **The cost figures are estimates.** They drive the threshold and every
  monetary claim above.
- **The data is thin.** Six months of repayment codes and balances, with no
  income, employment, bureau file, or cross-account behaviour. ROC AUC of
  0.778 is at the ceiling this dataset supports; a production model with
  bureau data would do considerably better.
- **2005, Taiwan.** Twenty-year-old data from a different market. The
  relationships would not transfer directly to UK lending today.
- **No drift monitoring.** A production deployment would need
  population stability monitoring and a retraining schedule.

What transfers is the method rather than the model: attach costs to each
error type, derive the threshold from those costs, respect the operational
constraint, verify calibration before trusting the arithmetic, and report
what capacity is worth.
