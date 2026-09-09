# Review Rating Prediction

Predict the star rating a customer would give from the text of their review
alone. Five specifications were benchmarked, from a majority-class baseline
to a fine-tuned transformer, on 194,000 Amazon product reviews.

**This is the one project where model complexity clearly earns its place.**

---

## Result

| Model | Accuracy | Macro F1 | MAE (stars) |
|---|---|---|---|
| Majority class (always 5) | 0.5601 | 0.1436 | 0.8813 |
| TF-IDF + Linear SVM | 0.6241 | 0.4295 | 0.5532 |
| TF-IDF + Logistic Regression | 0.6460 | 0.4418 | 0.5296 |
| TF-IDF + LogReg (class-balanced) | 0.5786 | 0.4536 | 0.6096 |
| **DistilBERT (fine-tuned)** | **0.6999** | **0.5445** | **0.3734** |

DistilBERT improves accuracy by 5.4 points over the best TF-IDF model,
macro F1 by 0.10, and reduces mean absolute error by 29% — from 0.53 to
0.37 stars.

The comparison is **conservative toward the transformer**: it was
fine-tuned on a 50,000-row subsample due to compute limits, against
155,430 rows for the TF-IDF models. It won with a third of the data.

## Where the gain actually is

Not evenly spread. The transformer's advantage concentrates in the errors
that matter.

**The expensive mistakes collapse.** TF-IDF predicted 569 actual 1-star
reviews as 5-star — a furious customer classified as delighted.
DistilBERT predicts 76. An **87% reduction** in the worst error type.

**The ambivalent middle improves most.** Class 3 recall rose from 0.275 to
0.493. During exploration, 3-star reviews were identified as genuinely
mixed — *"It's an Otterbox so you know it's going to protect your phone
but for whatever reason I have a tough time pushing the lock button"*.
Resolving that requires reading context, not counting keywords.

**Class 2 remains the weakest** — recall 0.100 → 0.224, F1 0.150 → 0.282.
It is the rarest class at 5.7% and sits between two neighbours, so it is
squeezed from both sides. Most of its errors go to classes 1 and 3 rather
than to 5, so they remain cheap in MAE terms.

### Confusion matrix — DistilBERT

|  | →1 | →2 | →3 | →4 | →5 |
|---|---|---|---|---|---|
| **1** | 1868 | 352 | 388 | 55 | 108 |
| **2** | 622 | 496 | 845 | 133 | 121 |
| **3** | 258 | 363 | 2081 | 1017 | 505 |
| **4** | 66 | 64 | 1031 | 3151 | 3388 |
| **5** | 76 | 26 | 380 | 1741 | 19311 |

Errors cluster along the diagonal. The 4↔5 boundary accounts for 5,129 of
them — cheap mistakes on an ordinal target, which is why MAE is 0.37 while
accuracy is 0.70.

## Why accuracy is the wrong headline

The majority-class baseline achieves **56% accuracy** by predicting 5 stars
for every review. Its macro F1 is 0.14, because it never predicts four of
the five classes at all.

That gap is why macro F1 is the primary metric here, and why MAE sits
alongside it: the target is **ordinal**, so predicting 4 when the truth is
5 is not the same mistake as predicting 1. Accuracy scores both as equally
wrong.

**Class weighting was tested and rejected.** `class_weight='balanced'`
improved macro F1 marginally (0.4418 → 0.4536) while costing 6.7 points of
accuracy and worsening MAE by 15%. It buys a small gain on the rarest class
by degrading everything else. The unweighted model is preferred on the
ordinal argument above.

## The leakage problem

**99.97% of reviewers appear more than once** — 27,871 of 27,878, averaging
7 reviews each.

A random train/test split would place the same person's reviews on both
sides, letting the model learn individual writing style and rating habits
rather than sentiment. This is **group leakage**, and it would inflate the
test score in a way that would not survive contact with a new customer.

`GroupShuffleSplit` on `reviewerID` was used instead, so every reviewer
falls entirely within either train or test. Reviewer overlap between the
two sets is zero. Class proportions were checked afterwards and held within
half a percentage point despite no stratification being possible.

A second leakage trap was avoided in feature selection: the `helpful`
column records votes that accrue **after** a review is posted, so it is not
available at prediction time.

## Approach

**Data.** Amazon product reviews, Cell Phones & Accessories category —
194,439 reviews across 10,429 products and 27,878 reviewers, 2007 to 2013.
Text and star rating only; no product, price or reviewer metadata is used.

**Cleaning.** Empty and near-empty reviews removed (99 empty, plus those
under 3 words), and 32 exact duplicate review texts dropped so identical
text could not appear on both sides of the split.

**Class balance.** 55.9% 5-star, 20.6% 4-star, 11.0% 3-star, 6.8% 1-star,
5.7% 2-star — roughly 10:1 between the largest and smallest classes.

**Text length.** Median 48 words, mean 92, maximum 5,263. 89.6% fall under
200 words and the 95th percentile is 307, so `max_length=256` tokens was
chosen for the transformer: it covers the substantial majority in full
while training roughly twice as fast as the 512-token maximum.

**Length is non-monotonic in rating.** 4-star reviews are longest (mean 110
words), while 5-star (85) and 1-star (77) are shortest. Extreme opinions
are stated briefly; qualified ones require explanation. This makes length a
weak feature that linear models cannot exploit directly.

**Vocabulary overlaps across classes.** `phone`, `case`, `battery`,
`charge` and `screen` appear in the top terms for both 1-star and 5-star
reviews — product nouns, not sentiment. Only `great`, `good`, `love` versus
`did`, `don`, `work` discriminate. This motivates TF-IDF, which downweights
terms common across the corpus, and bigrams, which capture negation
("not good" carries the opposite meaning to "good").

## Running it

```bash
git clone https://github.com/Wahaj2021/review-rating-prediction.git
cd review-rating-prediction
pip install -r requirements.txt
jupyter notebook notebooks/review_rating_prediction.ipynb
```

The TF-IDF models run on CPU in a few minutes. The DistilBERT fine-tuning
requires a GPU — `notebooks/distilbert_colab.ipynb` runs in Google Colab on
a free T4 in roughly 13 minutes for 2 epochs. The trained model (260MB)
is not committed to this repository.

Data: [Amazon product reviews](https://jmcauley.ucsd.edu/data/amazon/),
Cell Phones & Accessories 5-core subset.

## Limitations

- **One product category.** Phone accessories only. Vocabulary and
  complaint patterns would differ in books, groceries or clothing, and the
  model would need retraining per category.
- **2007–2013.** Language and product types have moved on.
- **Compute-limited training.** The transformer used 50,000 of 155,430
  available training rows and 2 epochs. Validation loss was still falling
  slightly at epoch 2, so more training would likely yield a small further
  gain.
- **Class 2 is not solved.** 0.224 recall means three quarters of 2-star
  reviews are misclassified.
- **Rating conflates product and service.** One sampled 5-star review
  contains "It took forever for it to ship tho" — the rating reflects the
  product while the text complains about delivery. Some ceiling on this
  task is irreducible.
- **No calibration.** The model outputs a predicted class, not a
  probability that has been checked against observed frequencies.

## Note on model complexity

Across four projects, the same question was tested: does added complexity
earn its place?

- **Credit default risk** — ten models within noise of each other; the
  simplest was selected.
- **Customer value** — ten models, none beat a one-line naive baseline.
- **Property price forecasting** — the winner beat a flat-line forecast by
  1.6%; eleven of fifteen models were worse than doing nothing.
- **Review rating prediction** — a 66-million-parameter transformer beats
  TF-IDF decisively, on a third of the training data.

The difference is the nature of the task. Where the signal is a small
number of strong numeric predictors, simple models capture it. Where the
signal depends on context, negation and word order, it does not survive
being reduced to term frequencies — and the extra capacity pays for itself.
