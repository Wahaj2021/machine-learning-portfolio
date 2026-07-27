# Symptom-Based Disease Diagnosis System

A multi-class classification model predicting the most likely disease from
reported symptoms, across 42 disease classes and 132 symptoms.

## Objective
Map a set of reported symptoms to the most probable diagnosis — a large
multi-class problem (42 classes).

## Approach
CRISP-DM workflow: encoding the symptom features, train/test split with
cross-validation, and training multiple classifiers.

## Results
All models achieved **100% accuracy** on train, cross-validation and test.
Rather than take this at face value, it's worth being clear about *why*: this
dataset is effectively a clean, deterministic symptom-to-disease mapping with
little noise, so near-perfect scores reflect the simplicity of the data rather
than exceptional model performance. A real diagnostic system would need messy,
overlapping, real-world symptom data to be meaningfully evaluated.

## Tools
Python, pandas, scikit-learn, matplotlib, seaborn.

## Dataset
Symptom–disease dataset (132 symptoms, 42 diseases).
