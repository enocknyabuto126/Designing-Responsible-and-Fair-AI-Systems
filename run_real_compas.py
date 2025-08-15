#!/usr/bin/env python3
"""
Simple script to run REAL COMPAS dataset analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load real COMPAS data
print("Loading REAL COMPAS dataset...")
df = pd.read_csv('data/compas-scores-two-years.csv')
print(f"Dataset shape: {df.shape}")

# Basic preprocessing
df_clean = df.dropna(subset=['race', 'sex', 'age', 'decile_score', 'two_year_recid'])
print(f"After cleaning: {df_clean.shape}")

# Create binary race variable
df_clean['race_binary'] = (df_clean['race'] == 'African-American').astype(int)
df_clean['sex_binary'] = (df_clean['sex'] == 'Male').astype(int)
df_clean['high_risk'] = (df_clean['decile_score'] >= 7).astype(int)

# Age categories
df_clean['age_cat'] = pd.cut(df_clean['age'], bins=[0, 25, 35, 45, 100], labels=['18-25', '26-35', '36-45', '45+'])
age_dummies = pd.get_dummies(df_clean['age_cat'], prefix='age')
df_clean = pd.concat([df_clean, age_dummies], axis=1)

# Features for modeling
feature_cols = ['race_binary', 'sex_binary', 'age_18-25', 'age_26-35', 'age_36-45', 'age_45+']
for col in feature_cols:
    if col not in df_clean.columns:
        df_clean[col] = 0

# Statistics
print("\nRace distribution:")
print(df_clean['race'].value_counts())

print("\nHigh risk rates by race:")
high_risk_by_race = df_clean.groupby('race')['high_risk'].mean()
print(high_risk_by_race)

print("\nRecidivism rates by race:")
recidivism_by_race = df_clean.groupby('race')['two_year_recid'].mean()
print(recidivism_by_race)

# Model training
X = df_clean[feature_cols]
y = df_clean['two_year_recid']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"\nModel accuracy: {np.mean(y_pred == y_test):.3f}")

# Fairness metrics
test_data = df_clean.iloc[X_test.index].copy()
aa_mask = test_data['race_binary'] == 1
ca_mask = test_data['race_binary'] == 0

aa_pred = y_pred[aa_mask]
ca_pred = y_pred[ca_mask]

aa_positive_rate = aa_pred.mean()
ca_positive_rate = ca_pred.mean()
spd = aa_positive_rate - ca_positive_rate

print(f"\nStatistical Parity Difference: {spd:.4f}")
print(f"African-American positive rate: {aa_positive_rate:.4f}")
print(f"Caucasian positive rate: {ca_positive_rate:.4f}")

print("\n🎉 REAL COMPAS analysis completed!")
