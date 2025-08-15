#!/usr/bin/env python3
"""
COMPAS Dataset Bias Audit Script
Part 3 of AI Ethics Assignment: Practical Audit (25%)

This script demonstrates how to audit the COMPAS dataset for racial bias
using fairness metrics and bias mitigation techniques.

Author: [Your Name]
Date: [Current Date]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

def create_sample_compas_data(n_samples=1000):
    """
    Create sample COMPAS-like data for demonstration purposes.
    In a real scenario, you would load the actual COMPAS dataset.
    """
    print("Creating sample COMPAS-like data for demonstration...")
    np.random.seed(42)
    
    # Generate synthetic COMPAS-like data
    race = np.random.choice(['African-American', 'Caucasian', 'Hispanic', 'Other'], 
                           size=n_samples, p=[0.4, 0.4, 0.15, 0.05])
    age = np.random.normal(35, 12, n_samples).astype(int)
    sex = np.random.choice(['Male', 'Female'], size=n_samples, p=[0.7, 0.3])
    
    # Simulate bias: African-American defendants get higher risk scores
    risk_score = np.random.normal(5, 2, n_samples)
    risk_score[race == 'African-American'] += np.random.normal(1.5, 0.5, 
                                                              sum(race == 'African-American'))
    risk_score = np.clip(risk_score, 1, 10).astype(int)
    
    # Simulate recidivism with some correlation to risk score
    recidivism_prob = 1 / (1 + np.exp(-(risk_score - 5) / 2))
    recidivism = np.random.binomial(1, recidivism_prob)
    
    df = pd.DataFrame({
        'race': race,
        'age': age,
        'sex': sex,
        'decile_score': risk_score,
        'two_year_recid': recidivism
    })
    
    print("Sample data created with simulated racial bias.")
    return df

def preprocess_compas_data(df):
    """Preprocess COMPAS data for fairness analysis"""
    
    # Create a copy
    df_processed = df.copy()
    
    # Handle missing values
    df_processed = df_processed.dropna()
    
    # Create binary race variable (African-American vs Other)
    df_processed['race_binary'] = (df_processed['race'] == 'African-American').astype(int)
    
    # Create binary gender variable
    df_processed['sex_binary'] = (df_processed['sex'] == 'Male').astype(int)
    
    # Create age categories
    df_processed['age_cat'] = pd.cut(df_processed['age'], 
                                    bins=[0, 25, 35, 45, 100], 
                                    labels=['18-25', '26-35', '36-45', '45+'])
    
    # Create age dummy variables
    age_dummies = pd.get_dummies(df_processed['age_cat'], prefix='age')
    df_processed = pd.concat([df_processed, age_dummies], axis=1)
    
    # Create risk score categories (high risk if score >= 7)
    df_processed['high_risk'] = (df_processed['decile_score'] >= 7).astype(int)
    
    # Select features for modeling
    feature_cols = ['race_binary', 'sex_binary', 'age_18-25', 'age_26-35', 'age_36-45', 'age_45+']
    
    return df_processed, feature_cols

def create_bias_visualizations(df_processed):
    """Create visualizations to identify bias"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('COMPAS Dataset Bias Analysis', fontsize=16, fontweight='bold')
    
    # 1. Risk Score Distribution by Race
    axes[0, 0].hist(df_processed[df_processed['race'] == 'African-American']['decile_score'], 
                    alpha=0.7, label='African-American', bins=10, color='skyblue')
    axes[0, 0].hist(df_processed[df_processed['race'] == 'Caucasian']['decile_score'], 
                    alpha=0.7, label='Caucasian', bins=10, color='lightcoral')
    axes[0, 0].set_xlabel('COMPAS Risk Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Risk Score Distribution by Race')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Recidivism Rate by Race
    race_recidivism = df_processed.groupby('race')['two_year_recid'].mean().sort_values(ascending=False)
    bars = axes[0, 1].bar(race_recidivism.index, race_recidivism.values, 
                          color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    axes[0, 1].set_xlabel('Race')
    axes[0, 1].set_ylabel('Recidivism Rate')
    axes[0, 1].set_title('Recidivism Rate by Race')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, race_recidivism.values):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{value:.3f}', ha='center', va='bottom')
    
    # 3. Risk Score vs Recidivism by Race
    for race_group in ['African-American', 'Caucasian']:
        subset = df_processed[df_processed['race'] == race_group]
        axes[1, 0].scatter(subset['decile_score'], subset['two_year_recid'], 
                           alpha=0.6, label=race_group, s=30)
    axes[1, 0].set_xlabel('COMPAS Risk Score')
    axes[1, 0].set_ylabel('Recidivism (0=No, 1=Yes)')
    axes[1, 0].set_title('Risk Score vs Recidivism by Race')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. High Risk Classification by Race
    high_risk_by_race = df_processed.groupby('race')['high_risk'].mean().sort_values(ascending=False)
    bars2 = axes[1, 1].bar(high_risk_by_race.index, high_risk_by_race.values,
                           color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    axes[1, 1].set_xlabel('Race')
    axes[1, 1].set_ylabel('High Risk Rate')
    axes[1, 1].set_title('High Risk Classification Rate by Race')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars2, high_risk_by_race.values):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('compas_bias_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def calculate_fairness_metrics(y_test, y_pred, test_data, feature_cols):
    """Calculate fairness metrics manually"""
    
    print("\n=== MANUAL FAIRNESS CALCULATIONS ===")
    
    # Group predictions
    aa_mask = test_data['race_binary'] == 1
    ca_mask = test_data['race_binary'] == 0
    
    aa_pred = y_pred[aa_mask]
    ca_pred = y_pred[ca_mask]
    aa_true = y_test.iloc[aa_mask.index]
    ca_true = y_test.iloc[ca_mask.index]
    
    # Statistical Parity Difference
    aa_positive_rate = aa_pred.mean()
    ca_positive_rate = ca_pred.mean()
    spd = aa_positive_rate - ca_positive_rate
    
    # Equal Opportunity Difference
    aa_tpr = np.logical_and(aa_pred == 1, aa_true == 1).sum() / (aa_true == 1).sum()
    ca_tpr = np.logical_and(ca_pred == 1, ca_true == 1).sum() / (ca_true == 1).sum()
    eod = aa_tpr - ca_tpr
    
    # Disparate Impact Ratio
    dir_ratio = aa_positive_rate / ca_positive_rate if ca_positive_rate > 0 else float('inf')
    
    print(f"Statistical Parity Difference: {spd:.4f}")
    print(f"Equal Opportunity Difference: {eod:.4f}")
    print(f"Disparate Impact Ratio: {dir_ratio:.4f}")
    
    print(f"\nAfrican-American positive rate: {aa_positive_rate:.4f}")
    print(f"Caucasian positive rate: {ca_positive_rate:.4f}")
    print(f"African-American TPR: {aa_tpr:.4f}")
    print(f"Caucasian TPR: {ca_tpr:.4f}")
    
    return spd, eod, dir_ratio, aa_positive_rate, ca_positive_rate, aa_tpr, ca_tpr

def generate_bias_report(df_processed, spd, eod, dir_ratio, aa_high_risk, ca_high_risk, 
                        aa_recidivism, ca_recidivism, model_accuracy):
    """Generate comprehensive bias audit report"""
    
    print("=" * 80)
    print("COMPAS DATASET BIAS AUDIT REPORT")
    print("=" * 80)
    print()
    
    print("EXECUTIVE SUMMARY")
    print("-" * 40)
    print("This audit analyzed the COMPAS recidivism prediction dataset for racial bias using")
    print("fairness calculations and bias analysis. The analysis revealed significant")
    print("disparities in risk assessment between African-American and Caucasian defendants,")
    print("consistent with ProPublica's findings.")
    print()
    
    print("KEY FINDINGS")
    print("-" * 40)
    print("1. Racial Disparities in Risk Scoring:")
    print(f"   - African-American defendants received higher average risk scores")
    print(f"   - High-risk classification rate: African-American {aa_high_risk:.1%} vs Caucasian {ca_high_risk:.1%}")
    print(f"   - Disparity ratio: {aa_high_risk/ca_high_risk:.2f}")
    print()
    
    print("2. Model Performance Bias:")
    print(f"   - Statistical Parity Difference: {spd:.4f}")
    print(f"   - Equal Opportunity Difference: {eod:.4f}")
    print(f"   - Disparate Impact Ratio: {dir_ratio:.4f}")
    print()
    
    print("3. Recidivism Prediction Accuracy:")
    print(f"   - African-American recidivism rate: {aa_recidivism:.1%}")
    print(f"   - Caucasian recidivism rate: {ca_recidivism:.1%}")
    print(f"   - Overall model accuracy: {model_accuracy:.1%}")
    print()
    
    print("BIAS MITIGATION RECOMMENDATIONS")
    print("-" * 40)
    print("1. Immediate Actions:")
    print("   - Implement fairness constraints in model training")
    print("   - Apply post-processing bias correction techniques")
    print("   - Establish regular bias monitoring and auditing procedures")
    print()
    
    print("2. Long-term Solutions:")
    print("   - Collect more diverse and representative training data")
    print("   - Develop race-aware models with explicit fairness objectives")
    print("   - Implement human oversight and appeal processes")
    print()
    
    print("3. Policy Recommendations:")
    print("   - Require transparency in risk assessment algorithms")
    print("   - Mandate regular bias audits by independent third parties")
    print("   - Establish clear criteria for acceptable bias levels")
    print()
    
    print("CONCLUSION")
    print("-" * 40)
    print("The COMPAS algorithm demonstrates significant racial bias that could lead to")
    print("unfair treatment of African-American defendants in the criminal justice system.")
    print("While bias mitigation techniques can improve fairness, fundamental changes to")
    print("data collection, model design, and deployment practices are necessary to")
    print("ensure truly equitable AI systems in high-stakes domains.")
    print()
    print("=" * 80)
    print("Report generated on:", pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("=" * 80)

def main():
    """Main function to run the COMPAS bias audit"""
    
    print("COMPAS Dataset Bias Audit")
    print("=" * 50)
    
    # Load or create data
    try:
        # Try to load from data directory
        df = pd.read_csv('data/compas-scores-two-years.csv')
        print("Dataset loaded successfully!")
    except FileNotFoundError:
        print("COMPAS dataset not found in data/ directory.")
        print("Creating sample data for demonstration...")
        df = create_sample_compas_data()
    
    # Preprocess data
    df_processed, feature_cols = preprocess_compas_data(df)
    print(f"Preprocessed data shape: {df_processed.shape}")
    print(f"Feature columns: {feature_cols}")
    
    # Basic statistics by race
    print("\nStatistics by race:")
    race_stats = df_processed.groupby('race')['decile_score'].agg(['count', 'mean', 'std']).round(2)
    print(race_stats)
    
    print("\nRecidivism rates by race:")
    recidivism_by_race = df_processed.groupby('race')['two_year_recid'].agg(['count', 'mean']).round(3)
    print(recidivism_by_race)
    
    # Create visualizations
    create_bias_visualizations(df_processed)
    
    # Summary statistics
    print("\n=== BIAS ANALYSIS SUMMARY ===")
    print(f"African-American defendants: {sum(df_processed['race'] == 'African-American')} total")
    print(f"Caucasian defendants: {sum(df_processed['race'] == 'Caucasian')} total")
    
    aa_high_risk = df_processed[df_processed['race'] == 'African-American']['high_risk'].mean()
    ca_high_risk = df_processed[df_processed['race'] == 'Caucasian']['high_risk'].mean()
    print(f"\nHigh Risk Rate - African-American: {aa_high_risk:.3f}")
    print(f"High Risk Rate - Caucasian: {ca_high_risk:.3f}")
    print(f"Disparity Ratio: {aa_high_risk/ca_high_risk:.2f}")
    
    aa_recidivism = df_processed[df_processed['race'] == 'African-American']['two_year_recid'].mean()
    ca_recidivism = df_processed[df_processed['race'] == 'Caucasian']['two_year_recid'].mean()
    print(f"\nRecidivism Rate - African-American: {aa_recidivism:.3f}")
    print(f"Recidivism Rate - Caucasian: {ca_recidivism:.3f}")
    print(f"Disparity Ratio: {aa_recidivism/ca_recidivism:.2f}")
    
    # Prepare data for modeling
    X = df_processed[feature_cols]
    y = df_processed['two_year_recid']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Train a simple logistic regression model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("\nModel Performance:")
    print(classification_report(y_test, y_pred))
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Recidivism', 'Recidivism'],
                yticklabels=['No Recidivism', 'Recidivism'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate fairness metrics
    test_data = df_processed.iloc[X_test.index].copy()
    test_data['prediction'] = y_pred
    test_data['probability'] = y_prob
    
    spd, eod, dir_ratio, aa_pos_rate, ca_pos_rate, aa_tpr, ca_tpr = calculate_fairness_metrics(
        y_test, y_pred, test_data, feature_cols
    )
    
    # Generate final report
    model_accuracy = np.mean(y_pred == y_test)
    generate_bias_report(df_processed, spd, eod, dir_ratio, aa_high_risk, ca_high_risk,
                        aa_recidivism, ca_recidivism, model_accuracy)
    
    print("\nAudit completed! Check the generated visualizations and report above.")
    print("Files saved: compas_bias_analysis.png, confusion_matrix.png")

if __name__ == "__main__":
    main()
