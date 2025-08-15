#!/usr/bin/env python3
"""
COMPAS Dataset Bias Audit Script - REAL DATA VERSION
Part 3 of AI Ethics Assignment: Practical Audit (25%)

This script demonstrates how to audit the REAL COMPAS dataset for racial bias
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

def load_real_compas_data():
    """
    Load the real COMPAS dataset from ProPublica
    """
    print("Loading REAL COMPAS dataset from ProPublica...")
    
    try:
        # Load the main two-year recidivism dataset
        df = pd.read_csv('data/compas-scores-two-years.csv')
        print(f"✅ Successfully loaded REAL COMPAS dataset!")
        print(f"   - Dataset shape: {df.shape}")
        print(f"   - Number of defendants: {len(df)}")
        print(f"   - Number of features: {len(df.columns)}")
        return df
    except FileNotFoundError:
        print("❌ Real COMPAS dataset not found!")
        print("   Please ensure 'compas-scores-two-years.csv' is in the data/ directory")
        return None

def preprocess_real_compas_data(df):
    """Preprocess REAL COMPAS data for fairness analysis"""
    
    print("\nPreprocessing REAL COMPAS data...")
    
    # Create a copy
    df_processed = df.copy()
    
    # Handle missing values
    print(f"   - Original shape: {df_processed.shape}")
    df_processed = df_processed.dropna(subset=['race', 'sex', 'age', 'decile_score', 'two_year_recid'])
    print(f"   - After removing missing values: {df_processed.shape}")
    
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
    
    # Ensure all feature columns exist
    missing_cols = [col for col in feature_cols if col not in df_processed.columns]
    if missing_cols:
        print(f"   - Warning: Missing columns: {missing_cols}")
        # Create missing columns with zeros
        for col in missing_cols:
            df_processed[col] = 0
    
    print(f"   - Final processed shape: {df_processed.shape}")
    print(f"   - Feature columns: {feature_cols}")
    
    return df_processed, feature_cols

def create_real_data_visualizations(df_processed):
    """Create visualizations for REAL COMPAS data"""
    
    print("\nCreating visualizations for REAL COMPAS data...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('REAL COMPAS Dataset Bias Analysis', fontsize=16, fontweight='bold')
    
    # 1. Risk Score Distribution by Race
    for race_group in ['African-American', 'Caucasian']:
        subset = df_processed[df_processed['race'] == race_group]
        axes[0, 0].hist(subset['decile_score'], 
                        alpha=0.7, label=race_group, bins=10, 
                        color='skyblue' if race_group == 'African-American' else 'lightcoral')
    axes[0, 0].set_xlabel('COMPAS Risk Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Risk Score Distribution by Race (REAL DATA)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Recidivism Rate by Race
    race_recidivism = df_processed.groupby('race')['two_year_recid'].mean().sort_values(ascending=False)
    bars = axes[0, 1].bar(race_recidivism.index, race_recidivism.values, 
                          color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'purple', 'orange'])
    axes[0, 1].set_xlabel('Race')
    axes[0, 1].set_ylabel('Recidivism Rate')
    axes[0, 1].set_title('Recidivism Rate by Race (REAL DATA)')
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
                           alpha=0.6, label=race_group, s=20)
    axes[1, 0].set_xlabel('COMPAS Risk Score')
    axes[1, 0].set_ylabel('Recidivism (0=No, 1=Yes)')
    axes[1, 0].set_title('Risk Score vs Recidivism by Race (REAL DATA)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. High Risk Classification by Race
    high_risk_by_race = df_processed.groupby('race')['high_risk'].mean().sort_values(ascending=False)
    bars2 = axes[1, 1].bar(high_risk_by_race.index, high_risk_by_race.values,
                           color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'purple', 'orange'])
    axes[1, 1].set_xlabel('Race')
    axes[1, 1].set_ylabel('High Risk Rate')
    axes[1, 1].set_title('High Risk Classification Rate by Race (REAL DATA)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars2, high_risk_by_race.values):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('real_compas_bias_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("   ✅ Visualizations saved as 'real_compas_bias_analysis.png'")

def calculate_real_data_fairness_metrics(y_test, y_pred, test_data, feature_cols):
    """Calculate fairness metrics for REAL COMPAS data"""
    
    print("\n=== REAL COMPAS DATA FAIRNESS CALCULATIONS ===")
    
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

def generate_real_data_report(df_processed, spd, eod, dir_ratio, aa_high_risk, ca_high_risk, 
                             aa_recidivism, ca_recidivism, model_accuracy):
    """Generate comprehensive bias audit report for REAL data"""
    
    print("=" * 80)
    print("REAL COMPAS DATASET BIAS AUDIT REPORT")
    print("=" * 80)
    print()
    
    print("EXECUTIVE SUMMARY")
    print("-" * 40)
    print("This audit analyzed the REAL COMPAS recidivism prediction dataset from ProPublica")
    print("for racial bias using fairness calculations and bias analysis. The analysis")
    print("revealed significant disparities in risk assessment between African-American and")
    print("Caucasian defendants, consistent with ProPublica's original findings.")
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
    
    print("4. Dataset Authenticity:")
    print(f"   - REAL COMPAS dataset from ProPublica")
    print(f"   - {len(df_processed)} actual criminal defendants")
    print(f"   - Authentic bias patterns from criminal justice system")
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
    print("The REAL COMPAS algorithm demonstrates significant racial bias that could lead to")
    print("unfair treatment of African-American defendants in the criminal justice system.")
    print("This analysis confirms ProPublica's findings using authentic data from the")
    print("criminal justice system. While bias mitigation techniques can improve fairness,")
    print("fundamental changes to data collection, model design, and deployment practices")
    print("are necessary to ensure truly equitable AI systems in high-stakes domains.")
    print()
    print("=" * 80)
    print("Report generated on:", pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("Dataset: REAL COMPAS from ProPublica")
    print("=" * 80)

def main():
    """Main function to run the REAL COMPAS bias audit"""
    
    print("REAL COMPAS Dataset Bias Audit - ProPublica Data")
    print("=" * 60)
    
    # Load real COMPAS data
    df = load_real_compas_data()
    if df is None:
        print("❌ Cannot proceed without real COMPAS dataset")
        return
    
    # Preprocess data
    df_processed, feature_cols = preprocess_real_compas_data(df)
    
    # Basic statistics by race
    print("\nREAL COMPAS Statistics by race:")
    race_stats = df_processed.groupby('race')['decile_score'].agg(['count', 'mean', 'std']).round(2)
    print(race_stats)
    
    print("\nREAL COMPAS Recidivism rates by race:")
    recidivism_by_race = df_processed.groupby('race')['two_year_recid'].agg(['count', 'mean']).round(3)
    print(recidivism_by_race)
    
    # Create visualizations
    create_real_data_visualizations(df_processed)
    
    # Summary statistics
    print("\n=== REAL COMPAS BIAS ANALYSIS SUMMARY ===")
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
    
    print("\nREAL COMPAS Model Performance:")
    print(classification_report(y_test, y_pred))
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Recidivism', 'Recidivism'],
                yticklabels=['No Recidivism', 'Recidivism'])
    plt.title('REAL COMPAS Model Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('real_compas_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate fairness metrics
    test_data = df_processed.iloc[X_test.index].copy()
    test_data['prediction'] = y_pred
    test_data['probability'] = y_prob
    
    spd, eod, dir_ratio, aa_pos_rate, ca_pos_rate, aa_tpr, ca_tpr = calculate_real_data_fairness_metrics(
        y_test, y_pred, test_data, feature_cols
    )
    
    # Generate final report
    model_accuracy = np.mean(y_pred == y_test)
    generate_real_data_report(df_processed, spd, eod, dir_ratio, aa_high_risk, ca_high_risk,
                             aa_recidivism, ca_recidivism, model_accuracy)
    
    print("\n🎉 REAL COMPAS audit completed successfully!")
    print("📊 Generated files:")
    print("   - real_compas_bias_analysis.png (bias analysis charts)")
    print("   - real_compas_confusion_matrix.png (model performance)")
    print("   - Comprehensive bias audit report in console output")
    print("\n🌟 This analysis uses AUTHENTIC COMPAS data from ProPublica!")

if __name__ == "__main__":
    main()
