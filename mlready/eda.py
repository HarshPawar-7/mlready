import pandas as pd
import numpy as np

class EDAReport:
    def __init__(self, findings, warnings):
        self.findings = findings
        self.warnings = warnings

    def print_summary(self):
        print("=== mlready EDA Report ===")
        
        print("\n🚨 WARNINGS (Potential Data Issues):")
        if not self.warnings:
            print("  - None detected.")
        for w in self.warnings:
            print(f"  - {w}")
        
        print("\n📊 FINDINGS (Dataset Profile):")
        for f in self.findings:
            print(f"  - {f}")
        print("==========================\n")

def analyze(df, target=None):
    findings = []
    warnings = []
    
    # 1. Basic Shape
    findings.append(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
    
    # 2. Check Target Imbalance
    if target:
        if target not in df.columns:
            warnings.append(f"Target column '{target}' not found in dataset.")
        else:
            target_counts = df[target].value_counts(normalize=True) * 100
            
            # Format the dictionary for cleaner printing
            dist_str = ", ".join([f"'{k}': {v:.1f}%" for k, v in target_counts.items()])
            findings.append(f"Target '{target}' distribution: {dist_str}")
            
            # Heuristic: If any class is > 90%, it's severely imbalanced
            if target_counts.max() > 90:
                majority_class = target_counts.idxmax()
                warnings.append(f"Severe target imbalance detected. Class '{majority_class}' holds {target_counts.max():.1f}% of the data. Consider SMOTE or class weights before training.")

    # 3. Check Missingness
    missing_percentages = (df.isnull().sum() / len(df)) * 100
    missing_cols = missing_percentages[missing_percentages > 0]
    
    if not missing_cols.empty:
        findings.append(f"{len(missing_cols)} columns contain missing values.")
        
        for col, pct in missing_cols.items():
            # Heuristic: >40% missing is usually garbage data
            if pct > 40:
                warnings.append(f"High missingness in '{col}': {pct:.1f}%. Recommend dropping this feature entirely.")
            else:
                findings.append(f"Column '{col}' has {pct:.1f}% missing data. Recommend KNN or median imputation.")
    else:
        findings.append("No missing values detected in the dataset.")

    # Generate and return the report object
    report = EDAReport(findings, warnings)
    report.print_summary()
    return report