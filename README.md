# quickprep 🚀

An automated Exploratory Data Analysis (EDA) and data cleaning pipeline designed to get tabular medical datasets ready for machine learning instantly.

## Why use quickprep?
Standard EDA tools give you numbers; `quickprep` gives you clinical context. It automatically diagnoses severe class imbalances, flags biologically impossible missingness patterns, and executes a clean preprocessing recipe.

## Installation
```bash
pip install quickprep
```

## Quick Start
```python
import pandas as pd
from quickprep import analyze, clean

df = pd.read_csv("your_medical_data.csv")

# 1. Diagnose the dataset
report = analyze(df, target='diagnosis')

# 2. Clean the dataset
clean_df, recipe = clean(df, target='diagnosis')