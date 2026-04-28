# mlready 🚀

An automated Exploratory Data Analysis (EDA) and data cleaning pipeline designed to get tabular medical datasets ready for machine learning instantly.

## Why use mlready?
Standard EDA tools give you numbers; `mlready` gives you clinical context. It automatically diagnoses severe class imbalances, flags biologically impossible missingness patterns, and executes a clean preprocessing recipe.

## Installation
*(Coming soon to PyPI)*
For now, clone the repository and run:
`pip install .`

## Quick Start
```python
import pandas as pd
from mlready import analyze, clean

df = pd.read_csv("your_medical_data.csv")

# 1. Diagnose the dataset
report = analyze(df, target='diagnosis')

# 2. Clean the dataset
clean_df, recipe = clean(df, target='diagnosis')