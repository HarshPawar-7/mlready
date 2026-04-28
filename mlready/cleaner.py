import pandas as pd

def clean(df, target=None, strategy="auto"):
    """
    Automatically cleans a dataframe based on clinical heuristics.
    Returns the cleaned DataFrame and a dictionary recipe of applied steps.
    """
    df_clean = df.copy()
    recipe = {
        "dropped_columns": [], 
        "imputed_columns": {},
        "target": target
    }
    
    # 1. Calculate missingness
    missing_percentages = (df_clean.isnull().sum() / len(df_clean)) * 100
    
    # 2. Drop columns with > 40% missing data
    cols_to_drop = missing_percentages[missing_percentages > 40].index.tolist()
    if cols_to_drop:
        df_clean.drop(columns=cols_to_drop, inplace=True)
        recipe["dropped_columns"] = cols_to_drop
    
    # 3. Impute columns with <= 40% missing data
    cols_to_impute = missing_percentages[(missing_percentages > 0) & (missing_percentages <= 40)].index.tolist()
    for col in cols_to_impute:
        # V1: Simple median imputation
        # (We can upgrade to KNN imputation later)
        median_val = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median_val)
        recipe["imputed_columns"][col] = f"median ({median_val})"
        
    return df_clean, recipe