import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE

def clean(df, target=None, strategy="auto"):
    """
    Intelligently cleans a dataframe using KNN imputation and SMOTE balancing.
    """
    df_clean = df.copy()
    recipe = {
        "dropped_columns": [], 
        "imputation": "None",
        "balancing": "None",
        "target": target
    }
    
    # 1. Calculate and drop > 40% missingness
    missing_percentages = (df_clean.isnull().sum() / len(df_clean)) * 100
    cols_to_drop = missing_percentages[missing_percentages > 40].index.tolist()
    
    if cols_to_drop:
        df_clean.drop(columns=cols_to_drop, inplace=True)
        recipe["dropped_columns"] = cols_to_drop
    
    # 2. KNN Imputation for remaining missing values
    # We only apply this to numerical columns to prevent string errors
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    missing_num = df_clean[num_cols].isnull().sum()
    
    if missing_num.sum() > 0:
        imputer = KNNImputer(n_neighbors=3) # Looks at the 3 most similar patients
        df_clean[num_cols] = imputer.fit_transform(df_clean[num_cols])
        recipe["imputation"] = "KNN Imputer (k=3) applied to numerical columns"

    # 3. Handle Target Imbalance with SMOTE
    if target and target in df_clean.columns:
        target_counts = df_clean[target].value_counts(normalize=True) * 100
        
        # If the majority class holds more than 75% of the data, we balance it
        if target_counts.max() > 75.0:
            X = df_clean.drop(columns=[target])
            y = df_clean[target]
            
            try:
                # Generate synthetic minority class data
                smote = SMOTE(random_state=42, k_neighbors=1) 
                X_res, y_res = smote.fit_resample(X, y)
                
                # Stitch the dataframe back together
                df_clean = pd.concat([X_res, y_res], axis=1)
                recipe["balancing"] = f"SMOTE applied. Target balanced from {target_counts.max():.1f}% majority to 50/50."
            except Exception as e:
                recipe["balancing"] = f"SMOTE skipped (ensure data is numerical). Error: {e}"

    return df_clean, recipe