import pandas as pd
import numpy as np

# --- 4. Revised Feature Engineering (One-Hot Strategy) ---
def process_features(df_train, df_test):
    print("Processing features with One-Hot Time Strategy...")
    df_train['is_train'] = 1
    df_test['is_train'] = 0
    target = df_train['total_consumption_Wh'].copy()
    
    # Combine
    cols_drop = ['total_consumption_Wh', 'Predicted', 'Id', 'locationid']
    full_df = pd.concat([df_train.drop(columns=cols_drop, errors='ignore'), df_test.drop(columns=cols_drop, errors='ignore')], axis=0, ignore_index=True)
    
    # This forces get_dummies to create specific columns for each hour (e.g., 'hour_18', 'hour_19')
    full_df['hour_cat'] = full_df['timestamp_local'].dt.hour.astype(str)
    full_df['month_cat'] = full_df['timestamp_local'].dt.month.astype(str)
    full_df['weekday_cat'] = full_df['timestamp_local'].dt.dayofweek.astype(str)
    
    # Encode Text (Now including Time!)
    obj_cols = full_df.select_dtypes(include=['object']).columns.tolist()
    full_df = pd.get_dummies(full_df, columns=obj_cols, dummy_na=True)
    
    # Clean non-numerics
    drop_cols = ['timestamp_local', 'starttime', 'endtime']
    full_df = full_df.drop(columns=[c for c in drop_cols if c in full_df.columns])
    
    # Force numeric and fill NaNs
    for col in full_df.columns:
        full_df[col] = pd.to_numeric(full_df[col], errors='coerce')
    full_df = full_df.fillna(0)
    
    # Split
    train_processed = full_df[full_df['is_train'] == 1].drop(columns=['is_train'])
    test_processed = full_df[full_df['is_train'] == 0].drop(columns=['is_train'])
    
    # Attach Target
    train_processed['total_consumption_Wh'] = target.values
    
    return train_processed.astype(float), test_processed.astype(float)

# Execute
train_ready, test_ready = process_features(train_final, test_final)
print(f"Data Reprocessed! New Train Shape: {train_ready.shape}") 
