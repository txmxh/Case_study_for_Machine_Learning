import pandas as pd
import numpy as np

print("Starting Data Pipeline Restoration...")

base_class_content = """
from abc import ABC, abstractmethod
import numpy as np

class BaseMLModel(ABC):
    def __init__(self): pass
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseMLModel': pass
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: pass
"""
with open('base_class.py', 'w') as f:
    f.write(base_class_content)

# --- 1. Load Data ---
try:
    df_train = pd.read_csv('energy_train.csv')
    df_test = pd.read_csv('submission.csv')
    homes = pd.read_csv('homes.csv')
    rooms = pd.read_csv('rooms.csv')
    persons = pd.read_csv('persons.csv')
    appliances = pd.read_csv('appliances.csv')
    other_appliances = pd.read_csv('other_appliances.csv')
    weather = pd.read_csv('weather.csv')
except FileNotFoundError:
    print("Error: CSV files not found. Please upload them again!")

# --- 2. Aggregations ---
# Persons
def parse_age_band(band):
    if not isinstance(band, str): return np.nan
    if '0-4' in band: return 2
    if '5-9' in band: return 7
    if '10-15' in band: return 12.5
    if '16-19' in band: return 17.5
    if '20-24' in band: return 22
    if '65+' in band: return 70
    if '-' in band:
        try:
            low, high = band.split('-')
            return (int(low) + int(high)) / 2
        except: return np.nan
    return np.nan

persons['age_numeric'] = persons['ageband'].apply(parse_age_band)
persons_agg = persons.groupby('homeid').agg({'personid': 'count', 'age_numeric': 'mean'}).rename(columns={'personid': 'num_residents', 'age_numeric': 'avg_resident_age'}).reset_index()

# Appliances
elec_apps = appliances[appliances['powertype'] == 'electric'].copy()
appliances_agg = elec_apps.groupby('homeid')['number'].sum().reset_index(name='num_electric_appliances')

# Other Appliances
other_appliances['number_clean'] = other_appliances['number'].apply(lambda x: int(x.replace('+', '')) if isinstance(x, str) and '+' in x else int(x))
other_agg = other_appliances.groupby('homeid')['number_clean'].sum().reset_index(name='num_other_appliances')

# Weather Pivot
weather_filtered = weather[weather['weather_type'].isin(['temperature', 'humidity', 'windspeed', 'conditions'])]
weather_pivot = weather_filtered.pivot_table(index=['date_hour', 'locationid'], columns='weather_type', values='value', aggfunc='first').reset_index()
for col in ['temperature', 'humidity', 'windspeed']:
    weather_pivot[col] = pd.to_numeric(weather_pivot[col], errors='coerce') * 0.1
weather_pivot['date_hour'] = pd.to_datetime(weather_pivot['date_hour'])

# --- 3. Merging ---
rooms_clean = rooms.rename(columns={'type': 'room_type'})

def merge_all_data(base_df, homes, rooms, persons, appliances, other, weather):
    df = base_df.copy()
    df['timestamp_local'] = pd.to_datetime(df['timestamp_local'])
    
    # Static Merges
    df = pd.merge(df, homes, on='homeid', how='left')
    df = pd.merge(df, rooms, on=['homeid', 'roomid'], how='left')
    df = pd.merge(df, persons, on='homeid', how='left')
    df = pd.merge(df, appliances, on='homeid', how='left')
    df = pd.merge(df, other, on='homeid', how='left')
    
    # Weather Merge
    df['date_hour'] = df['timestamp_local'].dt.round('h')
    df = pd.merge(df, weather, left_on=['date_hour', 'location'], right_on=['date_hour', 'locationid'], how='left')
    
    # Cleanup
    df = df.drop(columns=['locationid', 'date_hour'])
    for col in ['num_residents', 'num_electric_appliances', 'num_other_appliances']:
        if col in df.columns: df[col] = df[col].fillna(0)
    for col in ['temperature', 'humidity', 'windspeed']:
        if col in df.columns: df[col] = df[col].fillna(method='ffill')
    return df

train_final = merge_all_data(df_train, homes, rooms_clean, persons_agg, appliances_agg, other_agg, weather_pivot)
test_final = merge_all_data(df_test, homes, rooms_clean, persons_agg, appliances_agg, other_agg, weather_pivot)

# --- 4. Feature Engineering (With Top 10 Upgrade) ---
def process_features(df_train, df_test):
    df_train['is_train'] = 1
    df_test['is_train'] = 0
    target = df_train['total_consumption_Wh'].copy()
    
    # Combine
    cols_drop = ['total_consumption_Wh', 'Predicted', 'Id', 'locationid']
    full_df = pd.concat([df_train.drop(columns=cols_drop, errors='ignore'), df_test.drop(columns=cols_drop, errors='ignore')], axis=0, ignore_index=True)
    
    # --- UPGRADE: Cyclic Time Features ---
    # Instead of just "Hour 0" and "Hour 23", we map them to a circle
    full_df['hour_sin'] = np.sin(2 * np.pi * full_df['timestamp_local'].dt.hour / 24)
    full_df['hour_cos'] = np.cos(2 * np.pi * full_df['timestamp_local'].dt.hour / 24)
    full_df['month_sin'] = np.sin(2 * np.pi * full_df['timestamp_local'].dt.month / 12)
    full_df['month_cos'] = np.cos(2 * np.pi * full_df['timestamp_local'].dt.month / 12)
    
    # Encode Text
    obj_cols = full_df.select_dtypes(include=['object']).columns.tolist()
    full_df = pd.get_dummies(full_df, columns=obj_cols, dummy_na=True)
    
    # Clean non-numerics (The important fix!)
    drop_cols = ['timestamp_local', 'starttime', 'endtime']
    full_df = full_df.drop(columns=[c for c in drop_cols if c in full_df.columns])
    for col in full_df.columns:
        full_df[col] = pd.to_numeric(full_df[col], errors='coerce')
    full_df = full_df.fillna(0)
    
    # Split
    train_processed = full_df[full_df['is_train'] == 1].drop(columns=['is_train'])
    test_processed = full_df[full_df['is_train'] == 0].drop(columns=['is_train'])
    
    # Attach Target
    train_processed['total_consumption_Wh'] = target.values
    
    return train_processed.astype(float), test_processed.astype(float)

train_ready, test_ready = process_features(train_final, test_final)
print(f"Data Restored! Train Shape: {train_ready.shape}")
